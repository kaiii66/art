"""On-prem GRPO RL trainer for tau2-bench (rLLM v0.2.1.post1, verl backend).

Hydra entry point. Build:
  * the W&B run (so verl's `wandb` logger reuses it)
  * the rLLM `Dataset` (train / val), registered into the parquet store
    verl reads via `Dataset.get_verl_data_path()`
  * the `AgentTrainer` with `workflow_class=MultiTurnWorkflow`, plumbing
    in `Tau2Env` + `Tau2AssistantAgent` via `workflow_args`
  * starting LoRA discovery (local dir or W&B artifact URI -> local dir)

Run via the launcher (`onprem/scripts/run_rllm_rl.sh`):

  python -m onprem.scripts.rllm_train_tau2 \
      --config-path /workspace/configs \
      --config-name tau2_overrides

All Hydra/verl knobs (lr, n_gpus, max_prompt_length, ...) live in
`onprem/configs/tau2_overrides.yaml` (a Hydra overlay on top of rllm's
default `agent_ppo_trainer` config). Bash-side overrides still work via
`+key=value` Hydra syntax.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Make the existing repo importable (the K8s pod ships /workspace/repo on
# PYTHONPATH but a smoke `python -m onprem.scripts.rllm_train_tau2`
# from a checkout needs the explicit insert too).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("rllm_train_tau2")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_starting_lora() -> Optional[str]:
    """Return a local PEFT dir to start from, downloading from W&B if needed.

    Returns None if neither a local dir nor a W&B URI is provided -- the
    trainer will then start from the base model with a fresh LoRA.
    """
    starting_dir = os.environ.get("STARTING_LORA_DIR", "")
    if starting_dir and Path(starting_dir).is_dir():
        contents = list(Path(starting_dir).iterdir())
        if contents:
            logger.info("starting LoRA dir present locally: %s (%d entries)", starting_dir, len(contents))
            return starting_dir
        logger.warning("starting LoRA dir %s exists but is empty; trying W&B fallback", starting_dir)

    uri = os.environ.get("STARTING_LORA_URI", "")
    if not uri:
        logger.info("no starting LoRA configured -- training from scratch on top of base")
        return None

    prefix = "wandb-artifact:///"
    if not uri.startswith(prefix):
        raise RuntimeError(f"STARTING_LORA_URI must start with {prefix}; got {uri!r}")
    ref = uri[len(prefix):]
    target = Path(starting_dir or "/artifacts/sft-lora-from-wandb")
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading starting LoRA artifact %s -> %s", ref, target)
    import wandb
    api = wandb.Api()
    art = api.artifact(ref, type="lora")
    art.download(root=str(target))
    return str(target)


def _build_workflow_args(config: DictConfig, max_steps: int) -> dict:
    """Construct the workflow_args dict for AgentTrainer."""
    from onprem.scripts.tau2_rl_agent import Tau2AssistantAgent
    from onprem.scripts.tau2_rl_env import Tau2Env

    env_cfg = config.tau2.env
    agent_cfg = config.tau2.agent
    user_cfg = config.tau2.user

    env_args = {
        "domain": env_cfg.domain,
        "max_steps": max_steps,
        "user_llm": user_cfg.model,
        "user_llm_args": OmegaConf.to_container(user_cfg.llm_args, resolve=True),
        "shaped_reward_weights": OmegaConf.to_container(env_cfg.shaped_reward_weights, resolve=True),
        "max_user_errors": int(env_cfg.get("max_user_errors", 10)),
    }
    agent_args = {
        "domain": agent_cfg.domain,
        "parser_name": agent_cfg.get("parser_name", "qwen"),
    }

    return {
        "agent_cls": Tau2AssistantAgent,
        "env_cls": Tau2Env,
        "agent_args": agent_args,
        "env_args": env_args,
        "max_steps": max_steps,
        "timeout": int(config.rllm.workflow.workflow_args.get("timeout", 1_000_000)),
        "gamma": float(config.rllm.workflow.workflow_args.get("gamma", 0.0)),
        "reward_bonus_coeff": float(
            config.rllm.workflow.workflow_args.get("reward_bonus_coeff", 0.0)
        ),
    }


def _init_wandb(config: DictConfig) -> Optional[str]:
    """Init the W&B run that verl's wandb logger reuses. Returns the run name."""
    import wandb

    project = (
        os.environ.get("WANDB_PROJECT")
        or config.trainer.get("project_name", None)
        or config.tau2.get("project")
    )
    group = os.environ.get("WANDB_RUN_GROUP") or config.tau2.get("group")
    name = (
        os.environ.get("WANDB_NAME")
        or config.trainer.get("experiment_name")
        or f"rl-{os.environ.get('PIPELINE_SUFFIX', 'manual')}"
    )

    wandb.init(
        project=project,
        group=group,
        name=name,
        config=OmegaConf.to_container(config, resolve=True),
        job_type="rl-train",
    )
    return name


def _attach_lora(config: DictConfig, starting_lora: Optional[str]) -> None:
    """Wire the starting LoRA into the verl model config so verl loads it.

    verl 0.6+ exposes a single `model.lora_adapter_path` knob; the rollout
    side reads it from the actor's PEFT model directly (no separate
    rollout.lora_path / enable_lora arg). lora_rank / lora_alpha must
    already be set in the static config so peft can construct the adapter.
    """
    if not starting_lora:
        return

    OmegaConf.update(
        config, "actor_rollout_ref.model.lora_adapter_path",
        starting_lora, force_add=True,
    )
    logger.info("attached starting LoRA at %s via actor_rollout_ref.model.lora_adapter_path", starting_lora)


def _write_summary(config: DictConfig, output_dir: str, n_train: int, n_val: int) -> None:
    """Drop a small summary the K8s entrypoint can read after training."""
    summary = {
        "output_dir": output_dir,
        "domain": config.tau2.env.domain,
        "base_model": config.actor_rollout_ref.model.path,
        "n_train": n_train,
        "n_val": n_val,
        "lora_rank": config.actor_rollout_ref.model.get("lora_rank"),
        "max_steps_per_episode": config.tau2.workflow.max_steps,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rl_summary.json").write_text(json.dumps(summary, indent=2))


def _select_and_write_best_step(output_dir: str) -> Optional[int]:
    """Scan W&B run history for the best val step, then write .best_rl_step.json.

    Queries the *current* wandb run's history for validation success/reward
    metrics (tries several common key names used by verl + rLLM), finds the
    global_step with the highest value among steps that also have a saved
    lora_adapter checkpoint on disk, and writes the result to
    `output_dir/.best_rl_step.json` so run_rllm_rl.sh can upload the right
    checkpoint instead of always the last one.

    Falls back to the highest available step if W&B query fails or returns no
    matching rows (e.g. smoke run with no val pass, or all steps pruned).
    Returns the selected best step, or None if no checkpoints exist at all.
    """
    import wandb

    out = Path(output_dir)

    # Discover steps that have a saved lora_adapter on disk.
    available_steps: set[int] = set()
    for p in out.glob("global_step_*/actor/lora_adapter/adapter_config.json"):
        try:
            available_steps.add(int(p.parts[-4].split("_")[-1]))
        except (ValueError, IndexError):
            pass

    if not available_steps:
        logger.warning(
            "no lora_adapter checkpoints found under %s; skipping best-step selection",
            output_dir,
        )
        return None

    best_step: Optional[int] = None
    best_val: float = -float("inf")

    # Try to extract per-step val metrics from the current W&B run.
    # verl + rLLM can log under several key patterns depending on version.
    _VAL_METRIC_CANDIDATES = [
        "val/reward",
        "val/success",
        "val/success_rate",
        "val/mean_reward",
        "critic/val/reward",
    ]
    try:
        api = wandb.Api()
        run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
        wb_run = api.run(run_path)

        for metric_key in _VAL_METRIC_CANDIDATES:
            rows = list(wb_run.scan_history(keys=["trainer/global_step", metric_key]))
            if not rows:
                continue
            for row in rows:
                step = row.get("trainer/global_step")
                val = row.get(metric_key)
                if step is None or val is None:
                    continue
                step = int(step)
                if step not in available_steps:
                    continue
                # Higher is better; break ties by preferring the earlier step.
                if val > best_val or (val == best_val and (best_step is None or step < best_step)):
                    best_val = float(val)
                    best_step = step
            if best_step is not None:
                logger.info(
                    "best-step selected from W&B metric %r: step=%d val=%.4f",
                    metric_key, best_step, best_val,
                )
                break
        else:
            logger.warning(
                "no val metric rows found for keys %s on run %s",
                _VAL_METRIC_CANDIDATES, run_path,
            )
    except Exception as exc:
        logger.warning("W&B history query failed (%s); falling back to highest step", exc)

    if best_step is None:
        best_step = max(available_steps)
        logger.warning(
            "no val metrics matched on-disk steps; falling back to highest available step %d",
            best_step,
        )
        best_val_out: Optional[float] = None
    else:
        best_val_out = best_val

    result = {
        "best_step": best_step,
        "best_val_success": best_val_out,
        "available_steps": sorted(available_steps),
    }
    best_step_file = out / ".best_rl_step.json"
    best_step_file.write_text(json.dumps(result, indent=2))
    logger.info(
        "wrote best RL step -> %s  (step=%d, val=%s)",
        best_step_file, best_step, f"{best_val_out:.4f}" if best_val_out is not None else "N/A",
    )
    return best_step


# ---------------------------------------------------------------------------
# Hydra entry
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../configs",
    config_name="tau2_overrides",
    version_base=None,
)
def main(config: DictConfig) -> None:
    from rllm.trainer.agent_trainer import AgentTrainer
    from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow

    from onprem.scripts.tau2_rl_dataset import build_and_register

    # ----- W&B -----
    run_name = _init_wandb(config)
    logger.info("W&B run initialised: %s", run_name)

    # ----- starting LoRA (the SFT output) -----
    starting_lora = _resolve_starting_lora()
    _attach_lora(config, starting_lora)

    # ----- output dir -----
    output_dir = os.environ.get(
        "RL_OUTPUT_DIR",
        config.trainer.get("default_local_dir", "/artifacts/rl-lora"),
    )
    OmegaConf.update(config, "trainer.default_local_dir", output_dir, force_add=True)

    # ----- dataset -----
    from onprem.scripts.tau2_rl_dataset import prefilter_tasks

    domain = config.tau2.env.domain
    dataset_name = config.tau2.dataset.get("name", f"tau2-{domain}")

    train_ds, val_ds = build_and_register(
        domain=domain,
        project=config.trainer.get("project_name"),
        artifact_name=config.tau2.dataset.get("training_artifact"),
        val_artifact_name=config.tau2.dataset.get("validation_artifact"),
        val_split=config.tau2.dataset.get("validation_split", "test"),
        num_train_tasks=config.tau2.dataset.get("num_train_tasks"),
        num_val_tasks=config.tau2.dataset.get("num_val_tasks"),
        dataset_name=dataset_name,
    )
    logger.info(
        "registered datasets: train=%d, val=%s",
        len(train_ds.data),
        len(val_ds.data) if val_ds is not None else "skipped",
    )

    # ----- curriculum prefilter (optional) -----
    prefilter_cfg = config.tau2.get("prefilter", {})
    if OmegaConf.is_config(prefilter_cfg):
        prefilter_cfg = OmegaConf.to_container(prefilter_cfg, resolve=True)
    if prefilter_cfg.get("enable", False):
        keep_band_raw = prefilter_cfg.get("keep_band", [0.10, 0.90])
        filtered_rows = prefilter_tasks(
            list(train_ds.data),
            domain=domain,
            probe_llm=prefilter_cfg.get("probe_llm") or None,
            probe_llm_args=prefilter_cfg.get("probe_llm_args") or {},
            k_probe=int(prefilter_cfg.get("k_probe", 4)),
            keep_band=(float(keep_band_raw[0]), float(keep_band_raw[1])),
            env_kwargs={
                "domain": domain,
                "user_llm": config.tau2.user.model,
                "user_llm_args": OmegaConf.to_container(config.tau2.user.llm_args, resolve=True),
                "max_steps": int(config.tau2.workflow.max_steps),
            },
            cache_path=Path(output_dir) / ".prefilter_cache.json",
            concurrency=int(prefilter_cfg.get("concurrency", 8)),
        )
        if len(filtered_rows) < len(train_ds.data):
            # Re-register the filtered task list so verl reads only those tasks.
            from onprem.scripts.tau2_rl_dataset import register_dataset
            train_ds = register_dataset(dataset_name, filtered_rows, split="train")
            logger.info(
                "prefilter applied: train=%d -> %d tasks kept",
                len(filtered_rows) + (len(train_ds.data) - len(filtered_rows)),
                len(train_ds.data),
            )

    # ----- workflow_args -----
    max_steps = int(config.tau2.workflow.max_steps)
    workflow_args = _build_workflow_args(config, max_steps=max_steps)

    # Tell rllm that we are using a workflow (sets the trainer-side switch
    # in train_agent_ppo.py around the AgentWorkflowPPOTrainer branch).
    OmegaConf.update(config, "rllm.workflow.use_workflow", True, force_add=True)
    OmegaConf.update(config, "rllm.workflow.name", "multi_turn_workflow", force_add=True)
    OmegaConf.update(config, "rllm.agent.max_steps", max_steps, force_add=True)

    # ----- train -----
    trainer = AgentTrainer(
        workflow_class=MultiTurnWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        backend="verl",
    )
    # AgentTrainer only writes data.val_files when val_dataset is not None, so
    # for runs without a val split (and most smoke runs) verl falls back to
    # its built-in default path (~/data/rlhf/gsm8k/test.parquet) and crashes
    # on FileNotFoundError. Point val_files at the train parquet as a no-op
    # fallback (with val_before_train=false + a high test_freq, verl never
    # actually evaluates).
    if val_ds is None:
        OmegaConf.update(
            config, "data.val_files",
            train_ds.get_verl_data_path(), force_add=True,
        )
        # Push test_freq above total_epochs so verl never schedules a val pass.
        OmegaConf.update(config, "trainer.test_freq", 10**9, force_add=True)
        OmegaConf.update(config, "trainer.val_before_train", False, force_add=True)
        logger.info("no val dataset -> reusing train parquet for data.val_files; test_freq disabled")
    logger.info(
        "AgentTrainer ready (workflow=MultiTurnWorkflow, max_steps=%d, train=%d, val=%s)",
        max_steps,
        len(train_ds.data),
        len(val_ds.data) if val_ds is not None else "skipped",
    )
    trainer.train()

    _write_summary(
        config,
        output_dir,
        n_train=len(train_ds.data),
        n_val=len(val_ds.data) if val_ds is not None else 0,
    )

    # Select and persist the best checkpoint step based on val metrics so the
    # upload step in run_rllm_rl.sh publishes the peak checkpoint instead of
    # always the last (often post-peak / regressed) one.
    import wandb
    _select_and_write_best_step(output_dir)

    wandb.finish()
    logger.info("RL training finished. Output dir: %s", output_dir)


if __name__ == "__main__":
    main()

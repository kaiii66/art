"""
train_tau2_local.py — Local 8×H100 GRPO training for tau2-bench.

Mirrors art/train_tau2.py but uses ART's LocalBackend instead of
ServerlessBackend.  The key functional difference is that LocalBackend.train()
accepts `kl_penalty_coef` and `kl_penalty_reference_step`, which are silently
ignored by ServerlessBackend.  Wiring them here provides the KL anchor that
prevents the policy from drifting off the SFT distribution.

KL reference resolution (highest-precedence first):
  1. CLI  --kl-reference-step N
  2. .sft_endpoint_step file next to the config  (written by pull_sft_lora.py)
  3. model.get_step() at startup (== the seeded SFT checkpoint step)

Usage:
  uv run python local/train_tau2_local.py
  uv run python local/train_tau2_local.py --config pipeline_runs/05121000/train_config_local.yaml
  uv run python local/train_tau2_local.py --num-tasks 4  # smoke test
"""
import argparse
import asyncio
import json
import random
import tempfile
import yaml
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import weave
import wandb
import art
from art.local import LocalBackend

from tau2.run import get_tasks
from tau2_art_helpers import (
    Tau2TaskScenario,
    tau2_rollout,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Reuse helpers from train_tau2.py (same repo root, no package install needed)
# ---------------------------------------------------------------------------
import sys as _sys
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from train_tau2 import (
    load_training_tasks_from_artifact,
    load_validation_tasks,
)


# ─────────────────────────────────────────────────────────────────────
# Training mode
# ─────────────────────────────────────────────────────────────────────

async def run_training(
    model,
    backend: LocalBackend,
    training_tasks,
    config: dict,
    validation_tasks=None,
    best_step_file: Path | None = None,
    kl_reference_step: int | None = None,
):
    """GRPO training loop using LocalBackend with KL penalty.

    kl_reference_step: the checkpoint step used as the KL reference policy.
    When provided, LocalBackend pins the frozen reference adapter to that step
    so the loss includes beta * KL(pi_RL || pi_SFT).
    """
    if validation_tasks is None:
        validation_tasks = []

    domain = config["domain"]
    user_llm = config["user_llm"]
    user_llm_args = config.get("user_llm_args", {"temperature": 1.0, "max_tokens": 16384})
    agent_llm_args = config.get("agent_llm_args", {"temperature": 1.0, "max_tokens": 16384})
    max_steps = config.get("max_orchestrator_steps", 30)
    groups_per_step = config["groups_per_step"]
    rollouts_per_group = config["rollouts_per_group"]
    learning_rate = config["learning_rate"]
    num_epochs = config.get("num_epochs", 1)
    use_shaped = config.get("shaped_reward", False)
    shaped_weights = config.get("shaped_reward_weights")
    task_reward_blend = config.get("task_reward_blend", None)
    val_trials = int(config.get("validation_rollouts_per_task", 1))
    validation_interval = config.get("validation_step_interval", 5)
    early_stop_patience = int(config.get("early_stop_patience_evals") or 0)
    kl_coef = config.get("kl_penalty_coef", config.get("kl_beta", 0.0))

    from art.utils.iterate_dataset import iterate_dataset

    print(f"\n{'='*60}")
    print(f"LOCAL GRPO TRAINING")
    print(f"Model            : {config['base_model']}")
    print(f"Domain           : {domain}")
    print(f"Training tasks   : {len(training_tasks)}")
    if validation_tasks:
        print(f"Validation tasks : {len(validation_tasks)}")
    print(f"groups_per_step  : {groups_per_step}")
    print(f"rollouts_per_group: {rollouts_per_group}")
    print(f"learning_rate    : {learning_rate}")
    print(f"kl_penalty_coef  : {kl_coef}  (reference step: {kl_reference_step})")
    print(f"shaped_reward    : {use_shaped}")
    print(f"val_interval     : {validation_interval}")
    if early_stop_patience > 0:
        print(f"early_stop_patience: {early_stop_patience} validations without improvement")
    print(f"{'='*60}\n")

    training_scenarios = [
        Tau2TaskScenario(task_id=t.id, domain=domain) for t in training_tasks
    ]

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        initial_step=0,
    )

    best_val_reward = float("-inf")
    best_step = None
    evals_without_improvement = 0
    steps_completed = 0

    for batch in training_iterator:
        print(f"\n--- Step {batch.step} (epoch {batch.epoch}, epoch_step {batch.epoch_step}) ---")

        train_groups = []
        for scenario in batch.items:
            scenario.step = batch.step
            train_groups.append(
                art.TrajectoryGroup(
                    tau2_rollout(
                        model,
                        scenario,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        agent_llm_args=agent_llm_args,
                        max_steps=max_steps,
                        use_shaped_reward=use_shaped,
                        shaped_reward_weights=shaped_weights,
                        task_reward_blend=task_reward_blend,
                    )
                    for _ in range(rollouts_per_group)
                )
            )

        finished_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="rollouts",
            max_exceptions=rollouts_per_group * len(batch.items),
        )

        # Log training metrics before GRPO step
        all_trajs = [t for g in finished_groups for t in g.trajectories]
        if all_trajs:
            rewards = [t.reward for t in all_trajs]
            avg_reward = sum(rewards) / len(rewards)
            avg_task_reward = sum(t.metrics.get("task_reward", 0.0) for t in all_trajs) / len(all_trajs)
            avg_success = sum(t.metrics.get("success", 0.0) for t in all_trajs) / len(all_trajs)
            total_tokens = sum(t.metadata.get("completion_tokens", 0) for t in all_trajs)
            reward_std = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5

            label = "shaped" if use_shaped else "binary"
            print(f"  train reward({label})={avg_reward:.3f} (std={reward_std:.3f})  "
                  f"task_reward={avg_task_reward:.3f}  success={avg_success:.1%}  tokens={total_tokens}")

            train_metrics = {
                "train/reward": avg_reward,
                "train/task_reward": avg_task_reward,
                "train/success": avg_success,
                "train/reward_std": reward_std,
                "train/completion_tokens": total_tokens,
                "train/num_trajectories": len(all_trajs),
            }
            if use_shaped:
                for key in ("action_fraction", "termination_bonus",
                            "max_step_penalty", "repeat_message_penalty"):
                    vals = [t.metrics.get(key, -1.0) for t in all_trajs]
                    active = [v for v in vals if v >= 0]
                    if active:
                        train_metrics[f"train/{key}"] = sum(active) / len(active)
            wandb.log(train_metrics)

        # GRPO training step with KL anchor
        max_retries = 3
        train_success = False
        for attempt in range(max_retries):
            try:
                async with asyncio.timeout(1800):
                    result = await backend.train(
                        model,
                        finished_groups,
                        learning_rate=learning_rate,
                        kl_penalty_coef=kl_coef,
                        kl_penalty_reference_step=kl_reference_step,
                        save_checkpoint=config.get("save_checkpoint_artifact", True),
                    )
                    await model.log(
                        finished_groups,
                        metrics=result.metrics,
                        step=result.step,
                        split="train",
                    )
                    train_success = True
                    break
            except (asyncio.TimeoutError, Exception) as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 10
                    print(f"  Training error (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"  Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  Training FAILED after {max_retries} attempts. Skipping batch.")

        if not train_success:
            print("  Skipping validation because training failed")
            continue

        steps_completed += 1

        # Validation
        if validation_tasks and batch.step % validation_interval == 0:
            print(f"\n  Running validation on {len(validation_tasks)} tasks "
                  f"× {val_trials} trial(s)...")
            val_groups = []
            for task in validation_tasks:
                scenario = Tau2TaskScenario(step=batch.step, task_id=task.id, domain=domain)
                val_groups.append(
                    art.TrajectoryGroup([
                        tau2_rollout(
                            model,
                            scenario,
                            user_llm=user_llm,
                            user_llm_args=user_llm_args,
                            agent_llm_args=agent_llm_args,
                            max_steps=max_steps,
                        )
                        for _ in range(val_trials)
                    ])
                )
            finished_val_groups = await art.gather_trajectory_groups(
                val_groups,
                pbar_desc="validation",
                max_exceptions=len(validation_tasks) * val_trials,
            )
            await model.log(finished_val_groups, split="val")
            val_trajs = [t for g in finished_val_groups for t in g.trajectories]
            if val_trajs:
                val_reward = sum(t.reward for t in val_trajs) / len(val_trajs)
                val_success = sum(t.metrics.get("success", 0.0) for t in val_trajs) / len(val_trajs)
                val_tokens = sum(t.metadata.get("completion_tokens", 0) for t in val_trajs)
                current_step = await model.get_step()
                wandb.log({
                    "val/reward": val_reward,
                    "val/success": val_success,
                    "val/completion_tokens": val_tokens,
                    "val/num_tasks": len(val_trajs),
                    "rl/current_step": current_step,
                    "rl/current_val_reward": val_reward,
                })
                print(f"  val reward={val_reward:.3f}  success={val_success:.1%}")

                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    best_step = current_step
                    evals_without_improvement = 0
                    if best_step_file is not None:
                        try:
                            best_step_file.write_text(str(best_step))
                            print(
                                f"  [best-step] new best val/reward={best_val_reward:.3f} "
                                f"@ step {best_step}; wrote {best_step_file.name}"
                            )
                        except Exception as e:
                            print(f"  [best-step] failed to write {best_step_file}: {e}")
                    if wandb.run is not None:
                        wandb.run.summary["rl/best_step"] = best_step
                        wandb.run.summary["rl/best_val_reward"] = best_val_reward
                    wandb.log({
                        "rl/best_step": best_step,
                        "rl/best_val_reward": best_val_reward,
                    })
                else:
                    evals_without_improvement += 1
                    if early_stop_patience > 0:
                        print(
                            f"  [early-stop] {evals_without_improvement}/{early_stop_patience} "
                            f"validations without improvement (best={best_val_reward:.3f} @ step {best_step})"
                        )
                        if evals_without_improvement >= early_stop_patience:
                            print(
                                f"\n[early-stop] no improvement for {early_stop_patience} "
                                f"consecutive validations; stopping."
                            )
                            break

        # Checkpoint metadata artifact
        if config.get("save_checkpoint_artifact", True):
            current_step = await model.get_step()
            checkpoint_artifact = wandb.Artifact(
                name=f"{config['base_model'].replace('/', '-')}-checkpoint",
                type="model",
                description=f"Checkpoint at step {current_step}",
                metadata={
                    "step": current_step,
                    "epoch": batch.epoch,
                    "epoch_step": batch.epoch_step,
                    "learning_rate": learning_rate,
                    "kl_penalty_coef": kl_coef,
                    "kl_reference_step": kl_reference_step,
                    "base_model": config["base_model"],
                    "training_dataset_artifact": config.get("training_dataset_artifact"),
                    "validation_dataset_artifact": config.get("validation_dataset_artifact"),
                },
            )
            wandb.run.log_artifact(checkpoint_artifact)

    print(f"\nTraining complete. {steps_completed} steps finished.")
    if best_step is not None:
        msg = f"[rl] best step: {best_step}  best val/reward: {best_val_reward:.3f}"
        if best_step_file is not None:
            msg += f"  -> {best_step_file}"
        print(msg)
    elif validation_tasks and best_step_file is not None:
        if best_step_file.exists():
            try:
                best_step_file.unlink()
                print(f"[rl] no successful validation; removed stale {best_step_file.name}")
            except Exception as e:
                print(f"[rl] could not remove stale {best_step_file}: {e}")


# ─────────────────────────────────────────────────────────────────────
# RL pre-filter — unchanged logic from train_tau2.py
# ─────────────────────────────────────────────────────────────────────

async def prefilter_training_tasks(model, training_tasks, config):
    k_probe = int(config.get("rl_prefilter_k", 4))
    keep_band = config.get("rl_prefilter_keep_band", [0.25, 0.75])
    low, high = float(keep_band[0]), float(keep_band[1])
    domain = config["domain"]
    user_llm = config["user_llm"]
    user_llm_args = config.get("user_llm_args", {"temperature": 1.0, "max_tokens": 16384})
    agent_llm_args = config.get("agent_llm_args", {"temperature": 1.0, "max_tokens": 16384})
    max_steps = config.get("max_orchestrator_steps", 30)

    total = len(training_tasks)
    print(
        f"\n[rl-prefilter] probing {total} task(s) with k_probe={k_probe} "
        f"on the loaded policy; keeping success_rate in [{low}, {high}]"
    )

    probe_groups = []
    for task in training_tasks:
        scenario = Tau2TaskScenario(task_id=task.id, domain=domain)
        probe_groups.append(
            art.TrajectoryGroup(
                tau2_rollout(
                    model,
                    scenario,
                    user_llm=user_llm,
                    user_llm_args=user_llm_args,
                    agent_llm_args=agent_llm_args,
                    max_steps=max_steps,
                )
                for _ in range(k_probe)
            )
        )

    finished = await art.gather_trajectory_groups(
        probe_groups,
        pbar_desc="rl-prefilter",
        max_exceptions=k_probe * total,
    )

    kept_tasks = []
    for task, group in zip(training_tasks, finished):
        trajs = list(group.trajectories)
        if not trajs:
            continue
        success_rate = sum(t.metrics.get("success", 0.0) for t in trajs) / len(trajs)
        if low <= success_rate <= high:
            kept_tasks.append(task)

    kept = len(kept_tasks)
    print(
        f"[rl-prefilter] kept {kept}/{total} "
        f"({(kept / total * 100 if total else 0):.1f}%) tasks in band"
    )
    if kept == 0:
        print("[rl-prefilter] WARNING: 0 tasks in band; falling back to full set")
        return list(training_tasks)
    return kept_tasks


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

async def main(args):
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.num_tasks is not None:
        config["_num_tasks"] = args.num_tasks
    if args.groups_per_step is not None:
        config["groups_per_step"] = args.groups_per_step
    if args.rollouts_per_group is not None:
        config["rollouts_per_group"] = args.rollouts_per_group

    random.seed(config.get("random_seed", 42))

    # ── Names ──
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    kl_coef = config.get("kl_penalty_coef", config.get("kl_beta", 0.0))
    lr_str = f"{config['learning_rate']:.0e}".replace("-0", "-")
    run_name = (
        f"local-train-{config['domain']}-g{config['groups_per_step']}"
        f"-r{config['rollouts_per_group']}-lr{lr_str}-kl{kl_coef}"
        f"-{now_pt.strftime('%Y%m%d-%H%M')}"
    )
    model_name = config["model_name"]

    # ── W&B ──
    wandb.init(
        project=config["project"],
        name=run_name,
        config=config,
        job_type=config.get("wandb_job_type", "train"),
    )

    # ── LocalBackend + model ──
    # Single-GPU Unsloth (co-located trainer + vLLM on GPU 0).  The dedicated
    # multi-GPU path (trainer 0-3 + inference 7) failed because ART's vllm
    # subprocess crashes on launch in this image — needs separate debugging.
    #
    # gpu_memory_utilization=0.78 → leaves ~17 GiB of the 79 GiB H100 for
    # trainer activations.  Previous run with 0.90 OOMed once on a 44K-token
    # batch.  0.78 trades some KV-cache concurrency for stability.
    # max_model_len=49152 → fits long multi-turn tool histories.
    backend = LocalBackend()
    model = art.TrainableModel(
        name=model_name,
        project=config["project"],
        base_model=config["base_model"],
        _internal_config={
            "engine_args": {
                # Memory budget on 1×H100 (79 GiB total):
                #   model (60 GiB) + KV cache (2.5 GiB) + cuda graphs (1 GiB)
                #     = ~64 GiB → vLLM gets gpu_memory_utilization=0.82
                #   trainer = 79 - 64 = ~14 GiB → enough for 30B MoE LoRA
                #     backward pass with shaped batches up to ~24K tokens.
                # 0.78 left only 0.73 GiB for KV (too small even for 32K).
                # 0.85 left only ~12 GiB for trainer (OOMed at first batch).
                "max_model_len": 24576,
                "gpu_memory_utilization": 0.82,
            },
        },
    )
    await model.register(backend)

    starting_step = await model.get_step()
    print(
        f"LocalBackend model '{model_name}' at step {starting_step}. "
        f"RL will append step {starting_step + 1}, {starting_step + 2}, ..."
    )

    if starting_step == 0:
        print(
            "WARNING: starting_step is 0 — no seeded SFT checkpoint found in .art/. "
            "Run local/pull_sft_lora.py first."
        )

    # ── Resolve KL reference step ──
    # Priority: CLI arg > .sft_endpoint_step sidecar > starting_step
    kl_reference_step: int | None = None
    if args.kl_reference_step is not None:
        kl_reference_step = args.kl_reference_step
        print(f"KL reference step (CLI override): {kl_reference_step}")
    else:
        sft_step_file = config_path.resolve().parent / ".sft_endpoint_step"
        if sft_step_file.exists():
            try:
                kl_reference_step = int(sft_step_file.read_text().strip())
                print(f"KL reference step (from {sft_step_file.name}): {kl_reference_step}")
            except (ValueError, OSError) as e:
                print(f"WARNING: could not read {sft_step_file}: {e}")
        if kl_reference_step is None:
            kl_reference_step = starting_step
            print(f"KL reference step (from model.get_step): {kl_reference_step}")

    if kl_coef > 0 and kl_reference_step is not None:
        print(
            f"KL anchor: beta={kl_coef} against step {kl_reference_step}. "
            f"This prevents RL from drifting off the SFT distribution."
        )
    elif kl_coef == 0:
        print("WARNING: kl_penalty_coef=0. No KL anchor applied.")

    # Write sidecar files so leaderboard auto-discovery works
    (config_path.resolve().parent / ".last_trained_model").write_text(model_name)
    (config_path.resolve().parent / ".sft_endpoint_step").write_text(str(kl_reference_step))

    # ── Load tasks ──
    domain = config["domain"]
    task_split = config.get("task_split_name", "train")
    num_tasks = config.get("_num_tasks")

    training_tasks = load_training_tasks_from_artifact(config, num_tasks=num_tasks)
    if training_tasks is None:
        training_tasks = get_tasks(
            task_set_name=domain,
            task_split_name=task_split,
            num_tasks=num_tasks,
        )
        print(f"Loaded {len(training_tasks)} training tasks from {domain}/{task_split}")
    else:
        print(f"Loaded {len(training_tasks)} training tasks from W&B artifact")

    validation_tasks = load_validation_tasks(config)
    if validation_tasks:
        print(f"Loaded {len(validation_tasks)} validation tasks")

    if config.get("rl_prefilter_tasks", False) and training_tasks:
        training_tasks = await prefilter_training_tasks(model, training_tasks, config)
        print(f"RL training set size after prefilter: {len(training_tasks)} tasks")

    best_step_file = config_path.resolve().parent / ".best_rl_step"
    await run_training(
        model,
        backend,
        training_tasks,
        config,
        validation_tasks=validation_tasks,
        best_step_file=best_step_file,
        kl_reference_step=kl_reference_step,
    )

    await backend.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LocalBackend GRPO training for tau2-bench")
    parser.add_argument("--config", default="train_config_local.yaml", help="Path to YAML config")
    parser.add_argument("--num-tasks", type=int, default=None, help="Limit number of training tasks")
    parser.add_argument("--groups-per-step", type=int, default=None, help="Override groups_per_step")
    parser.add_argument("--rollouts-per-group", type=int, default=None, help="Override rollouts_per_group")
    parser.add_argument(
        "--kl-reference-step",
        type=int,
        default=None,
        help="Pin KL reference to this checkpoint step. Overrides .sft_endpoint_step auto-discovery.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
    # Clean shutdown is impossible:
    #   1. LocalBackend's `_monitor_openai_server` is a fire-and-forget task
    #      that ignores asyncio cancellation.
    #   2. vLLM EngineCore + multiprocessing.resource_tracker spawn off as
    #      grand-children whose PPID gets reparented to init when their
    #      direct parent dies — so kill-direct-children misses them.  They
    #      stay alive holding our stdout-pipe, blocking the orchestrator's
    #      `subprocess.wait()` forever.
    #
    # Drop a sentinel file the orchestrator can read after `proc.wait()`
    # returns to distinguish "killpg-clobbered exit -9 but training succeeded"
    # from a real failure.  Then killpg the whole session so all descendants
    # die at once (including the multiprocessing reparented orphans).
    import os as _os, signal as _signal
    _os.sys.stdout.flush()
    _os.sys.stderr.flush()
    cfg_path = _os.path.abspath(args.config)
    snapshot_dir = _os.path.dirname(cfg_path)
    try:
        with open(_os.path.join(snapshot_dir, ".rl_complete"), "w") as f:
            f.write("ok\n")
    except OSError:
        pass
    try:
        _os.killpg(_os.getpgrp(), _signal.SIGKILL)
    except Exception:
        pass
    _os._exit(0)

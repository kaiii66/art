"""
On-prem GRPO RL trainer for tau2-bench.

Replaces the ART ServerlessBackend GRPO loop in train_tau2.py with rLLM's
AgentTrainer (verl backend) running on a single 8xH100 K8s pod. Colocated
vLLM server (TP=8) hosts the in-training student LoRA; the user simulator
keeps hitting W&B Inference.

Inputs (env vars, set by the K8s Job):
  WANDB_API_KEY, HF_TOKEN, RLLM_API_KEY
  WANDB_PROJECT, WANDB_RUN_GROUP, WANDB_NAME, PIPELINE_SUFFIX
  STARTING_LORA_DIR        (e.g. /artifacts/sft-lora -- already merged-in PEFT dir)
  STARTING_LORA_URI        (optional fallback: wandb-artifact:///... ; downloaded if dir missing)
  RL_OUTPUT_DIR            (default: /artifacts/rl-lora)
  LORA_ARTIFACT_NAME       (e.g. tau2-rl-Qwen3-30B-A3B-Instruct-2507-04270927)
  POLICY_BASE_URL          (default: http://127.0.0.1:8000/v1 -- the colocated vLLM)
  POLICY_MODEL_NAME        (default: "policy")

CLI:
  python rllm_train_tau2.py \\
      --config /workspace/configs/rllm_train_config.yaml \\
      --starting-lora $STARTING_LORA_DIR \\
      --output-dir   $RL_OUTPUT_DIR
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import yaml
from dotenv import load_dotenv

import wandb

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_tau2 import (
    load_training_tasks_from_artifact,
    load_validation_tasks,
)
from tau2.run import get_tasks

from onprem.scripts.tau2_rllm_rollout import (
    tau2_rl_rollout,
    build_policy_client,
    build_user_client,
)

load_dotenv()
logger = logging.getLogger("rllm_train_tau2")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")


def _import_agent_trainer():
    """rLLM moves fast; try a couple of import paths for AgentTrainer."""
    try:
        from rllm.trainer import AgentTrainer  # type: ignore[import-not-found]
        return AgentTrainer
    except ImportError:
        pass
    try:
        from rllm import AgentTrainer  # type: ignore[import-not-found]
        return AgentTrainer
    except ImportError:
        pass
    raise ImportError(
        "Could not import rllm.trainer.AgentTrainer or rllm.AgentTrainer. "
        "Verify the installed rllm version matches the one this script targets."
    )


def _resolve_starting_lora(starting_lora_arg: str | None) -> str:
    """Return a local PEFT dir to start from. Download from W&B if needed."""
    starting_dir = starting_lora_arg or os.environ.get("STARTING_LORA_DIR", "")
    if starting_dir and Path(starting_dir).is_dir():
        logger.info("starting LoRA dir present locally: %s", starting_dir)
        return starting_dir

    uri = os.environ.get("STARTING_LORA_URI", "")
    if not uri:
        raise RuntimeError(
            "Neither --starting-lora / STARTING_LORA_DIR (existing dir) nor "
            "STARTING_LORA_URI is set. RL needs an SFT LoRA to initialize from."
        )
    prefix = "wandb-artifact:///"
    if not uri.startswith(prefix):
        raise RuntimeError(f"unexpected STARTING_LORA_URI shape: {uri}")
    ref = uri[len(prefix):]
    target = Path(starting_dir or "/artifacts/sft-lora-from-wandb")
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading starting LoRA artifact %s -> %s", ref, target)
    api = wandb.Api()
    art = api.artifact(ref, type="lora")
    art.download(root=str(target))
    return str(target)


def _build_train_dataset(config: dict, num_tasks: int | None) -> list[dict]:
    """Return [{task_id, domain}, ...] rows for AgentTrainer.fit."""
    tasks = load_training_tasks_from_artifact(config, num_tasks=num_tasks)
    if tasks is None:
        tasks = get_tasks(
            task_set_name=config["domain"],
            task_split_name="train",
            num_tasks=num_tasks,
        )
        logger.info("loaded %d training tasks from %s/train (fallback)", len(tasks), config["domain"])
    else:
        logger.info(
            "loaded %d training tasks from W&B artifact (%s)",
            len(tasks), config.get("training_dataset_artifact"),
        )
    return [{"task_id": t.id, "domain": config["domain"]} for t in tasks]


async def main(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # ----- W&B (the rllm 'wandb' logger reuses this run) -----
    project = os.environ.get("WANDB_PROJECT") or config["project"]
    group = os.environ.get("WANDB_RUN_GROUP") or config.get("group")
    name = os.environ.get("WANDB_NAME") or f"rl-{os.environ.get('PIPELINE_SUFFIX','manual')}"
    wandb.init(
        project=project,
        group=group,
        name=name,
        config=config,
        job_type="rl-train",
    )

    # ----- starting LoRA (the SFT output) -----
    starting_lora = _resolve_starting_lora(args.starting_lora)

    # ----- dataset -----
    train_rows = _build_train_dataset(config, num_tasks=args.num_tasks)

    # ----- tracked clients passed into every rollout -----
    policy_base_url = os.environ.get("POLICY_BASE_URL", "http://127.0.0.1:8000/v1")
    policy_model_name = os.environ.get("POLICY_MODEL_NAME", "policy")
    user_model_name = config.get("user_llm", "wandb/Qwen/Qwen3-30B-A3B-Instruct-2507")

    policy_client = build_policy_client(base_url=policy_base_url, api_key="EMPTY")
    user_client = build_user_client()  # WANDB_API_KEY from env

    # ----- bind rollout kwargs (rllm AgentTrainer takes a callable) -----
    rollout_callable = partial(
        tau2_rl_rollout,
        policy_client=policy_client,
        user_client=user_client,
        policy_model_name=policy_model_name,
        user_model_name=user_model_name,
        domain=config["domain"],
        max_steps=config.get("max_orchestrator_steps", 100),
        user_llm_args=config.get("user_llm_args"),
        agent_llm_args=config.get("agent_llm_args"),
        shaped_reward_weights=config.get("shaped_reward_weights"),
    )

    # ----- AgentTrainer -----
    AgentTrainer = _import_agent_trainer()

    output_dir = args.output_dir or os.environ.get("RL_OUTPUT_DIR", "/artifacts/rl-lora")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rollout_cfg = config.get("rollout", {})
    trainer = AgentTrainer(
        rollout=rollout_callable,
        backend="verl",
        model=config["base_model"],
        starting_lora=starting_lora,
        algorithm=config.get("algorithm", "grpo"),
        lora={
            "r":              config.get("lora_rank", 16),
            "alpha":          config.get("lora_alpha", 32),
            "target_modules": config.get("lora_target_modules", ["q_proj","k_proj","v_proj","o_proj"]),
        },
        rollouts_per_group=config.get("rollouts_per_group", 16),
        groups_per_step=config.get("groups_per_step", 4),
        learning_rate=float(config.get("learning_rate", 5e-7)),
        num_epochs=config.get("num_epochs", 1),
        rollout_server={
            "name":                  rollout_cfg.get("name", "vllm"),
            "tensor_parallel_size":  rollout_cfg.get("tensor_parallel_size", 8),
            "gpu_memory_utilization": rollout_cfg.get("gpu_memory_utilization", 0.85),
            "max_model_len":          rollout_cfg.get("max_model_len", 16384),
            "enforce_eager":          rollout_cfg.get("enforce_eager", False),
            "enable_lora":            rollout_cfg.get("enable_lora", True),
            "max_lora_rank":          rollout_cfg.get("max_lora_rank", 16),
        },
        logger=config.get("loggers", ["console", "wandb", "ui"]),
        project=project,
        output_dir=output_dir,
        seed=config.get("random_seed", 42),
    )

    logger.info(
        "AgentTrainer ready. Starting fit() with %d training tasks; "
        "groups_per_step=%s rollouts_per_group=%s lr=%s output=%s",
        len(train_rows),
        config.get("groups_per_step"),
        config.get("rollouts_per_group"),
        config.get("learning_rate"),
        output_dir,
    )

    # rllm AgentTrainer.fit may be sync or async depending on version.
    fit_result = trainer.fit(dataset=train_rows)
    if asyncio.iscoroutine(fit_result):
        await fit_result

    # ----- write a small summary so the K8s entrypoint can publish the LoRA -----
    summary = {
        "output_dir": output_dir,
        "starting_lora": starting_lora,
        "train_rows": len(train_rows),
        "config_path": str(config_path),
    }
    Path(output_dir, "rl_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("RL training finished. Output dir: %s", output_dir)

    wandb.finish()
    return 0


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to rllm_train_config.yaml.")
    parser.add_argument("--starting-lora", default=None, help="Local PEFT dir to start from (overrides STARTING_LORA_DIR).")
    parser.add_argument("--output-dir",    default=None, help="Where to write the trained LoRA (default: /artifacts/rl-lora).")
    parser.add_argument("--num-tasks", type=int, default=None, help="Cap training tasks (default: full train split).")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args)))


if __name__ == "__main__":
    _cli()

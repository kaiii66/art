"""
Phase A driver for the on-prem SFT path.

What it does:
  1. Reads the snapshot's train_distill_config.yaml (same file train_tau2_distill.py uses).
  2. Runs the existing teacher rollout flow from train_tau2_distill.py
     (W&B Inference for both teacher and user simulator) -> list[art.Trajectory].
  3. Filters successful rollouts (per `filter_successful_only`).
  4. Converts to chat-template JSONL via trajectory_to_jsonl.write_trajectories_jsonl.
  5. Uploads the JSONL as a W&B *dataset* artifact named
        `tau2-sft-trajectories-<suffix>:latest`
     so the SFT K8s pod can pull it deterministically.

It runs OUTSIDE the K8s cluster (laptop or CPU pod) — Phase A is CPU-bound;
all GPU work is the W&B Inference endpoint plus the downstream SFT pod.

Usage (called from run_pipeline.py, can also be run by hand):
    python -m onprem.scripts.prepare_sft_data \
        --config pipeline_runs/04270927/train_distill_config.yaml \
        --out-dir pipeline_runs/04270927/sft-data
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

import wandb

# Re-use the existing Phase A implementation (teacher rollouts) verbatim.
# This module is part of the repo, so we add the repo root to sys.path when
# invoked as a CLI from outside the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_tau2 import (
    load_training_tasks_from_artifact,
    load_validation_tasks,
)
from train_tau2_distill import generate_teacher_trajectories
from tau2.run import get_tasks
from onprem.scripts.trajectory_to_jsonl import write_trajectories_jsonl

load_dotenv()


def _suffix_from_group(group: str | None) -> str:
    if not group:
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")
    return group.removeprefix("pipeline-") if group.startswith("pipeline-") else group


async def main(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        config = yaml.safe_load(f)

    suffix = _suffix_from_group(config.get("group"))
    project = config["project"]
    domain = config["domain"]
    artifact_name = args.artifact_name or f"tau2-sft-trajectories-{suffix}"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else config_path.parent / "sft-data"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "sft.jsonl"
    metadata_path = out_dir / "metadata.json"

    run = wandb.init(
        project=project,
        group=config.get("group"),
        name=f"sft-prepare-{suffix}",
        config=config,
        job_type="sft-prepare",
    )

    # ---- Phase A: teacher rollouts ----
    cli_num_tasks = args.num_tasks
    training_tasks = load_training_tasks_from_artifact(config, num_tasks=cli_num_tasks)
    if training_tasks is None:
        training_tasks = get_tasks(
            task_set_name=domain,
            task_split_name="train",
            num_tasks=cli_num_tasks,
        )
        print(f"Loaded {len(training_tasks)} training tasks from {domain}/train (fallback)")
    else:
        print(
            f"Loaded {len(training_tasks)} training tasks from W&B artifact "
            f"({config.get('training_dataset_artifact')})"
        )

    teacher_trajectories = await generate_teacher_trajectories(training_tasks, config)
    if not teacher_trajectories:
        print("No teacher trajectories produced. Aborting.")
        wandb.finish(exit_code=1)
        return 1

    # ---- filter + write JSONL ----
    filter_successful = config.get("filter_successful_only", True)
    n_written = write_trajectories_jsonl(
        teacher_trajectories,
        out_path=jsonl_path,
        filter_successful_only=filter_successful,
    )
    if n_written == 0:
        print("All teacher trajectories filtered out (no successes). Aborting.")
        wandb.finish(exit_code=1)
        return 1

    metadata = {
        "suffix": suffix,
        "project": project,
        "domain": domain,
        "n_total_trajectories": len(teacher_trajectories),
        "n_written": n_written,
        "filter_successful_only": filter_successful,
        "teacher_llm": config.get("teacher_llm"),
        "user_llm": config.get("user_llm"),
        "max_orchestrator_steps": config.get("max_orchestrator_steps"),
        "source_config": str(config_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # ---- upload as W&B dataset artifact ----
    artifact = wandb.Artifact(
        name=artifact_name,
        type="sft-dataset",
        description=(
            f"Chat-template JSONL of {n_written} successful teacher rollouts "
            f"(domain={domain}, teacher={config.get('teacher_llm')}). "
            f"Consumed by the on-prem Axolotl SFT job."
        ),
        metadata=metadata,
    )
    artifact.add_file(str(jsonl_path), name="sft.jsonl")
    artifact.add_file(str(metadata_path), name="metadata.json")
    run.log_artifact(artifact)

    # ---- write deterministic URI marker the pipeline reads next ----
    uri = f"wandb-artifact:///{run.entity}/{project}/{artifact_name}:latest"
    (config_path.parent / ".sft_dataset_artifact_uri").write_text(uri)

    print()
    print(f"  wrote        : {jsonl_path}")
    print(f"  records      : {n_written}")
    print(f"  artifact     : {artifact_name}")
    print(f"  uri          : {uri}")
    print(f"  marker file  : {config_path.parent / '.sft_dataset_artifact_uri'}")
    wandb.finish()
    return 0


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to train_distill_config.yaml (typically the snapshot copy).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write sft.jsonl + metadata.json (default: <snapshot>/sft-data).",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Override the W&B dataset artifact name (default: tau2-sft-trajectories-<suffix>).",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Cap training tasks for teacher rollouts (default: full train split).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args)))


if __name__ == "__main__":
    _cli()

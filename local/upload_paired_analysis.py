"""
upload_paired_analysis.py — Stage 7 of the local RL pipeline.

Uploads the paired-task analysis report and CRN simulation JSONs to W&B as a
'paired-analysis' artifact so the results are retrievable from the run
regardless of NFS sync.

The artifact bundles:
  - The paired_analysis log (NN-paired_analysis.log) from the snapshot dir.
  - The N newest tau2cli_lb_{sft,rl}_*.json files from the simulations dir
    (one per CRN seed for each model, as written by
    create_leaderboard_shaped_reward.py).

Usage (standalone):
    uv run python local/upload_paired_analysis.py \\
        --config pipeline_runs/06021512/train_config_local.yaml \\
        --snapshot-dir pipeline_runs/06021512

Usage inside run_pipeline_local.py:
    The pipeline passes --config, --snapshot-dir, and --n automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import wandb
from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# Simulation files are written relative to create_leaderboard_shaped_reward.py
_SIM_DIR = _REPO_ROOT / "data" / "simulations"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload paired-analysis report + CRN sim JSONs to W&B"
    )
    parser.add_argument(
        "--config",
        default="train_config_local.yaml",
        help="Path to local RL config YAML (reads project / entity)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Pipeline snapshot directory containing NN-paired_analysis.log. "
             "Defaults to the directory containing --config.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of CRN seeds — picks the N newest sim files per label (default: 3)",
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=None,
        help="Override simulation directory (default: data/simulations/ under repo root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    project: str = cfg["project"]
    entity: str = cfg.get("entity", "kwt")
    suffix: str = project.split("-")[-1] if "-" in project else project

    snapshot = args.snapshot_dir or config_path.parent
    sim_dir = args.sim_dir or _SIM_DIR

    # Locate the paired-analysis log — the filename has a stage-index prefix.
    log_candidates = sorted(snapshot.glob("*paired_analysis.log"))
    if not log_candidates:
        print(
            f"[upload_paired_analysis] ERROR: no *paired_analysis.log found in {snapshot}",
            file=sys.stderr,
        )
        sys.exit(1)
    log_path = log_candidates[0]

    # Collect the N newest CRN sim files for each model label.
    def newest_sims(label: str) -> list[Path]:
        files = sorted(
            sim_dir.glob(f"tau2cli_lb_{label}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[: args.n]

    sft_sims = newest_sims("sft")
    rl_sims = newest_sims("rl")
    crn_sim_files = sft_sims + rl_sims

    print(f"[upload_paired_analysis] log      : {log_path}")
    print(f"[upload_paired_analysis] sft sims : {[f.name for f in sft_sims]}")
    print(f"[upload_paired_analysis]  rl sims : {[f.name for f in rl_sims]}")

    if args.dry_run:
        print("[upload_paired_analysis] DRY RUN: skipping actual upload")
        return

    artifact_name = f"paired-analysis-{suffix}"
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="paired-analysis",
        name=artifact_name,
        settings=wandb.Settings(silent=True),
    )
    try:
        art = wandb.Artifact(
            name=artifact_name,
            type="paired-analysis",
            description="SFT-vs-RL CRN paired-task analysis report + per-task sims",
            metadata={"crn_seeds": args.n, "source": "local-crn"},
        )
        art.add_file(str(log_path))
        for f in crn_sim_files:
            art.add_file(str(f))
        run.log_artifact(art)
        print(f"[upload_paired_analysis] artifact logged, waiting for upload...")
        art.wait()
        print(f"[upload_paired_analysis] done: {entity}/{project}/{artifact_name}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()

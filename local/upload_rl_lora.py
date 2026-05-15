"""
upload_rl_lora.py — Stage 3 of the local RL pipeline.

Uploads local RL LoRA checkpoint(s) back to the same W&B collection that the
SFT run used, so that create_leaderboard_shaped_reward.py can resolve the RL
row via ServerlessBackend / W&B Inference without any changes.

Artifact type "lora" and aliases "step{N}" / "latest" mirror exactly what
the ART serverless pipeline writes, so the existing alias-resolution in
ServerlessBackend._model_inference_name() and in the leaderboard script works
as-is.

By default only the best RL step (read from .best_rl_step) is uploaded.
Pass --upload-all to also upload every other checkpoint under .art/.../checkpoints/
(without giving them "latest").

Usage (standalone):
    uv run python local/upload_rl_lora.py --config train_config_local.yaml

Usage inside run_pipeline_local.py:
    The pipeline passes --config <snapshot>/train_config_local.yaml
    --snapshot-dir <snapshot> automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
import wandb
from dotenv import load_dotenv

load_dotenv()

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import art
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _upload_one_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    project: str,
    entity: str,
    step: int,
    extra_aliases: list[str],
    dry_run: bool,
    base_model: str,
) -> None:
    """Upload a single checkpoint directory as a W&B lora artifact."""
    aliases = [f"step{step}"] + extra_aliases
    alias_str = ", ".join(f":{a}" for a in aliases)

    if not checkpoint_dir.exists():
        print(f"[upload_rl] ERROR: checkpoint dir not found: {checkpoint_dir}")
        return

    files = list(checkpoint_dir.iterdir())
    if not files:
        print(f"[upload_rl] ERROR: checkpoint dir is empty: {checkpoint_dir}")
        return

    print(f"[upload_rl] uploading step {step} -> {entity}/{project}/{model_name} [{alias_str}]")
    print(f"[upload_rl]   from: {checkpoint_dir}  ({len(files)} files)")

    if dry_run:
        print(f"[upload_rl]   DRY RUN: skipping actual upload")
        return

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="checkpoint-upload",
        name=f"upload-local-rl-step{step}",
        settings=wandb.Settings(silent=True),
    )
    try:
        # wandb.base_model metadata is REQUIRED by W&B Inference to resolve
        # an artifact-backed LoRA — without it, the leaderboard rows get
        # `400 The model ID is invalid: model ID must be given or included
        # in artifact metadata under key "wandb.base_model"`.  Mirror what
        # ART's own deployment helper writes (art/utils/deployment/wandb.py).
        artifact = wandb.Artifact(
            name=model_name,
            type="lora",
            description=f"Local RL checkpoint at step {step}",
            metadata={
                "wandb.base_model": base_model,
                "step": step,
                "source": "local-rl",
                "checkpoint_dir": str(checkpoint_dir),
            },
            storage_region="coreweave-us",
        )
        artifact.add_dir(str(checkpoint_dir))
        run.log_artifact(artifact, aliases=aliases)
        print(f"[upload_rl]   artifact logged, waiting for upload to finish...")
        artifact.wait()
        print(f"[upload_rl]   done: {entity}/{project}/{model_name}:step{step}")
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local RL LoRA checkpoint(s) back to W&B"
    )
    parser.add_argument(
        "--config",
        default="train_config_local.yaml",
        help="Path to local RL config YAML",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Pipeline snapshot directory where .best_rl_step is read from. "
             "Defaults to the directory containing --config.",
    )
    parser.add_argument(
        "--art-path",
        default=None,
        help="Override the .art/ root directory path",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Upload a specific step instead of reading .best_rl_step",
    )
    parser.add_argument(
        "--upload-all",
        action="store_true",
        help="Upload every checkpoint step under .art/ (best step still gets :latest alias)",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity to upload to (default: read from config sft_source.entity or 'kwt')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually doing it",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    model_name = config["model_name"]
    project = config["project"]
    entity = args.entity or config.get("sft_source", {}).get("entity", "kwt")
    resolved_art_path = args.art_path or os.path.join(str(_REPO_ROOT), ".art")

    sidecar_dir = args.snapshot_dir or config_path.parent

    # ── Determine which step(s) to upload ──────────────────────────────
    local_model_stub = art.TrainableModel(
        name=model_name,
        project=project,
        base_model=config["base_model"],
    )
    model_dir = Path(get_model_dir(model=local_model_stub, art_path=resolved_art_path))
    checkpoints_dir = model_dir / "checkpoints"

    if not checkpoints_dir.exists():
        print(f"ERROR: no checkpoints directory found at {checkpoints_dir}", file=sys.stderr)
        print("Did local RL training complete successfully?", file=sys.stderr)
        sys.exit(1)

    # Resolve best step
    best_step: int | None = args.step
    if best_step is None:
        best_step_file = sidecar_dir / ".best_rl_step"
        if best_step_file.exists():
            try:
                best_step = int(best_step_file.read_text().strip())
                print(f"[upload_rl] best RL step (from {best_step_file.name}): {best_step}")
            except (ValueError, OSError) as e:
                print(f"[upload_rl] WARNING: could not read {best_step_file}: {e}")

    if best_step is None:
        # Fall back to the highest step in .art/
        available = sorted(
            int(d.name) for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if not available:
            print(f"ERROR: no checkpoint subdirs found in {checkpoints_dir}", file=sys.stderr)
            sys.exit(1)
        best_step = available[-1]
        print(f"[upload_rl] no .best_rl_step found; using latest local step: {best_step}")

    # Collect steps to upload
    if args.upload_all:
        steps_to_upload = sorted(
            int(d.name) for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        print(f"[upload_rl] --upload-all: uploading {len(steps_to_upload)} steps: {steps_to_upload}")
    else:
        steps_to_upload = [best_step]

    # ── Upload ─────────────────────────────────────────────────────────
    for step in steps_to_upload:
        checkpoint_dir = Path(get_step_checkpoint_dir(str(model_dir), step))
        extra_aliases = ["latest"] if step == best_step else []
        _upload_one_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            project=project,
            entity=entity,
            step=step,
            extra_aliases=extra_aliases,
            dry_run=args.dry_run,
            base_model=config["base_model"],
        )

    if not args.dry_run:
        print(f"\n[upload_rl] all done.")
        print(f"[upload_rl] RL artifact now at: {entity}/{project}/{model_name}:step{best_step}")
        print(f"[upload_rl] leaderboard can resolve via ServerlessBackend / W&B Inference.")
    else:
        print(f"\n[upload_rl] DRY RUN complete — nothing was uploaded.")


if __name__ == "__main__":
    main()

"""
pull_sft_lora.py — Stage 1 of the local RL pipeline.

Downloads an existing W&B SFT LoRA into the LocalBackend checkpoint directory
so that train_tau2_local.py can continue RL from that exact checkpoint.

The local checkpoint path follows ART's output_dirs convention:
    .art/{project}/models/{model_name}/checkpoints/{step:04d}/

After a successful download this script also writes two sidecar files next
to the snapshot config (or, if --snapshot-dir is not given, next to itself):
    .sft_endpoint_step   — integer step number of the downloaded checkpoint
    .last_trained_model  — model collection name (so leaderboard auto-discovers it)

Usage (standalone):
    uv run python local/pull_sft_lora.py --config train_config_local.yaml

Usage (inside run_pipeline_local.py):
    The pipeline passes --config <snapshot>/train_config_local.yaml
    --snapshot-dir <snapshot> automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Ensure the repo root (art/) is on the path when run as a script from repo root.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import art
from art.serverless.backend import ServerlessBackend
from art.utils.output_dirs import get_step_checkpoint_dir, get_model_dir


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


async def pull_sft_lora(
    config_path: Path,
    snapshot_dir: Path | None,
    art_path: str | None,
    dry_run: bool,
) -> int:
    """Download the SFT LoRA to the local .art/ directory.

    Returns the resolved SFT step number.
    """
    config = load_config(config_path)

    sft_src = config.get("sft_source", {})
    src_entity = sft_src.get("entity", "kwt")
    src_project = sft_src.get("project", config["project"])
    src_name = sft_src.get("name", config["model_name"])
    src_step = sft_src.get("step", "latest")

    model_name = config["model_name"]
    project = config["project"]
    base_model = config["base_model"]

    resolved_art_path = art_path or os.path.join(str(_REPO_ROOT), ".art")

    print(f"[pull_sft] source  : {src_entity}/{src_project}/{src_name}:{src_step}")
    print(f"[pull_sft] local   : {resolved_art_path}/{project}/models/{model_name}/")

    # ── Resolve the source model via ServerlessBackend ──────────────────
    sft_model = art.TrainableModel(
        name=src_name,
        project=src_project,
        base_model=base_model,
    )
    serverless = ServerlessBackend()
    await sft_model.register(serverless)

    # Resolve "latest" to a concrete step number for determinism
    if src_step == "latest" or src_step is None:
        resolved_step = await sft_model.get_step()
    else:
        resolved_step = int(src_step)

    print(f"[pull_sft] resolved step : {resolved_step}")

    # ── Build the target checkpoint directory ───────────────────────────
    # For LocalBackend the model dir is .art/{project}/models/{model_name}
    # and the step subdir is checkpoints/{step:04d}.
    #
    # We construct a temporary Model with the LOCAL project/name so that
    # get_model_dir returns the right path.
    local_model_stub = art.TrainableModel(
        name=model_name,
        project=project,
        base_model=base_model,
    )
    model_dir = get_model_dir(model=local_model_stub, art_path=resolved_art_path)
    target_dir = get_step_checkpoint_dir(model_dir, resolved_step)

    # ── Idempotency check ───────────────────────────────────────────────
    sentinel = Path(target_dir) / "adapter_model.safetensors"
    if sentinel.exists():
        print(f"[pull_sft] checkpoint already present at {target_dir}; skipping download")
    else:
        if dry_run:
            print(f"[pull_sft] DRY RUN: would download step {resolved_step} to {target_dir}")
        else:
            print(f"[pull_sft] downloading step {resolved_step} to {target_dir} ...")
            os.makedirs(target_dir, exist_ok=True)
            await serverless._experimental_pull_model_checkpoint(
                sft_model,
                step=resolved_step,
                local_path=target_dir,
                verbose=True,
            )
            print(f"[pull_sft] download complete: {target_dir}")

    # ── Write sidecar files ─────────────────────────────────────────────
    sidecar_dir = snapshot_dir or config_path.resolve().parent
    sft_step_file = sidecar_dir / ".sft_endpoint_step"
    last_model_file = sidecar_dir / ".last_trained_model"

    if not dry_run:
        sft_step_file.write_text(str(resolved_step))
        print(f"[pull_sft] wrote {sft_step_file}")

        last_model_file.write_text(model_name)
        print(f"[pull_sft] wrote {last_model_file}")
    else:
        print(f"[pull_sft] DRY RUN: would write {sft_step_file} = {resolved_step}")
        print(f"[pull_sft] DRY RUN: would write {last_model_file} = {model_name}")

    await serverless.close()
    return resolved_step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download W&B SFT LoRA into local .art/ checkpoint directory"
    )
    parser.add_argument(
        "--config",
        default="train_config_local.yaml",
        help="Path to local RL config YAML (default: train_config_local.yaml)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Pipeline snapshot directory where sidecar files are written. "
             "Defaults to the directory containing --config.",
    )
    parser.add_argument(
        "--art-path",
        default=None,
        help="Override the .art/ root directory path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without downloading anything",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    snapshot_dir = args.snapshot_dir
    if snapshot_dir is not None:
        snapshot_dir = snapshot_dir.resolve()

    step = asyncio.run(
        pull_sft_lora(
            config_path=config_path,
            snapshot_dir=snapshot_dir,
            art_path=args.art_path,
            dry_run=args.dry_run,
        )
    )
    print(f"[pull_sft] done. SFT step = {step}")


if __name__ == "__main__":
    main()

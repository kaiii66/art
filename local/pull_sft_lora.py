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

    # ── Resolve which checkpoint step to download ───────────────────────
    # Precedence (highest first):
    #   1. sft_source.step is a concrete integer (or numeric string)
    #      — use as-is. This is what run_full_pipeline.py bakes into the
    #      docker config after reading .best_sft_step on the SFT host.
    #   2. .best_sft_step sidecar in --snapshot-dir (or config dir)
    #      — written by train_tau2_distill.py at the best val/reward chunk.
    #      Prefer this over any non-integer src_step so a manual
    #      `step: latest` in the config never silently overrides the best
    #      checkpoint when both are available.
    #   3. sft_source.step == "best"
    #      — explicit opt-in to .best_sft_step. Errors loudly if the
    #      sidecar is missing rather than falling back to :latest.
    #   4. sft_source.step == "latest" / None / anything else
    #      — old behaviour: resolve via the W&B collection's :latest alias.
    sidecar_dir = snapshot_dir or config_path.resolve().parent
    best_file = sidecar_dir / ".best_sft_step"

    src_step_is_int = isinstance(src_step, int) or (
        isinstance(src_step, str) and src_step.isdigit()
    )

    if src_step_is_int:
        resolved_step = int(src_step)
        print(f"[pull_sft] using config-pinned step {resolved_step}")
    elif best_file.exists():
        try:
            resolved_step = int(best_file.read_text().strip())
        except (ValueError, OSError) as e:
            raise RuntimeError(
                f"[pull_sft] {best_file} exists but is not an integer: {e}"
            )
        print(
            f"[pull_sft] using .best_sft_step = {resolved_step} "
            f"(overrides sft_source.step={src_step!r})"
        )
    elif src_step == "best":
        raise RuntimeError(
            f"sft_source.step='best' but {best_file} not found. "
            "Run SFT first (writes .best_sft_step) or copy the sidecar into "
            f"{sidecar_dir} before pulling."
        )
    else:
        resolved_step = await sft_model.get_step()
        print(f"[pull_sft] using collection :latest = {resolved_step}")

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

    # ── Write sidecar files (sidecar_dir already resolved above) ────────
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

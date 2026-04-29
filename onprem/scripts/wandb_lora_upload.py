"""
Publish a PEFT LoRA directory as a W&B Inference-compatible `type=lora`
artifact, then write its `wandb-artifact:///...` URI to a marker file the
pipeline driver picks up.

Per https://docs.wandb.ai/inference/lora the artifact MUST:
  - have type=='lora'
  - carry metadata['wandb.base_model'] == the exact base model id
  - live in storage_region='coreweave-us'
  - contain the PEFT save_pretrained directory (adapter_config.json,
    adapter_model.safetensors, tokenizer files, etc.)

Usage (from inside the SFT or RL pod):
    python wandb_lora_upload.py \\
        --lora-dir       /artifacts/sft-lora \\
        --artifact-name  tau2-sft-Qwen3-30B-A3B-Instruct-2507-04270927 \\
        --base-model     Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --project        tau2-ART-autoresearch-telecom \\
        --group          pipeline-04270927 \\
        --job-type       sft-upload \\
        --uri-out        /artifacts/.sft_lora_artifact_uri-04270927
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import wandb


def _validate_lora_dir(p: Path) -> None:
    """Spot-check the PEFT contract before we waste an upload."""
    if not p.is_dir():
        raise FileNotFoundError(f"LoRA dir not found: {p}")
    expected = ["adapter_config.json"]
    missing = [name for name in expected if not (p / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"LoRA dir {p} is missing expected files: {missing}. "
            "Was Axolotl / verl trained with adapter=lora and save_pretrained?"
        )
    has_safetensors = any(p.glob("adapter_model*.safetensors"))
    has_bin = any(p.glob("adapter_model*.bin"))
    if not (has_safetensors or has_bin):
        raise FileNotFoundError(
            f"LoRA dir {p} has no adapter_model weights (.safetensors or .bin)."
        )


def upload_lora(
    *,
    lora_dir: str | Path,
    artifact_name: str,
    base_model: str,
    project: str,
    group: str | None = None,
    job_type: str = "lora-upload",
    storage_region: str = "coreweave-us",
    description: str | None = None,
    uri_out: str | Path | None = None,
) -> str:
    """Returns the wandb-artifact:/// URI of the new artifact version."""
    lora_dir = Path(lora_dir)
    _validate_lora_dir(lora_dir)

    run = wandb.init(
        project=project,
        group=group,
        name=f"{job_type}-{artifact_name}",
        job_type=job_type,
        config={
            "lora_dir": str(lora_dir),
            "artifact_name": artifact_name,
            "base_model": base_model,
            "storage_region": storage_region,
        },
    )
    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="lora",
            description=(description or
                         f"PEFT LoRA over {base_model} for tau2 (W&B Inference servable)."),
            metadata={
                "wandb.base_model": base_model,
            },
            # `storage_region` is honored by recent wandb versions; older ones
            # fall back to the project's default region (still uploads, just
            # may be slower for serving). Keep this kwarg in any case.
            storage_region=storage_region,
        )
        artifact.add_dir(str(lora_dir))
        run.log_artifact(artifact)
        artifact.wait()  # block until W&B has registered the new version

        version = artifact.version  # e.g. "v3"
        entity = run.entity
        uri = f"wandb-artifact:///{entity}/{project}/{artifact_name}:{version}"

        # Also record the URI as a run summary key so it's discoverable from
        # the W&B run UI in addition to the marker file.
        run.summary["lora_artifact_uri"] = uri
        run.summary["lora_artifact_version"] = version

        print(f"  uploaded: {uri}")
    finally:
        wandb.finish()

    if uri_out is not None:
        out = Path(uri_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(uri)
        print(f"  marker  : {out}")

    return uri


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-dir",       required=True)
    parser.add_argument("--artifact-name",  required=True)
    parser.add_argument("--base-model",     required=True)
    parser.add_argument("--project",        required=True)
    parser.add_argument("--group",          default=None)
    parser.add_argument("--job-type",       default="lora-upload")
    parser.add_argument("--storage-region", default="coreweave-us")
    parser.add_argument("--description",    default=None)
    parser.add_argument("--uri-out",        default=None,
                        help="Path to write the wandb-artifact:/// URI for downstream stages.")
    args = parser.parse_args()
    try:
        upload_lora(
            lora_dir=args.lora_dir,
            artifact_name=args.artifact_name,
            base_model=args.base_model,
            project=args.project,
            group=args.group,
            job_type=args.job_type,
            storage_region=args.storage_region,
            description=args.description,
            uri_out=args.uri_out,
        )
    except Exception as e:
        print(f"upload failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

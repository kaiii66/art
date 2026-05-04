"""Merge an SFT PEFT adapter into the Qwen3 base model and save the merged
weights under a local dir we can hand to verl as `actor_rollout_ref.model.path`.

Why we need this even though verl supports `lora_adapter_path`:
  vLLM 0.11's PackedLoRA dummy-run path (used during KV-cache profiling) crashes
  with `AttributeError: 'NoneType' object has no attribute 'shape'` when the
  adapter targets the q/k/v_proj split underlying Qwen3-MoE's fused qkv_proj
  (and `target_modules="all-linear"` doesn't fix it because `gate_up_proj`
  exhibits the same packed-LoRA dummy issue). Merging the SFT adapter into
  the base bf16 weights up-front lets vLLM treat the model as plain weights;
  verl trains a fresh LoRA on top during RL.

Run via:
  python -m onprem.scripts.merge_sft_lora \
      --base Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --adapter /artifacts/sft-lora \
      --output /artifacts/sft-merged
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("merge_sft_lora")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="HF base model id or local dir")
    parser.add_argument("--adapter", required=True, help="Local PEFT adapter dir")
    parser.add_argument("--output", required=True, help="Output dir for merged weights")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing output dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")

    out = Path(args.output)
    if out.exists():
        if any(out.iterdir()):
            if not args.force:
                logger.info("output dir %s already populated; skipping merge", out)
                return
            logger.info("--force set; removing existing %s", out)
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    logger.info("loading base model %s (%s)", args.base, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",   # do the merge on CPU; the vLLM workers will place
                            # the merged weights on GPU later.
        trust_remote_code=False,
    )

    logger.info("loading PEFT adapter from %s", args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter, device_map="cpu")

    logger.info("merging adapter into base")
    model = model.merge_and_unload()

    logger.info("writing merged model to %s", out)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    tokenizer.save_pretrained(str(out))

    logger.info("done. files in %s:", out)
    for p in sorted(out.iterdir()):
        size = p.stat().st_size
        logger.info("  %s (%.1f MiB)", p.name, size / 1024 / 1024)


if __name__ == "__main__":
    sys.exit(main())

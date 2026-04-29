#!/usr/bin/env bash
# Entrypoint for the SFT pod.
#
# Steps:
#   1. Sanity checks (GPUs visible, dataset URI present).
#   2. Pull the JSONL dataset artifact (W&B) into /data.
#   3. Launch Axolotl across all 8 GPUs via `accelerate launch`.
#   4. On success, upload the resulting LoRA dir to W&B as a `type=lora`
#      W&B Inference artifact and write the URI to the artifacts PVC for the
#      pipeline driver to pick up.
#
# Required env (passed in via the K8s Job manifest):
#   WANDB_API_KEY            (Secret)
#   HF_TOKEN                 (Secret)
#   SFT_DATASET_ARTIFACT_URI = wandb-artifact:///<team>/<project>/<name>:<alias>
#   WANDB_PROJECT
#   WANDB_RUN_GROUP
#   WANDB_NAME
#   PIPELINE_SUFFIX          (e.g. 04270927)
#   LORA_ARTIFACT_NAME       (e.g. tau2-sft-Qwen3-30B-A3B-Instruct-2507-04270927)
set -euo pipefail

log()  { echo "[$(date -u +%FT%TZ)] [run_axolotl_sft] $*"; }
fail() { echo "[$(date -u +%FT%TZ)] [run_axolotl_sft] FATAL: $*" >&2; exit 1; }

# ----- 1. sanity -----
log "GPU check:"
nvidia-smi -L || fail "no GPUs visible"
test -n "${WANDB_API_KEY:-}"            || fail "WANDB_API_KEY unset"
test -n "${SFT_DATASET_ARTIFACT_URI:-}" || fail "SFT_DATASET_ARTIFACT_URI unset"
test -n "${LORA_ARTIFACT_NAME:-}"       || fail "LORA_ARTIFACT_NAME unset"
test -n "${WANDB_PROJECT:-}"            || fail "WANDB_PROJECT unset"
test -n "${PIPELINE_SUFFIX:-}"          || fail "PIPELINE_SUFFIX unset"

mkdir -p /data /artifacts

# ----- 2. pull the dataset artifact -----
log "downloading dataset artifact: $SFT_DATASET_ARTIFACT_URI"
python - <<PY
import os, sys, wandb
uri = os.environ["SFT_DATASET_ARTIFACT_URI"]
prefix = "wandb-artifact:///"
if not uri.startswith(prefix):
    sys.exit(f"unexpected URI shape: {uri}")
ref = uri[len(prefix):]   # <team>/<project>/<name>:<alias>
api = wandb.Api()
art = api.artifact(ref, type="sft-dataset")
out = art.download(root="/data")
print(f"downloaded -> {out}")
PY

ls -lh /data/sft.jsonl || fail "/data/sft.jsonl not present after download"

# ----- 3. run Axolotl across 8 GPUs -----
log "launching Axolotl on 8 GPUs"
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    -m axolotl.cli.train /workspace/configs/axolotl_sft.yaml

# ----- 4. publish the LoRA to W&B Inference -----
log "uploading LoRA to W&B Inference: $LORA_ARTIFACT_NAME"
python /workspace/scripts/wandb_lora_upload.py \
    --lora-dir /artifacts/sft-lora \
    --artifact-name "$LORA_ARTIFACT_NAME" \
    --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --project   "$WANDB_PROJECT" \
    --group     "$WANDB_RUN_GROUP" \
    --job-type  "sft-upload" \
    --uri-out   "/artifacts/.sft_lora_artifact_uri-${PIPELINE_SUFFIX}"

log "done. SFT LoRA artifact URI written to /artifacts/.sft_lora_artifact_uri-${PIPELINE_SUFFIX}"

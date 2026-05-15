#!/usr/bin/env bash
# Entrypoint for the tau2-art k8s Job (openpipe-art LocalBackend RL).
#
# Environment variables consumed (all injected by the Job YAML):
#   PIPELINE_SUFFIX   MMDDHHMM identifier; forwarded to run_pipeline_local.py
#                     so each k8s run gets its own isolated snapshot dir +
#                     W&B artifact collection.
#   SKIP_STAGES       Optional space-separated stage names to skip
#                     (e.g. "leaderboard" or "upload_rl leaderboard")
#   NUM_TASKS         Optional integer; if set, forwarded to RL stage as
#                     --num-tasks (smoke tests use NUM_TASKS=4).
#   WANDB_API_KEY     Injected from k8s secret 'wandb' (key: api)
#   HF_TOKEN          Injected from k8s secret 'hf'    (key: token)
#   HF_HOME           Persistent HF model cache (set to /artifacts/hf-cache)
#   HUGGINGFACE_HUB_CACHE  Persistent hub cache dir

set -euo pipefail

export PATH="/root/.local/bin:/workspace/.venv/bin:$PATH"

# Persist debug logs (vllm-dedicated, etc.) to the artifacts PVC so they
# survive pod death.  This is essential when the vLLM subprocess crashes
# during init: its log lives at .art/.../logs/vllm-dedicated.log inside the
# container and is otherwise lost when k8s tears the pod down.
ART_DEBUG_DIR="/artifacts/debug/${PIPELINE_SUFFIX:-unknown}"
mkdir -p "$ART_DEBUG_DIR"
trap 'echo "[trap] copying .art logs + pipeline_runs to $ART_DEBUG_DIR"; \
      cp -rv /workspace/.art/*/models/*/logs "$ART_DEBUG_DIR/" 2>/dev/null || true; \
      cp -rv /workspace/pipeline_runs/${PIPELINE_SUFFIX:-unknown} "$ART_DEBUG_DIR/" 2>/dev/null || true' EXIT

# ── Optional: use persistent HF cache from the PVC ──────────────────────────
if [[ -n "${HF_HOME:-}" ]]; then
    mkdir -p "${HF_HOME}/hub"
fi

echo "=== tau2-art RL pipeline starting ==="
echo "    PIPELINE_SUFFIX : ${PIPELINE_SUFFIX:-<auto>}"
echo "    WANDB_PROJECT   : ${WANDB_PROJECT:-<from config>}"
echo "    SKIP_STAGES     : ${SKIP_STAGES:-<none>}"

cd /workspace

# Build --skip argument if SKIP_STAGES is set
SKIP_ARGS=()
if [[ -n "${SKIP_STAGES:-}" ]]; then
    read -ra _stages <<< "$SKIP_STAGES"
    SKIP_ARGS=(--skip "${_stages[@]}")
fi

# Build --num-tasks argument if NUM_TASKS is set (smoke tests)
NUM_TASKS_ARGS=()
if [[ -n "${NUM_TASKS:-}" ]]; then
    NUM_TASKS_ARGS=(--num-tasks "$NUM_TASKS")
fi

# Use the venv's python directly — `uv run` re-installs the local project on
# every invocation and during that step uv resets dependency versions to match
# uv.lock, which clobbers the huggingface-hub<1.0 / gql>=4 pins baked into the
# image (causing transformers / weave import failures at runtime).
exec python local/run_pipeline_local.py \
    ${PIPELINE_SUFFIX:+--project-suffix "$PIPELINE_SUFFIX"} \
    "${SKIP_ARGS[@]}" \
    "${NUM_TASKS_ARGS[@]}"

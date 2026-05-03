#!/usr/bin/env bash
# Entrypoint for the RL pod.
#
# Steps:
#   1. Sanity checks (GPUs, secrets, starting LoRA).
#   2. Run rllm_train_tau2.py via Hydra.  The verl backend internally manages
#      vLLM (colocated TP=8 on the 8 H100s) and FSDP actor sharding.
#   3. On success, upload the resulting LoRA dir to W&B Inference and write
#      its URI to the artifacts PVC for the pipeline driver to read.
#
# Required env (passed in via the K8s Job manifest):
#   WANDB_API_KEY            (Secret/wandb)
#   HF_TOKEN                 (Secret/hf)
#   RLLM_API_KEY             (Secret/rllmui, optional)
#   STARTING_LORA_URI        wandb-artifact:///team/proj/sft-lora-name:alias
#   STARTING_LORA_DIR        local cache dir (default /artifacts/sft-lora)
#   WANDB_PROJECT, WANDB_RUN_GROUP, WANDB_NAME, PIPELINE_SUFFIX
#   LORA_ARTIFACT_NAME       e.g. tau2-rl-Qwen3-30B-A3B-Instruct-2507-04270927
#   RL_OUTPUT_DIR            default /artifacts/rl-lora
#
# Optional smoke-test knobs (override Hydra config from the K8s spec):
#   RL_HYDRA_OVERRIDES       extra Hydra k=v overrides, e.g.
#                            "tau2.dataset.num_train_tasks=2 tau2.workflow.max_steps=4
#                             actor_rollout_ref.rollout.n=2 trainer.total_epochs=1"
set -euo pipefail

log()  { echo "[$(date -u +%FT%TZ)] [run_rllm_rl] $*"; }
fail() { echo "[$(date -u +%FT%TZ)] [run_rllm_rl] FATAL: $*" >&2; exit 1; }

# ----- 1. sanity -----
log "GPU check:"
nvidia-smi -L || fail "no GPUs visible"
test -n "${WANDB_API_KEY:-}"      || fail "WANDB_API_KEY unset"
test -n "${RLLM_API_KEY:-}"       || log "WARN: RLLM_API_KEY unset; rllm-ui traces will be skipped"
test -n "${WANDB_PROJECT:-}"      || fail "WANDB_PROJECT unset"
test -n "${PIPELINE_SUFFIX:-}"    || fail "PIPELINE_SUFFIX unset"
test -n "${LORA_ARTIFACT_NAME:-}" || fail "LORA_ARTIFACT_NAME unset"

export RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-/artifacts/rl-lora}"
export STARTING_LORA_DIR="${STARTING_LORA_DIR:-/artifacts/sft-lora}"

mkdir -p "$RL_OUTPUT_DIR"

# ----- 2. run training (Hydra entry point) -----
log "starting rllm_train_tau2 (Hydra: tau2_overrides)"
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Hydra overrides forwarded from the env (smoke vs. prod runs differ here).
# RL_HYDRA_OVERRIDES is split on whitespace by the shell; quoted args inside
# the env variable are passed through verbatim.
read -r -a EXTRA_OVERRIDES <<< "${RL_HYDRA_OVERRIDES:-}"

cd /workspace/repo

python -m onprem.scripts.rllm_train_tau2 \
    --config-path /workspace/configs \
    --config-name tau2_overrides \
    "trainer.default_local_dir=$RL_OUTPUT_DIR" \
    "trainer.experiment_name=rl-${PIPELINE_SUFFIX}" \
    "${EXTRA_OVERRIDES[@]}"

# ----- 3. publish the LoRA to W&B Inference -----
log "uploading RL LoRA to W&B Inference: $LORA_ARTIFACT_NAME"
python /workspace/scripts/wandb_lora_upload.py \
    --lora-dir "$RL_OUTPUT_DIR" \
    --artifact-name "$LORA_ARTIFACT_NAME" \
    --base-model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --project   "$WANDB_PROJECT" \
    --group     "$WANDB_RUN_GROUP" \
    --job-type  "rl-upload" \
    --uri-out   "/artifacts/.rl_lora_artifact_uri-${PIPELINE_SUFFIX}"

log "done. RL LoRA artifact URI written to /artifacts/.rl_lora_artifact_uri-${PIPELINE_SUFFIX}"

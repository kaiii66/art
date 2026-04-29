# RL trainer container.
#
# Base : nvcr.io/nvidia/pytorch:24.10-py3 (CUDA 12.6, PyTorch 2.5, NCCL, cuBLAS)
# Adds : vLLM (rollout serving with hot-swap LoRA), FlashAttention,
#        verl (gradient engine), rllm (agent-RL wrapper), tau2 + this repo
#        (so the @rllm.rollout function can drive the tau2 orchestrator).
#
# Hardware target: H100 (sm_90). FlashAttention is built for sm_90.
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 \
#     -t ghcr.io/<gh-user>/tau2-rllm:$(git rev-parse --short HEAD) \
#     -f onprem/docker/Dockerfile.rl --push .

# ----- base layer (heavy GPU deps, cached across iterations) -----
FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# vLLM + FlashAttention pinned to versions known to work with sm_90 + Qwen3 MoE.
# Bump deliberately, not on every build.
RUN pip install --no-cache-dir \
        "vllm>=0.7.0,<0.9.0"

RUN pip install --no-cache-dir \
        "flash-attn>=2.7.0" --no-build-isolation

# verl (gradient engine for GRPO) and rLLM (agent-RL wrapper).
# Both move fast; if you need to pin to a known-good revision do it here.
RUN pip install --no-cache-dir \
        "verl>=0.4.0" \
        "rllm>=0.1.0"

# Standard helpers used by the trainer + upload scripts.
RUN pip install --no-cache-dir \
        wandb>=0.18 \
        peft>=0.14 \
        python-dotenv>=1.0 \
        hf_transfer>=0.1.8 \
        litellm>=1.65.0

# ----- app layer (your repo + tau2; rebuilt on code changes) -----
FROM base AS app

WORKDIR /workspace/repo

# Install the tau2 package + repo deps from pyproject.toml. Use --no-deps in a
# follow-up if you want to keep this layer thin; for now full install.
COPY pyproject.toml pdm.lock README.md /workspace/repo/
COPY src/ /workspace/repo/src/
RUN pip install --no-cache-dir -e .

# The pieces of the existing repo the rollout needs at runtime: the helpers,
# the configs, the scripts. Code-only; no data.
COPY tau2_art_helpers.py /workspace/repo/
COPY train_tau2.py /workspace/repo/
COPY train_tau2_distill.py /workspace/repo/
COPY onprem/scripts/ /workspace/scripts/
COPY onprem/configs/ /workspace/configs/

RUN chmod +x /workspace/scripts/*.sh

ENV PATH=/workspace/scripts:$PATH \
    PYTHONPATH=/workspace/repo:$PYTHONPATH

# Default command: the RL entrypoint script.  K8s Job command can override.
CMD ["/workspace/scripts/run_rllm_rl.sh"]

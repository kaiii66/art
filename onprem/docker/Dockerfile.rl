# RL trainer container.
#
# Base : nvcr.io/nvidia/pytorch:25.08-py3 (CUDA 12.x, PyTorch 2.7+, Ubuntu
#        24.04, Python 3.12). 25.08 is the first NGC PyTorch container with
#        Python 3.12; we previously tried 24.10 (Python 3.10) and hit the
#        2026-ecosystem-wide Python>=3.11 floor in scikit-learn>=1.8 and
#        pandas>=3.0 (both enforced inside meson.build, so
#        --ignore-requires-python doesn't help).
# Adds : vLLM (rollout serving with hot-swap LoRA), FlashAttention,
#        rllm[verl] (gradient engine + agent-RL wrapper), tau2 + this repo
#        (so the @rllm.rollout function can drive the tau2 orchestrator).
#
# Hardware target: H100 (sm_90). FlashAttention is built for sm_90.
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 \
#     -t ghcr.io/<gh-user>/tau2-rllm:$(git rev-parse --short HEAD) \
#     -f onprem/docker/Dockerfile.rl --push .

# ----- base layer (heavy GPU deps, cached across iterations) -----
FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging ninja

# rllm[verl] v0.2.1 pulls vllm==0.11.0 and flash-attn>=2.8.1 (with their
# own torch==2.11.0 pin). flash-attn's setup.py imports torch at build
# time, so we install it FIRST with --no-build-isolation so it can see
# the NGC base's torch. Then `rllm[verl]` sees flash-attn already
# satisfied and skips the rebuild. Pinned to 2.8.3 to match what the
# rllm v0.2.1 resolver picked previously; bump deliberately.
RUN pip install --no-cache-dir --no-build-isolation "flash-attn==2.8.3"

# rLLM (agent-RL wrapper) + verl (gradient engine for GRPO). rLLM is
# GitHub-only (no PyPI package) and ships verl + vllm as the [verl] extra
# so we install both with one git+ URL. Pinned to v0.2.1.post1 (latest
# stable as of 2026-05) for reproducibility; bump deliberately.
RUN pip install --no-cache-dir \
        "rllm[verl] @ git+https://github.com/rllm-org/rllm.git@v0.2.1.post1"

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

# Install the tau2 package + repo deps from pyproject.toml. The Python 3.12
# base (NGC 25.08+) satisfies tau2's `requires-python = ">=3.11"` natively
# so no --ignore-requires-python escape hatch is needed.
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

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

# rLLM (agent-RL wrapper) + verl (gradient engine for GRPO). rLLM is
# GitHub-only -- no PyPI package -- and ships verl as an [verl] extra so
# we install both with one git+ URL. Pinned to v0.2.1.post1 (latest stable
# as of 2026-05) for reproducibility; bump deliberately. The previous
# `pip install rllm>=0.1.0` failed at build time with "No matching
# distribution" because the PyPI name `rllm` is squatted/empty.
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

# Install the tau2 package + repo deps from pyproject.toml. Use --no-deps in a
# follow-up if you want to keep this layer thin; for now full install.
#
# `--ignore-requires-python` is intentional: the NVIDIA NGC PyTorch base
# (24.10-py3) ships Python 3.10.12 because NGC tags either go 3.10 (<=25.06)
# or jump to 3.12 (>=25.08), with no 3.11 in between. tau2's pyproject.toml
# declares requires-python = ">=3.11" but the actual code on the RL hot
# path (tau2.orchestrator + tau2.environment + telecom domain) is plain
# 3.10-compatible -- no PEP 695 type aliases, no exception groups, no
# ExceptionGroup catches in the rollout helpers we exercise. Any 3.11-only
# tau2 module that does sneak in would fail at import time, not silently,
# so we'd catch it on the first RL pod run.
COPY pyproject.toml pdm.lock README.md /workspace/repo/
COPY src/ /workspace/repo/src/
RUN pip install --no-cache-dir --ignore-requires-python -e .

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

# RL trainer container.
#
# Base : nvcr.io/nvidia/pytorch:25.08-py3 (CUDA 13.0/12.8, PyTorch 2.7+, Ubuntu
#        24.04, Python 3.12). 25.08 is the first NGC PyTorch container with
#        Python 3.12; we previously tried 24.10 (Python 3.10) and hit the
#        2026-ecosystem-wide Python>=3.11 floor in scikit-learn>=1.8 and
#        pandas>=3.0 (both enforced inside meson.build, so
#        --ignore-requires-python doesn't help).
# Adds : SGLang (disaggregated rollout, TP=4 on GPUs 4-7), FlashAttention,
#        rllm[verl,ui] v0.3.0-pre (gradient engine + agent-RL wrapper + UI
#        logger), tau2 + this repo.
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

# rllm[verl] v0.3.0-pre pulls in vllm==0.17.0 which requires
# opencv-python-headless → numpy>=2.0.0, but verl==0.7.1 requires numpy<2.0.0.
# These are irreconcilable in a single pip resolve. Since we use SGLang as the
# rollout engine (not vllm), skip the [verl] extra entirely and install
# components separately: rllm base + verl + httpx (for [ui] backend).
RUN pip install --no-cache-dir \
        "rllm @ git+https://github.com/rllm-org/rllm.git@v0.3.0-pre" \
        "verl==0.7.1" \
        "httpx>=0.26.0"

# SGLang rollout engine. SGLang natively supports Qwen3-MoE LoRA without the
# fused-qkv workarounds that vLLM colocated required.
# NOTE: sglang>=0.4.0 upgrades torch from the NGC base (2.8.0a0) to 2.9.1,
# so flash-attn MUST be built AFTER sglang to link against the correct ABI.
RUN pip install --no-cache-dir "sglang>=0.4.0"

# sglang>=0.4.0 downgrades torchao to 0.9.0, but PEFT>=0.14 checks torchao at
# import time and requires >=0.16.0 (for its torchao quantization feature gate).
# Upgrade torchao after sglang so PeftModel.from_pretrained doesn't error out
# when loading the SFT LoRA adapter into the actor.
RUN pip install --no-cache-dir "torchao>=0.16.0"

# Build flash-attn wheel against the torch that sglang installed (2.9.1),
# then install it. Building after sglang ensures ABI compatibility.
# --no-build-isolation: lets setup.py import the installed (sglang) torch.
RUN mkdir -p /root/wheelhouse && \
    pip wheel --no-build-isolation --no-deps \
        -w /root/wheelhouse "flash-attn==2.8.1" && \
    pip install --no-cache-dir \
        --find-links /root/wheelhouse \
        "flash-attn==2.8.1"

# Standard helpers used by the trainer + upload scripts.
# litellm>=1.83.0 is required for native `wandb/<model>` routing to W&B
# Inference (provider added late 2025); earlier versions raise
# `LLM Provider NOT provided` from llm_utils.generate.
RUN pip install --no-cache-dir \
        wandb>=0.18 \
        python-dotenv>=1.0 \
        hf_transfer>=0.1.8 \
        "litellm>=1.83.0"

# NGC PyTorch 25.08 ships a CUDA 13.0 toolkit (ptxas) alongside the
# CUDA 12.8 PyTorch wheel. Triton 3.4 only knows CUDA 10/11/12, so any
# torch.compile / Inductor invocation crashes with
#   "Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0".
# Patch Triton's ptx_get_version to treat CUDA 13.x the same as 12.8
# (PTX 8.8) until upstream Triton adds first-class CUDA 13 support.
RUN python -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/compiler.py'); s = p.read_text(); marker = '    if major == 11:'; assert marker in s, 'triton ptx_get_version patch marker not found'; p.write_text(s.replace(marker, '    if major >= 13:\n        return 88  # CUDA 13.x -> map to CUDA 12.8 PTX (8.8); good enough for sm_90.\n    if major == 11:'))"

# verl loads the actor model via HF `from_pretrained` without
# `low_cpu_mem_usage=True`, so EVERY rank materializes the full
# Qwen3-30B-A3B (~60GB bf16) on CPU before FSDP shards it. With 8 ranks
# that's 480GB of pre-shard host RAM -> OOM on the 2TB host node during
# FSDP init. Patch fsdp_workers to add the kwarg so HF uses meta init +
# lazy load. Guard with `if old in s` so the patch is a no-op if verl
# ever fixes this upstream.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/verl/workers/fsdp_workers.py'); s = p.read_text(); old = '            actor_module = actor_module_class.from_pretrained(\n                pretrained_model_name_or_path=local_path,\n                torch_dtype=torch_dtype,\n                config=actor_model_config,\n                trust_remote_code=trust_remote_code,\n                attn_implementation=attn_implementation,\n            )\n'; new = '            actor_module = actor_module_class.from_pretrained(\n                pretrained_model_name_or_path=local_path,\n                torch_dtype=torch_dtype,\n                config=actor_model_config,\n                trust_remote_code=trust_remote_code,\n                attn_implementation=attn_implementation,\n                low_cpu_mem_usage=True,\n            )\n'; p.write_text(s.replace(old, new)) if old in s else print('WARNING: from_pretrained anchor not found in fsdp_workers.py -- patch skipped (may already be fixed in verl 0.7.1)')"

# verl 0.7.1's apply_fsdp2 (fsdp_utils.py:543-546) converts str→list for
# transformer_layer_cls_to_wrap but not set→list. PeftModelForCausalLM._no_split_modules
# returns a set on newer PEFT, causing `set[0]` TypeError at line 546.
# Patch: add `elif isinstance(..., (set, frozenset)): sorted(...)` so the assert
# can subscript [0] regardless of type. Guard with `if old in s`.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/verl/utils/fsdp_utils.py'); s = p.read_text(); old = '    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):\n        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]\n\n    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None'; new = '    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):\n        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]\n    elif isinstance(fsdp_transformer_layer_cls_to_wrap, (set, frozenset)):\n        fsdp_transformer_layer_cls_to_wrap = sorted(fsdp_transformer_layer_cls_to_wrap)\n\n    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None'; p.write_text(s.replace(old, new)) if old in s else print('WARNING: fsdp_utils.py patch anchor not found -- skipped')"

# ----- app layer (your repo + tau2; rebuilt on code changes) -----
FROM base AS app

WORKDIR /workspace/repo

# Install the tau2 package + repo deps from pyproject.toml. The Python 3.12
# base (NGC 25.08+) satisfies tau2's `requires-python = ">=3.11"` natively
# so no --ignore-requires-python escape hatch is needed.
COPY pyproject.toml pdm.lock README.md /workspace/repo/
COPY src/ /workspace/repo/src/
RUN pip install --no-cache-dir -e . \
    && python3 -c "from rllm.trainer.agent_trainer import AgentTrainer" \
    && python3 -c "import sglang"

# The pieces of the existing repo the rollout needs at runtime: the helpers,
# the configs, the scripts, and the tau2 task/policy data files (telecom
# subset only -- ~30MB; the SFT pod doesn't need them because it works
# off a pre-rolled-out W&B JSONL artifact, but the RL pod runs the
# orchestrator inside the container and needs telecom/tasks.json).
COPY tau2_art_helpers.py /workspace/repo/
COPY train_tau2.py /workspace/repo/
COPY train_tau2_distill.py /workspace/repo/
COPY onprem/scripts/ /workspace/scripts/
COPY onprem/configs/ /workspace/configs/
COPY data/tau2/domains/telecom/ /workspace/repo/data/tau2/domains/telecom/
# UserSimulator.system_prompt reads data/tau2/user_simulator/*.md at runtime
# (the user-sim guidelines are domain-agnostic). Bake them in too so the
# rollout doesn't crash with FileNotFoundError on the first reset.
COPY data/tau2/user_simulator/ /workspace/repo/data/tau2/user_simulator/
# The Hydra entrypoint (`python -m onprem.scripts.rllm_train_tau2`) imports
# `from onprem.scripts.tau2_rl_env import Tau2Env` and friends, so the
# `onprem` namespace package needs to live on PYTHONPATH. We copy the whole
# onprem/ tree into /workspace/repo/onprem/ so Python's namespace-package
# machinery (3.3+, no __init__.py needed) can resolve the import.
COPY onprem/ /workspace/repo/onprem/

RUN chmod +x /workspace/scripts/*.sh

ENV PATH=/workspace/scripts:$PATH \
    PYTHONPATH=/workspace/repo:$PYTHONPATH

# Default command: the RL entrypoint script. K8s Job command can override.
CMD ["/workspace/scripts/run_rllm_rl.sh"]

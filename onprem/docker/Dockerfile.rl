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

# NGC PyTorch 25.08 ships transformers 5.x and verl's `transformers` requirement
# is unpinned, so pip happily lets the 5.x installation stand. verl 0.6.1's
# trainer code still uses the 4.x AutoModelForVision2Seq import path
# (removed in transformers 5.0). Pin to the last working 4.x stable.
RUN pip install --no-cache-dir "transformers==4.55.4"

# peft 0.16+ added a TP-aware adapter loader that imports
# `transformers.integrations.tensor_parallel.EmbeddingParallel`; that symbol
# only exists in transformers >= 4.56, so with our 4.55.4 pin we have to
# stay on the last pre-TP peft (0.15.x). 0.15.2 still works against verl 0.6.1
# (LoraConfig / get_peft_model / PeftModel.from_pretrained signatures
# unchanged) and avoids the torchao>=0.16 eager gate that 0.19 introduced.
RUN pip install --no-cache-dir "peft==0.15.2"

# Standard helpers used by the trainer + upload scripts.
# litellm>=1.83 is required for native `wandb/<model>` routing to W&B
# Inference (provider added late 2025); earlier versions raise
# `LLM Provider NOT provided` from llm_utils.generate.
# litellm 1.83 transitively requires tokenizers/transformers >= 4.59 which
# breaks verl 0.6.1's AutoModelForVision2Seq import. Re-pin transformers
# back down to 4.55.4 immediately after.
RUN pip install --no-cache-dir \
        wandb>=0.18 \
        peft>=0.14 \
        python-dotenv>=1.0 \
        hf_transfer>=0.1.8 \
        "litellm>=1.83.0" \
    && pip install --no-cache-dir --force-reinstall --no-deps "transformers==4.55.4"

# Tokenizers must be downgraded to <0.22 to satisfy transformers 4.55.4's
# version pin. We do this in a separate RUN as the last layer to ensure no
# subsequent pip install can pull in tokenizers>=0.22 transitively. The
# explicit rm + assert + pinned version are belt-and-suspenders because
# buildx/cache-mount sometimes leaves the old dist-info directory on disk.
RUN pip uninstall -y tokenizers \
    && rm -rf /usr/local/lib/python3.12/dist-packages/tokenizers \
                /usr/local/lib/python3.12/dist-packages/tokenizers-*.dist-info \
    && pip install --no-cache-dir --no-deps "tokenizers==0.21.4" \
    && python3 -c "import tokenizers; assert tokenizers.__version__ == '0.21.4', tokenizers.__version__"

# vLLM 0.11's Qwen3-MoE LoRA dummy-warmup path crashes during
# determine_available_memory()/profile_run() with
#   AttributeError: 'NoneType' object has no attribute 'shape'
# because PackedLoRA.set_lora() on the FusedMoE expert layers receives a
# packed list whose lora_a is None (mismatch between the dummy LoRA generator
# and the layer wrapper class for FusedMoE). The actual rollout LoRA path
# works fine; only the warmup is broken. Patch maybe_select_dummy_loras to a
# no-op so profile_run skips the buggy dummy-LoRA activation.
RUN python -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/lora_model_runner_mixin.py'); s = p.read_text(); marker = '            self._set_active_loras(tuple(prompt_lora_mapping),\n                                   tuple(token_lora_mapping), lora_requests)'; assert marker in s, 'maybe_select_dummy_loras patch marker not found'; p.write_text(s.replace(marker, '            # vLLM-0.11 + Qwen3-MoE FusedMoE LoRA dummy warmup is broken;\n            # skip the dummy activation. Real LoRA requests still go through\n            # set_active_loras() unmodified.\n            pass'))"

# NGC PyTorch 25.08 ships a CUDA 13.0 toolkit (ptxas) alongside the
# CUDA 12.8 PyTorch wheel. Triton 3.4 only knows CUDA 10/11/12, so any
# torch.compile / Inductor invocation crashes with
#   "Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0".
# Patch Triton's ptx_get_version to treat CUDA 13.x the same as 12.8
# (PTX 8.8) until upstream Triton adds first-class CUDA 13 support.
RUN python -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/compiler.py'); s = p.read_text(); marker = '    if major == 11:'; assert marker in s, 'triton ptx_get_version patch marker not found'; p.write_text(s.replace(marker, '    if major >= 13:\n        return 88  # CUDA 13.x -> map to CUDA 12.8 PTX (8.8); good enough for sm_90.\n    if major == 11:'))"

# verl 0.6.1 loads the actor model via HF `from_pretrained` without
# `low_cpu_mem_usage=True`, so EVERY rank materializes the full
# Qwen3-30B-A3B (~60GB bf16) on CPU before FSDP shards it. With 8 ranks
# that's 480GB of pre-shard host RAM, plus optimizer/grad replicates
# during init -> ~1.8TB peak, which OOMs the 2TB host node during
# rollout_engine.wake_up(). Patch fsdp_workers to add the kwarg so HF
# uses meta init + lazy load and only the rank-0 broadcast path holds
# real weights.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/verl/workers/fsdp_workers.py'); s = p.read_text(); old = '            actor_module = actor_module_class.from_pretrained(\n                pretrained_model_name_or_path=local_path,\n                torch_dtype=torch_dtype,\n                config=actor_model_config,\n                trust_remote_code=trust_remote_code,\n                attn_implementation=attn_implementation,\n            )\n'; assert old in s, 'verl actor from_pretrained anchor not found'; new = '            actor_module = actor_module_class.from_pretrained(\n                pretrained_model_name_or_path=local_path,\n                torch_dtype=torch_dtype,\n                config=actor_model_config,\n                trust_remote_code=trust_remote_code,\n                attn_implementation=attn_implementation,\n                low_cpu_mem_usage=True,\n            )\n'; p.write_text(s.replace(old, new))"

# ----- app layer (your repo + tau2; rebuilt on code changes) -----
FROM base AS app

WORKDIR /workspace/repo

# Install the tau2 package + repo deps from pyproject.toml. The Python 3.12
# base (NGC 25.08+) satisfies tau2's `requires-python = ">=3.11"` natively
# so no --ignore-requires-python escape hatch is needed.
COPY pyproject.toml pdm.lock README.md /workspace/repo/
COPY src/ /workspace/repo/src/
# tau2 / litellm transitively depend on tokenizers >= 0.21 and pip's resolver
# happily upgrades to >= 0.22 here, breaking the transformers 4.55.4 pin in
# the base layer. Re-apply the same downgrade after the editable install.
RUN pip install --no-cache-dir -e . \
    && pip uninstall -y tokenizers \
    && rm -rf /usr/local/lib/python3.12/dist-packages/tokenizers \
                /usr/local/lib/python3.12/dist-packages/tokenizers-*.dist-info \
    && pip install --no-cache-dir --no-deps "tokenizers==0.21.4" \
    && python3 -c "import tokenizers; assert tokenizers.__version__ == '0.21.4', tokenizers.__version__" \
    && python3 -c "from transformers import AutoModelForVision2Seq" \
    && python3 -c "from verl.workers.config import ActorConfig"

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

# Default command: the RL entrypoint script.  K8s Job command can override.
CMD ["/workspace/scripts/run_rllm_rl.sh"]

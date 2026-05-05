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

# SGLang rollout engine. verl 0.7.1 requires sglang==0.5.6 exactly (its
# async_sglang_server.py imports _launch_subprocesses from sglang.srt.entrypoints.http_server
# which was removed in sglang>=0.5.7). Pin to 0.5.6 with the [openai,srt] extras
# verl specifies.
# NOTE: sglang 0.5.6 upgrades torch from the NGC base (2.8.0a0) to 2.9.1,
# so flash-attn MUST be built AFTER sglang to link against the correct ABI.
RUN pip install --no-cache-dir "sglang[openai,srt]==0.5.6"

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

# sglang's scheduler subprocess is launched as /usr/bin/python via
# multiprocessing.spawn. NGC 25.08 only ships libcudart.so.13 (CUDA 13.0);
# libcudart.so.12 lives in the nvidia-cuda-runtime-cu12 Python package but
# is NOT in ldconfig or LD_LIBRARY_PATH. Register it so the subprocess can
# find it regardless of LD_LIBRARY_PATH inheritance.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib'); assert (p / 'libcudart.so.12').exists(), 'libcudart.so.12 not found in nvidia cuda-runtime package'; open('/etc/ld.so.conf.d/nvidia-cuda-12.conf', 'w').write(str(p) + '\n'); print('Registered:', p)" && ldconfig && ldconfig -p | grep "libcudart.so.12"

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

# sglang 0.5.6's apply_torchao_config_to_model() requires torchao==0.9.0 API
# (functional int4_weight_only, float8_dynamic_activation_float8_weight, etc.)
# but torchao>=0.10 replaced all of these with class-based configs. We always
# pass torchao_config=None, so the fix is to move ALL torchao imports to AFTER
# the `if torchao_config is None: return model` early-exit. The first `elif`
# also becomes `if` since it's now an independent conditional chain.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/torchao_utils.py'); s = p.read_text(); old = '    # Lazy import to suppress some warnings\n    from torchao.quantization import (\n        float8_dynamic_activation_float8_weight,\n        float8_weight_only,\n        int4_weight_only,\n        int8_dynamic_activation_int8_weight,\n        int8_weight_only,\n        quantize_,\n    )\n    from torchao.quantization.observer import PerRow, PerTensor\n\n    if torchao_config == \"\" or torchao_config is None:\n        return model\n    elif \"int8wo\" in torchao_config:'; new = '    if torchao_config == \"\" or torchao_config is None:\n        return model\n    # Lazy import to suppress some warnings\n    from torchao.quantization import (\n        float8_dynamic_activation_float8_weight,\n        float8_weight_only,\n        int4_weight_only,\n        int8_dynamic_activation_int8_weight,\n        int8_weight_only,\n        quantize_,\n    )\n    from torchao.quantization.observer import PerRow, PerTensor\n    if \"int8wo\" in torchao_config:'; p.write_text(s.replace(old, new)) if old in s else print('WARNING: torchao_utils.py patch anchor not found -- skipped')"

# sglang 0.5.6's monkey_patch_torch_reductions() patches a bug fixed in
# PyTorch 2.8 (PR pytorch/pytorch#149248). With torch 2.9.1 the patch is
# unnecessary AND breaks: reduce_tensor's output tuple shrank, so the
# hardcoded _REDUCE_TENSOR_ARG_DEVICE_INDEX = 6 causes IndexError during
# actor→rollout weight sync (update_weights → ForkingPickler.dump). Add a
# version guard so the function returns immediately for torch >= 2.8.0.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/patch_torch.py'); s = p.read_text(); old = 'def monkey_patch_torch_reductions():\n    \"\"\"Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed\"\"\"\n\n    # Currently, NPU does not support UUID. This has been temporarily commented out, with support expected in the fourth quarter.\n    if _is_npu:\n        return\n\n    if hasattr(reductions, \"_reduce_tensor_original\"):\n        return'; new = 'def monkey_patch_torch_reductions():\n    \"\"\"Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed\"\"\"\n    import torch as _torch\n    from packaging import version as _version\n    if _version.parse(_torch.__version__.split(\"+\")[0]) >= _version.parse(\"2.8.0\"):\n        return  # PR #149248 already merged in torch>=2.8; patch not needed and breaks 2.9.1\n    # Currently, NPU does not support UUID. This has been temporarily commented out, with support expected in the fourth quarter.\n    if _is_npu:\n        return\n    if hasattr(reductions, \"_reduce_tensor_original\"):\n        return'; p.write_text(s.replace(old, new)) if old in s else print('WARNING: monkey_patch_torch_reductions anchor not found -- skipped')"

# PEFT's _maybe_shard_state_dict_for_tp (save_and_load.py) unconditionally
# imports EmbeddingParallel from transformers.integrations.tensor_parallel,
# which was added in transformers>=4.48. The transformers installed via
# sglang 0.5.6 dependencies predates this → ImportError when loading the
# SFT LoRA adapter into the FSDP actor. We use FSDP (not tensor parallel),
# so the function is a no-op: append a shadowing definition at module end
# (Python last-def wins) that skips the broken import entirely.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/peft/utils/save_and_load.py'); s = p.read_text(); patch = '\n# Patched: no-op for FSDP -- transformers lacks EmbeddingParallel pre-4.48\ndef _maybe_shard_state_dict_for_tp(model, peft_model_state_dict, adapter_name): pass\n'; (p.write_text(s + patch), print('peft TP shard no-op applied')) if '_maybe_shard_state_dict_for_tp' in s and 'Patched: no-op' not in s else print('peft TP shard patch already applied or function not found')"

# When the actor base model is loaded with low_cpu_mem_usage=True, PEFT wraps it
# inside init_empty_weights(), which may leave some parameters (e.g. tied weights,
# buffers not in the adapter checkpoint) as meta tensors.  verl's FSDP2 init then
# calls fsdp2_load_full_state_dict, which does:
#   if dist.get_rank() == 0: model.to(device)   ← crashes on meta tensor
#   else:                     model.to_empty(device)
# Rank-0 should also use to_empty() because set_model_state_dict(broadcast_from_rank0=True)
# fills the actual weights from full_state anyway.  Patch fsdp_utils to unify both
# branches to to_empty().
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/verl/utils/fsdp_utils.py'); s = p.read_text(); old = '    if dist.get_rank() == 0:\n        model = model.to(device=get_device_id(), non_blocking=True)\n    else:\n        model = model.to_empty(device=get_device_id())'; new = '    model = model.to_empty(device=get_device_id())'; (p.write_text(s.replace(old, new, 1)), print('fsdp_utils to_empty patch applied')) if old in s else print('WARNING: fsdp_utils to_empty anchor not found -- skipped')"

# verl's get_init_weight_context_manager uses init_empty_weights() on non-rank-0 workers
# so the base model parameters are meta on those ranks. PEFT's LoraLayer.update_layer
# calls _move_adapter_to_device_of_base_layer at the end of init, which sees the base
# layer device=meta and moves the freshly created CPU LoRA A/B params to meta too.
# That causes set_peft_model_state_dict (called without assign=True) to silently skip
# loading the adapter weights → 2672 "copying from non-meta to meta, no-op" warnings →
# LoRA adapter not applied → base model generates max-length thinking → all rollouts
# filtered → empty batch crash.
# Fix: return early in _move_adapter_to_device_of_base_layer when base is meta; this
# keeps LoRA params on CPU so the subsequent load_state_dict copy_ succeeds and
# full_state captures correct weights before FSDP2 scattering.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/peft/tuners/lora/layer.py'); s = p.read_text(); old = '        device = self.get_param().device\n        meta = torch.device(\"meta\")\n        param = self.get_param()\n'; new = '        device = self.get_param().device\n        meta = torch.device(\"meta\")\n        param = self.get_param()\n        if device == meta:\n            return  # base on meta (FSDP2 deferred init on non-rank-0); keep LoRA on CPU\n'; (p.write_text(s.replace(old, new, 1)), print('peft lora meta-move patch applied')) if old in s else print('WARNING: peft lora layer anchor not found -- skipped')"

# PEFT's set_peft_model_state_dict has two branches: low_cpu_mem_usage (uses assign=True)
# and the default else branch (uses strict=False only, no assign). When LoRA params are
# meta at load time — which happens on FSDP non-rank-0 workers even after the
# _move_adapter_to_device_of_base_layer early-return fix above, for layers where the
# base is also meta — load_state_dict copy_() is a no-op (meta→meta no-copy warning).
# Fix: always use assign=True in the else branch so the checkpoint tensor replaces (not
# copies into) the meta param, correctly materializing it for all 8 ranks.
RUN python3 -c "import pathlib; p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/peft/utils/save_and_load.py'); s = p.read_text(); old = '        load_result = model.load_state_dict(peft_model_state_dict, strict=False)\n'; new = '        load_result = model.load_state_dict(peft_model_state_dict, strict=False, assign=True)\n'; (p.write_text(s.replace(old, new, 1)), print('peft set_peft_model_state_dict assign=True patch applied')) if old in s else print('WARNING: set_peft_model_state_dict else-branch anchor not found -- skipped')"

# verl's collect_lora_params(base_sync_done=False) sends base model params to SGLang
# with names like "q_proj.base_layer.weight" (after replace_lora_wrapper). Qwen3MoE's
# load_weights maps "q_proj" → "qkv_proj" but NOT "qkv_proj.base_layer.weight" → not
# in params_dict → silently skipped → SGLang keeps dummy-random weights for all
# attention/MLP layers → model generates garbage → all rollouts hit MAX_RESPONSE_LENGTH.
#
# Fix part A (fsdp_utils.py): in the FSDP2 base_sync_done=False path, call
# peft_model.merge_adapter() before collecting base model state dict so the collected
# weights already include the LoRA delta (W_eff = W_base + lora_B @ lora_A * scaling).
# Call unmerge_adapter() after collection to restore the separate LoRA representation.
# Both operations happen within FSDP.summon_full_params(writeback=False), so the FSDP2
# sharded params are never permanently modified.
#
# Fix part B (fsdp_workers.py): remove replace_lora_wrapper for base_sync_done=False.
# After merging, params have standard HF names ("q_proj.weight" not "q_proj.base_layer.weight"),
# which Qwen3MoE's load_weights correctly maps q_proj→qkv_proj via stacked_params_mapping.
RUN echo 'aW1wb3J0IHBhdGhsaWIsIHN5cwoKIyAtLS0gUGF0Y2ggQTogZnNkcF91dGlscy5weSAtLS0KcDEgPSBwYXRobGliLlBhdGgoJy91c3IvbG9jYWwvbGliL3B5dGhvbjMuMTIvZGlzdC1wYWNrYWdlcy92ZXJsL3V0aWxzL2ZzZHBfdXRpbHMucHknKQpzMSA9IHAxLnJlYWRfdGV4dCgpCm9sZDEgPSAoJyAgICAgICAgICAgICAgICBlbHNlOlxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgIG1vZGVsID0gcGVmdF9tb2RlbC5iYXNlX21vZGVsLm1vZGVsXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgb3JpZ19kZXYgPSAiY3B1IiBpZiAiY3B1IiBpbiBzdHIobmV4dChtb2RlbC5wYXJhbWV0ZXJzKCkpLmRldmljZSkgZWxzZSBnZXRfZGV2aWNlX25hbWUoKVxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgIG1vZGVsID0gbW9kZWwudG8oImNwdSIpXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgZm9yIG5hbWUsIHBhcmFtIGluIG1vZGVsLnN0YXRlX2RpY3QoKS5pdGVtcygpOlxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgICAgICBpZiBhbnkoeCBpbiBuYW1lIGZvciB4IGluIFsiX2ZsYXRfcGFyYW0iLCAibG9yYV8iXSk6XG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb250aW51ZVxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgICAgICBuYW1lID0gbmFtZS5yZXBsYWNlKCJfZnNkcF93cmFwcGVkX21vZHVsZS4iLCAiIikucmVwbGFjZSgiLmJhc2VfbGF5ZXIiLCAiIilcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICAgICAgbG9yYV9wYXJhbXNbbmFtZV0gPSAoXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgICAgICBwYXJhbS5mdWxsX3RlbnNvcigpLmRldGFjaCgpLmNwdSgpXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiBoYXNhdHRyKHBhcmFtLCAiZnVsbF90ZW5zb3IiKVxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgICAgICAgICAgZWxzZSBwYXJhbS5kZXRhY2goKS5jcHUoKVxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgICAgICApXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgbW9kZWwgPSBtb2RlbC50byhvcmlnX2RldilcbicKICAgICAgICAnICAgICAgICAgICAgZ2V0X3RvcmNoX2RldmljZSgpLmVtcHR5X2NhY2hlKCknKQpuZXcxID0gKCcgICAgICAgICAgICAgICAgZWxzZTogICMgc2dsYW5nLWxvcmEtbWVyZ2UtMjAyNlxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgIG1vZGVsID0gcGVmdF9tb2RlbC5iYXNlX21vZGVsLm1vZGVsXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgb3JpZ19kZXYgPSAiY3B1IiBpZiAiY3B1IiBpbiBzdHIobmV4dChtb2RlbC5wYXJhbWV0ZXJzKCkpLmRldmljZSkgZWxzZSBnZXRfZGV2aWNlX25hbWUoKVxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgIG1vZGVsID0gbW9kZWwudG8oImNwdSIpXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgX3BlZnRfbWVyZ2VkID0gRmFsc2VcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICBpZiBoYXNhdHRyKHBlZnRfbW9kZWwsICJtZXJnZV9hZGFwdGVyIikgYW5kIG5vdCBnZXRhdHRyKHBlZnRfbW9kZWwsICJtZXJnZWRfYWRhcHRlcnMiLCBbXSk6XG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgIHBlZnRfbW9kZWwubWVyZ2VfYWRhcHRlcigpXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgIF9wZWZ0X21lcmdlZCA9IFRydWVcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICBmb3IgbmFtZSwgcGFyYW0gaW4gbW9kZWwuc3RhdGVfZGljdCgpLml0ZW1zKCk6XG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgIGlmIGFueSh4IGluIG5hbWUgZm9yIHggaW4gWyJfZmxhdF9wYXJhbSIsICJsb3JhXyJdKTpcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgIG5hbWUgPSBuYW1lLnJlcGxhY2UoIl9mc2RwX3dyYXBwZWRfbW9kdWxlLiIsICIiKS5yZXBsYWNlKCIuYmFzZV9sYXllciIsICIiKVxuJwogICAgICAgICcgICAgICAgICAgICAgICAgICAgICAgICBsb3JhX3BhcmFtc1tuYW1lXSA9IChcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmFtLmZ1bGxfdGVuc29yKCkuZGV0YWNoKCkuY3B1KClcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmIGhhc2F0dHIocGFyYW0sICJmdWxsX3RlbnNvciIpXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgICAgICBlbHNlIHBhcmFtLmRldGFjaCgpLmNwdSgpXG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgIClcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICBpZiBfcGVmdF9tZXJnZWQ6XG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgICAgIHBlZnRfbW9kZWwudW5tZXJnZV9hZGFwdGVyKClcbicKICAgICAgICAnICAgICAgICAgICAgICAgICAgICBtb2RlbCA9IG1vZGVsLnRvKG9yaWdfZGV2KVxuJwogICAgICAgICcgICAgICAgICAgICBnZXRfdG9yY2hfZGV2aWNlKCkuZW1wdHlfY2FjaGUoKScpCmlmICdzZ2xhbmctbG9yYS1tZXJnZS0yMDI2JyBpbiBzMToKICAgIHByaW50KCdmc2RwX3V0aWxzLnB5IHNnbGFuZy1sb3JhLW1lcmdlIGFscmVhZHkgYXBwbGllZCcpCmVsaWYgb2xkMSBpbiBzMToKICAgIHAxLndyaXRlX3RleHQoczEucmVwbGFjZShvbGQxLCBuZXcxLCAxKSkKICAgIHByaW50KCdmc2RwX3V0aWxzLnB5IHNnbGFuZy1sb3JhLW1lcmdlIGFwcGxpZWQnKQplbHNlOgogICAgcHJpbnQoJ1dBUk5JTkc6IGZzZHBfdXRpbHMucHkgZnNkcDIgYW5jaG9yIG5vdCBmb3VuZCAtLSBza2lwcGVkJywgZmlsZT1zeXMuc3RkZXJyKQogICAgc3lzLmV4aXQoMSkKCiMgLS0tIFBhdGNoIEI6IGZzZHBfd29ya2Vycy5weSAtLS0KcDIgPSBwYXRobGliLlBhdGgoJy91c3IvbG9jYWwvbGliL3B5dGhvbjMuMTIvZGlzdC1wYWNrYWdlcy92ZXJsL3dvcmtlcnMvZnNkcF93b3JrZXJzLnB5JykKczIgPSBwMi5yZWFkX3RleHQoKQpvbGQyID0gKCcgICAgICAgICAgICBpZiBub3Qgc2VsZi5iYXNlX3N5bmNfZG9uZTpcbicKICAgICAgICAnICAgICAgICAgICAgICAgIHBhcmFtcyA9IHtyZXBsYWNlX2xvcmFfd3JhcHBlcihrLCBwZWZ0X2NvbmZpZyk6IHYgZm9yIGssIHYgaW4gcGFyYW1zLml0ZW1zKCl9JykKbmV3MiA9ICgnICAgICAgICAgICAgIyBzZ2xhbmctbG9yYS1tZXJnZS0yMDI2OiBtZXJnZWQgd2VpZ2h0cyBoYXZlIHN0ZCBIRiBuYW1lcztcbicKICAgICAgICAnICAgICAgICAgICAgIyBza2lwIHJlcGxhY2VfbG9yYV93cmFwcGVyIHNvIFNHTGFuZyBjYW4gbWFwIHFfcHJvai0+cWt2X3Byb2ogY29ycmVjdGx5LicpCmlmICdzZ2xhbmctbG9yYS1tZXJnZS0yMDI2JyBpbiBzMjoKICAgIHByaW50KCdmc2RwX3dvcmtlcnMucHkgc2dsYW5nLWxvcmEtbWVyZ2UgYWxyZWFkeSBhcHBsaWVkJykKZWxpZiBvbGQyIGluIHMyOgogICAgcDIud3JpdGVfdGV4dChzMi5yZXBsYWNlKG9sZDIsIG5ldzIsIDEpKQogICAgcHJpbnQoJ2ZzZHBfd29ya2Vycy5weSBzZ2xhbmctbG9yYS1tZXJnZSBhcHBsaWVkJykKZWxzZToKICAgIHByaW50KCdXQVJOSU5HOiBmc2RwX3dvcmtlcnMucHkgYW5jaG9yIG5vdCBmb3VuZCAtLSBza2lwcGVkJywgZmlsZT1zeXMuc3RkZXJyKQogICAgc3lzLmV4aXQoMSkK' | base64 -d | python3

# MultiTurnWorkflow.run() discards the ModelOutput (which carries prompt_ids,
# completion_ids, logprobs from VerlEngine) and only passes output.text to
# agent.update_from_model(). Trajectory steps are created with model_output=None.
# rllm.experimental.verl.transform._process_trajectory skips every step that has
# model_output=None → empty sequence list → "received an empty list of sequences" crash.
# Fix: after agent.update_from_model, backfill model_output on the last step so
# transform.py can extract token IDs for PPO training.
RUN echo 'aW1wb3J0IHBhdGhsaWIsIHN5cwpwID0gcGF0aGxpYi5QYXRoKCcvdXNyL2xvY2FsL2xpYi9weXRob24zLjEyL2Rpc3QtcGFja2FnZXMvcmxsbS93b3JrZmxvd3MvbXVsdGlfdHVybl93b3JrZmxvdy5weScpCnMgPSBwLnJlYWRfdGV4dCgpCm9sZCA9ICcgICAgICAgICAgICBhY3Rpb24gPSBzZWxmLmFnZW50LnVwZGF0ZV9mcm9tX21vZGVsKHJlc3BvbnNlKVxuXG4gICAgICAgICAgICBuZXh0X29icywgcmV3YXJkLCBkb25lLCBpbmZvID0gYXdhaXQgc2VsZi50aW1lZF9lbnZfY2FsbChzZWxmLmVudi5zdGVwLCBhY3Rpb24pXG4nCm5ldyA9ICgnICAgICAgICAgICAgYWN0aW9uID0gc2VsZi5hZ2VudC51cGRhdGVfZnJvbV9tb2RlbChyZXNwb25zZSlcbicKICAgICAgICdcbicKICAgICAgICcgICAgICAgICAgICAjIG11bHRpLXR1cm4tbW9kZWwtb3V0cHV0LTIwMjY6IGF0dGFjaCBNb2RlbE91dHB1dCBzb1xuJwogICAgICAgJyAgICAgICAgICAgICMgZXhwZXJpbWVudGFsLnZlcmwudHJhbnNmb3JtIGNhbiByZWFkIHByb21wdF9pZHMvY29tcGxldGlvbl9pZHMuXG4nCiAgICAgICAnICAgICAgICAgICAgaWYgc2VsZi5hZ2VudC50cmFqZWN0b3J5LnN0ZXBzOlxuJwogICAgICAgJyAgICAgICAgICAgICAgICBfbGFzdCA9IHNlbGYuYWdlbnQudHJhamVjdG9yeS5zdGVwc1stMV1cbicKICAgICAgICcgICAgICAgICAgICAgICAgaWYgX2xhc3QubW9kZWxfb3V0cHV0IGlzIE5vbmU6XG4nCiAgICAgICAnICAgICAgICAgICAgICAgICAgICBfbGFzdC5tb2RlbF9vdXRwdXQgPSBvdXRwdXRcbicKICAgICAgICdcbicKICAgICAgICcgICAgICAgICAgICBuZXh0X29icywgcmV3YXJkLCBkb25lLCBpbmZvID0gYXdhaXQgc2VsZi50aW1lZF9lbnZfY2FsbChzZWxmLmVudi5zdGVwLCBhY3Rpb24pXG4nKQppZiAnbXVsdGktdHVybi1tb2RlbC1vdXRwdXQtMjAyNicgaW4gczoKICAgIHAud3JpdGVfdGV4dChzLnJlcGxhY2UoCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgX2xhc3QubW9kZWxfb3V0cHV0ID0gb3V0cHV0XG4nCiAgICAgICAgJyAgICAgICAgICAgICAgICAgICAgX2xhc3QuYmFja2ZpbGxfZnJvbV9tb2RlbF9vdXRwdXQoKVxuJywKICAgICAgICAnICAgICAgICAgICAgICAgICAgICBfbGFzdC5tb2RlbF9vdXRwdXQgPSBvdXRwdXRcbicKICAgICkpCiAgICBwcmludCgnbXVsdGlfdHVybl93b3JrZmxvdy5weSBiYWNrZmlsbF9jYWxsIHJlbW92ZWQnKQplbGlmIG9sZCBpbiBzOgogICAgcC53cml0ZV90ZXh0KHMucmVwbGFjZShvbGQsIG5ldywgMSkpCiAgICBwcmludCgnbXVsdGlfdHVybl93b3JrZmxvdy5weSBtb2RlbF9vdXRwdXQgYmFja2ZpbGwgYXBwbGllZCcpCmVsc2U6CiAgICBwcmludCgnV0FSTklORzogbXVsdGlfdHVybl93b3JrZmxvdy5weSBhbmNob3Igbm90IGZvdW5kIC0tIHNraXBwZWQnLCBmaWxlPXN5cy5zdGRlcnIpCiAgICBzeXMuZXhpdCgxKQo=' | base64 -d | python3

# torch 2.9.1 serializes FSDP de-sharded CPU tensors via ForkingPickler's
# FD-based shared memory (rebuild_storage_fd). The FD is passed through
# multiprocessing.resource_sharer, which creates a connection.Listener with
# process.current_process().authkey. The FSDP actor (Ray worker, authkey_A)
# creates the Listener; the SGLang scheduler (spawned process, fresh random
# authkey_B ≠ authkey_A) connects with authkey_B → AuthenticationError.
# Full trace: scheduler→tp_worker:167→model_runner:2871→common.py:2215
# →rebuild_storage_fd→resource_sharer.detach→Client(authkey=B).
#
# Fix: write a standalone patch module _sglang_rs_fix.py that overrides
# _ResourceSharer._start (Listener) and get_connection (Client) to use a
# fixed shared authkey b"sglang-verl-ipc-2026". Install via a .pth file in
# site-packages so it runs at Python startup in ALL processes (FSDP Ray
# workers AND sglang spawned processes) without from __future__ ordering issues.
RUN printf 'import multiprocessing.resource_sharer as _rs_fix\nimport multiprocessing.connection as _mc_fix\n_AUTHKEY_FIXED = b"sglang-verl-ipc-2026"\ndef _rs_start_patched(self):\n    assert self._listener is None\n    self._listener = _mc_fix.Listener(authkey=_AUTHKEY_FIXED, backlog=128)\n    self._address = self._listener.address\n    import threading; _t = threading.Thread(target=self._serve); _t.daemon = True; _t.start(); self._thread = _t\n_rs_fix._ResourceSharer._start = _rs_start_patched\n@staticmethod\ndef _rs_get_conn_patched(ident):\n    import os; addr, key = ident\n    _c = _mc_fix.Client(addr, authkey=_AUTHKEY_FIXED); _c.send((key, os.getpid())); return _c\n_rs_fix._ResourceSharer.get_connection = _rs_get_conn_patched\n' > /usr/local/lib/python3.12/dist-packages/_sglang_rs_fix.py && printf 'import _sglang_rs_fix\n' > /usr/local/lib/python3.12/dist-packages/_sglang_rs_fix.pth && python3 -c "import _sglang_rs_fix; print('resource_sharer authkey fix loaded OK')"

# rllm VerlEngine builds train_sampling_params / val_sampling_params without
# stop_token_ids, so SGLang never sees Qwen3's <|im_end|> (token 151645) as an
# EOS token.  Every rollout generates to max_tokens, gets finish_reason="length",
# triggers MAX_RESPONSE_LENGTH_EXCEEDED, gets filtered out, and the episode list
# is empty → pad_sequence([]) crash.
# Fix: directly add stop_token_ids=[151645] to both sampling_params dicts via
# an in-place text edit of the two verl_engine.py files.  Do NOT use a .pth
# startup import — importing verl.experimental.agent_loop before ray.init()
# corrupts Ray's GPU assignment for WorkerDict actors, forcing all 8 FSDP
# workers onto GPU 0 and triggering an immediate OOM during actor_rollout_init_model.
RUN echo 'aW1wb3J0IHBhdGhsaWIsIHN5cwpBTkNIT1IgPSAnICAgICAgICBwcmludChmInRyYWluX3NhbXBsaW5nX3BhcmFtczoge3NlbGYudHJhaW5fc2FtcGxpbmdfcGFyYW1zfSIpJwpJTlNFUlQgPSAnICAgICAgICBzZWxmLnRyYWluX3NhbXBsaW5nX3BhcmFtc1sic3RvcF90b2tlbl9pZHMiXSA9IFsxNTE2NDVdXG4gICAgICAgIHNlbGYudmFsX3NhbXBsaW5nX3BhcmFtc1sic3RvcF90b2tlbl9pZHMiXSA9IFsxNTE2NDVdXG4nCm9rID0gVHJ1ZQpmb3IgcGF0aCBpbiBbCiAgICAiL3Vzci9sb2NhbC9saWIvcHl0aG9uMy4xMi9kaXN0LXBhY2thZ2VzL3JsbG0vZW5naW5lL3JvbGxvdXQvdmVybF9lbmdpbmUucHkiLAogICAgIi91c3IvbG9jYWwvbGliL3B5dGhvbjMuMTIvZGlzdC1wYWNrYWdlcy9ybGxtL2V4cGVyaW1lbnRhbC9yb2xsb3V0L3ZlcmxfZW5naW5lLnB5IiwKXToKICAgIHAgPSBwYXRobGliLlBhdGgocGF0aCkKICAgIGlmIG5vdCBwLmV4aXN0cygpOgogICAgICAgIHByaW50KGYiU0tJUCAobm90IGZvdW5kKToge3BhdGh9IikKICAgICAgICBjb250aW51ZQogICAgdGV4dCA9IHAucmVhZF90ZXh0KCkKICAgIGlmICJzdG9wX3Rva2VuX2lkcyIgaW4gdGV4dDoKICAgICAgICBwcmludChmImFscmVhZHkgcGF0Y2hlZDoge3BhdGh9IikKICAgICAgICBjb250aW51ZQogICAgaWYgQU5DSE9SIG5vdCBpbiB0ZXh0OgogICAgICAgIHByaW50KGYiRVJST1I6IGFuY2hvciBub3QgZm91bmQgaW4ge3BhdGh9IikKICAgICAgICBvayA9IEZhbHNlCiAgICAgICAgY29udGludWUKICAgIG5ld190ZXh0ID0gdGV4dC5yZXBsYWNlKEFOQ0hPUiwgSU5TRVJUICsgQU5DSE9SLCAxKQogICAgcC53cml0ZV90ZXh0KG5ld190ZXh0KQogICAgcHJpbnQoZiJwYXRjaGVkIHtwYXRofSIpCnN5cy5leGl0KDAgaWYgb2sgZWxzZSAxKQo=' | base64 -d | python3 && python3 -c "import rllm.engine.rollout.verl_engine as m, inspect; src=inspect.getsource(m.VerlEngine.__init__); assert 'stop_token_ids' in src; print('stop_token_ids direct-patch OK (engine)')" && python3 -c "import rllm.experimental.rollout.verl_engine as m, inspect; src=inspect.getsource(m.VerlEngine.__init__); assert 'stop_token_ids' in src; print('stop_token_ids direct-patch OK (experimental)')"

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

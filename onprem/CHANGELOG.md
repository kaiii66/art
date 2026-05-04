# On-prem RL pipeline rewrite for rLLM v0.2.1 — change log

This document summarizes everything that landed on `feature/onprem` to make
the on-prem tau2 RL pipeline (Qwen3-30B-A3B-Instruct-2507, LoRA, GRPO via
verl + colocated vLLM TP=8 on 8x H100) work end-to-end on K8s.

The starting point was a from-scratch rewrite against rLLM
v0.2.1.post1 (`BaseAgent` / `BaseEnv` / `MultiTurnWorkflow` / `AgentTrainer`)
plus a long debug loop on K8s. The terminal state: smoke + full RL +
leaderboard all green.

## Final outcome

| Stage | Wall clock | Status | Artifact / URL |
|---|---|---|---|
| SFT (pre-existing) | -- | done | `wandb-artifact:///kwt/tau2-ART-autoresearch-telecom-0430/tau2-sft-Qwen3-30B-A3B-Instruct-2507-05022004:v0` |
| Smoke RL (`smoke05040422`) | ~10 min | done | training + LoRA save + W&B upload all OK |
| Full RL (`prod05040508`) | ~85 min | done | `wandb-artifact:///kwt/tau2-ART-autoresearch-telecom-0430/tau2-rl-Qwen3-30B-A3B-Instruct-2507-prod05040508:v0` |
| Leaderboard | ~12 min | done | [tau2-telecom-leaderboard-shaped-v1](https://wandb.ai/kwt/tau2-ART-autoresearch-telecom-0430/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1) |

Leaderboard scores (120 telecom validation tasks, 1 trial):

| Row | success.mean | task_reward.mean | tokens | latency |
|---|---|---|---|---|
| `sft-04301749-alias-v0` | 0.0833 | 0.2504 | 806 | 38.0 s |
| `rl-04301749-alias-v0` | 0.0667 | 0.2436 | 796 | 42.0 s |

(Base row was auto-skipped via the project's sentinel artifact.)

## Architectural change: rLLM v0.2 rewrite

The previous integration targeted a speculative rLLM v0.3-pre API that does
not match what `pip install rllm[verl]==0.2.1.post1` ships. The whole
rollout/training stack was rewritten against the actually-released API.

### New modules

- [onprem/scripts/tau2_rl_env.py](onprem/scripts/tau2_rl_env.py) — `Tau2Env(BaseEnv)` adapter that wraps tau2's `Orchestrator` as a Gym-style `(reset, step)` so rLLM's `MultiTurnWorkflow` can drive it. Routes:
  - agent text -> `UserSimulator.generate_next_message`
  - agent tool calls -> `Environment.get_response`
  - mixed/empty -> `AGENT_ERROR` termination
  - `STOP`/`TRANSFER`/`OUT_OF_SCOPE` in user text -> `USER_STOP`
  - terminal step computes shaped reward via `compute_shaped_reward` (action / nl_assertions / termination / tool_accuracy / tool_arg_accuracy / step_penalty).
- [onprem/scripts/tau2_rl_agent.py](onprem/scripts/tau2_rl_agent.py) — `Tau2AssistantAgent(BaseAgent)` that emits OpenAI-tools formatted messages, parses Qwen tool calls via `QwenToolParser`, and feeds tau2's tool schemas to the policy.
- [onprem/scripts/tau2_rl_dataset.py](onprem/scripts/tau2_rl_dataset.py) — Builds rLLM `Dataset` rows from a tau2 task list (W&B artifact or local file).
- [onprem/scripts/rllm_train_tau2.py](onprem/scripts/rllm_train_tau2.py) — Hydra entry point that initializes W&B, resolves the starting LoRA, registers the train/val datasets, and runs `AgentTrainer`. Falls back to the train parquet for `data.val_files` (and turns off `test_freq`) when no validation dataset is provided, so smoke runs don't crash on `FileNotFoundError`.
- [onprem/configs/tau2_overrides.yaml](onprem/configs/tau2_overrides.yaml) — Hydra overlay layered on top of verl's `ppo_trainer` (with `agent_ppo_trainer`'s rLLM defaults inlined). All non-default values for actor / rollout / FSDP / rLLM / trainer live here.
- [onprem/scripts/run_rllm_rl.sh](onprem/scripts/run_rllm_rl.sh) — K8s pod entry point: sanity checks, `python -m onprem.scripts.rllm_train_tau2`, then auto-discovery + upload of the saved LoRA.
- [onprem/tests/](onprem/tests/) — Offline pytest suite for `Tau2Env._classify_action` and `Tau2AssistantAgent` formatting.

### Removed

- [onprem/scripts/tau2_rllm_rollout.py](onprem/scripts/tau2_rllm_rollout.py) — old `@rllm.rollout` shim against the v0.3-pre API.
- [onprem/configs/rllm_train_config.yaml](onprem/configs/rllm_train_config.yaml) — replaced by the Hydra overlay above.

## Dependency hell — Dockerfile pins

[onprem/docker/Dockerfile.rl](onprem/docker/Dockerfile.rl) accumulated the
following constraints to keep verl 0.6.1, rLLM 0.2.1.post1, vLLM 0.11.0,
and tau2's editable install all working under a single Python 3.12 NGC
PyTorch base (`nvcr.io/nvidia/pytorch:25.08-py3`):

| Pin | Reason |
|---|---|
| `flash-attn==2.8.3` (built `--no-build-isolation`) | flash-attn's `setup.py` imports torch at build time; it must be installed before rLLM so rLLM doesn't try to rebuild it under PEP-517 isolation. |
| `rllm[verl] @ v0.2.1.post1` | Pinned commit; this is what the new agent/env/workflow modules target. |
| `transformers==4.55.4` | verl 0.6.1 imports `AutoModelForVision2Seq`, removed in transformers 5.x. |
| `peft==0.15.2` | peft >= 0.16 imports `transformers.integrations.tensor_parallel.EmbeddingParallel`, only present in transformers >= 4.56. |
| `litellm>=1.83.0` | tau2's user simulator uses `wandb/Qwen/...` model ids; the native `wandb` provider was added late 2025. Earlier versions fail with `LLM Provider NOT provided`. |
| `tokenizers==0.21.4` (force-reinstall) | litellm 1.83 transitively pulls tokenizers 0.22.x, which violates transformers 4.55.4's pin. |
| `huggingface-hub==0.34.4` (force-reinstall) | tau2's editable install pulls hf-hub 1.x, also violating the transformers 4.55.4 pin. |

The downgrades use `pip uninstall -y <pkg> && rm -rf .../<pkg>* &&
pip install --no-cache-dir --no-deps "<pkg>==<ver>"` with a build-time
`python -c "import x; assert x.__version__ == '...'`" to make any future
upstream regression fail the image build instead of a K8s pod.

The same downgrades have to be re-applied **after** the editable
`pip install -e .` of the tau2 repo, because tau2's deps re-bump them.

## Source patches inside the image

These are applied via `RUN python -c "..."` in the Dockerfile so they
re-run on every build but are scoped to specific marker strings (so they
fail loudly if upstream renames or refactors).

### vLLM 0.11 LoRA dummy-warmup crash

`vllm/v1/worker/lora_model_runner_mixin.py` calls `set_lora` on FusedMoE
expert layers during `profile_run()` with `PackedLoRA(lora_a=None)`,
which crashes on `'NoneType' object has no attribute 'shape'`. The actual
rollout LoRA path is fine; only the dummy warmup is broken.

Patch: replace the dummy `_set_active_loras` call with `pass` so profile
memory determination skips the buggy activation.

### Triton 3.4 vs CUDA 13.0

NGC PyTorch 25.08 ships a CUDA 13.0 toolkit (`ptxas`) alongside the CUDA
12.8 PyTorch wheel. Triton 3.4's `ptx_get_version` only knows
CUDA 10/11/12, so any `torch.compile` invocation crashes with
`Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0`.

Patch: map `major >= 13` to PTX 8.8 (== CUDA 12.8 PTX), which is correct
for sm_90 H100s.

### verl `from_pretrained` host-RAM OOM

verl 0.6.1 calls `AutoModelForCausalLM.from_pretrained(...)` without
`low_cpu_mem_usage=True` for the actor, so every rank materializes the
full 60 GB Qwen3-30B-A3B on CPU before FSDP shards it. With 8 ranks
that's 480 GB pre-shard + optimizer/grad replicates -> ~1.8 TB peak,
which OOMs the 2 TB host node.

Patch: add `low_cpu_mem_usage=True` to the `from_pretrained` call so HF
uses meta init + lazy load and only the rank-0 broadcast path holds the
real weights.

## Memory tuning ([tau2_overrides.yaml](onprem/configs/tau2_overrides.yaml))

| Setting | Value | Reason |
|---|---|---|
| `actor.fsdp_config.param_offload` / `optimizer_offload` | `false` | CPU offload combined with vLLM sleep mode pinned ~120 GB/rank in host RAM and OOMed the node. H100 80 GB has enough headroom for sharded LoRA training of 30B-MoE. |
| `rollout.gpu_memory_utilization` | `0.5` | Lower from default 0.85 so vLLM weights + KV cache fit alongside FSDP actor weights (both kept on GPU permanently). |
| `rollout.free_cache_engine` | `false` | Disables vLLM sleep mode (which otherwise pins ~230 GB of weights per rank in CPU RAM during `wake_up`). |
| `rollout.layered_summon` | `true` | Forces verl's FSDP -> vLLM weight sync to summon one layer at a time. The default LoRA `__collect_lora_params` path gathers `.full_tensor().cpu()` of the entire base model on every rank -> host OOM. |
| `rollout.load_format` | `safetensors` | Required by `layered_summon`: vLLM needs the base model preloaded so per-layer DTensor shards can replace base weights in-place. |
| `rollout.enforce_eager` | `true` | Skips CUDA-graph capture (which goes through Triton and trips the CUDA 13.0 issue, even with the Triton patch above). Throughput hit acceptable for now. |
| `rollout.engine_kwargs.vllm.disable_custom_all_reduce` | `true` | vLLM's custom all-reduce uses `cuMem`, which can't rendezvous when FSDP has already pinned the same devices. Falls back to plain NCCL. |
| `rollout.engine_kwargs.vllm.fully_sharded_loras` | `true` | Shards LoRA tensors across TP=8 ranks; without it, vLLM's dummy-LoRA profile run still tripped on Qwen3-MoE's fused `qkv_proj`. |
| `model.target_modules` | `"all-linear"` (string, not list) | verl's `HFModelConfig.target_modules` is `Optional[str]`. Passing `q/k/v/o_proj` as a regex tricked vLLM into emitting `None` subloras for the fused `qkv_proj` packed module. `"all-linear"` lets peft pick all linear layers, and vLLM resolves the same string consistently. |

Companion env vars in [run_rllm_rl.sh](onprem/scripts/run_rllm_rl.sh):

```bash
export VLLM_USE_V1=1                      # required by rllm AgentWorkflowEngine
export VLLM_ALLREDUCE_USE_SYMM_MEM=0      # disable PyTorch symmetric-memory all-reduce
export NCCL_CUMEM_ENABLE=0                # disable cuMem-based NCCL allocator
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

## Checkpoint extraction + W&B upload

verl writes a HF-compatible LoRA adapter when the actor is a `PeftModel`,
but only when `trainer.save_freq > 0` and only at:

```
$RL_OUTPUT_DIR/global_step_<N>/actor/lora_adapter/{adapter_config.json,
adapter_model.safetensors}
```

Two changes were needed to publish it:

1. [tau2_overrides.yaml](onprem/configs/tau2_overrides.yaml): set
   `trainer.save_freq=1` (and `max_actor_ckpt_to_keep=2` so the PVC
   doesn't fill across iterations).
2. [run_rllm_rl.sh](onprem/scripts/run_rllm_rl.sh): auto-discover the
   highest `global_step_*/actor/lora_adapter` dir, stage it together
   with the sibling `actor/huggingface/` tokenizer files into a single
   `upload/` directory, then call `wandb_lora_upload.py` against that
   staged directory. W&B Inference requires the tokenizer files
   alongside the adapter for LoRA serving.

## K8s manifest changes ([rl_job.yaml](onprem/k8s/rl_job.yaml))

| Env / mount | Reason |
|---|---|
| `HF_HOME=/artifacts/hf-cache` + `HUGGINGFACE_HUB_CACHE=/artifacts/hf-cache/hub` | Persist the 60 GB Qwen3-30B safetensors snapshot on the artifacts PVC so the model survives pod restarts and doesn't hit HF Hub on cold start. The first prod run failed with `OSError: ... does not appear to have a file named model.safetensors` because the rate-limited / partial download didn't deposit a usable index. |
| Existing `artifacts` PVC mount | Now also serves as the LoRA checkpoint sink + HF cache. |

A one-shot `tau2-rl-hfprefetch` pod was used to populate the PVC cache
once (`huggingface_hub.snapshot_download`).

## Pre-existing scripts touched

- [onprem/scripts/wandb_lora_upload.py](onprem/scripts/wandb_lora_upload.py) — unchanged contract; still validates `adapter_config.json` + `adapter_model*.{safetensors,bin}`. The runscript now stages a directory that satisfies its contract.

## Operational gotchas worth remembering

- The full RL pod uploaded the LoRA into `tau2-ART-autoresearch-telecom`,
  but the SFT artifact lives in `tau2-ART-autoresearch-telecom-0430`. The
  leaderboard registers both via `art.TrainableModel` against a single
  project, so they have to be in the same project. For future runs, set
  `WANDB_PROJECT=tau2-ART-autoresearch-telecom-0430` directly in the
  K8s job env. For this run, a 1-shot republish pod copied the adapter
  into the `-0430` project.
- The user simulator's "This model isn't mapped yet ..." log line is
  noise — litellm just doesn't have a token-cost map for the wandb
  provider. The actual generation works.
- Total host RAM headroom on the 2 TB node is fragile. The combination
  `param_offload=false`, `optimizer_offload=false`,
  `gpu_memory_utilization=0.5`, `free_cache_engine=false`,
  `layered_summon=true`, `load_format=safetensors`, and the verl
  `low_cpu_mem_usage=True` patch are all required together; removing
  any one of them brought back the OOM in testing.

## Commit history (most recent first, on `feature/onprem`)

```
53b9137 fix(rl): pin HF_HOME on artifacts PVC (avoid re-download + cold-start OSError)
772c1e1 fix(rl): correct lora_adapter path + stage tokenizer for W&B upload
0f54178 feat(rl): save_freq=1 + auto-discover verl lora_adapter dir for W&B upload
005270b fix(rl): also re-pin huggingface-hub<1.0 after editable install
fe8b7e4 fix(rl): re-pin tokenizers after editable install (was being upgraded by tau2 deps)
f0d5b1e fix(rl): rm dist-info + assert tokenizers==0.21.4 (build cache leftover)
ee31a15 fix(rl): uninstall+reinstall tokenizers to <0.22
13f6ac8 fix(rl): force-reinstall tokenizers to 0.21.4 (downgrade)
de526b8 fix(rl): pin tokenizers<0.22 alongside transformers re-pin
a94fa0d fix(rl): re-pin transformers==4.55.4 after litellm bump (Vision2Seq import)
f4d7f91 fix(rl): bump litellm to >=1.83 (wandb/ provider routing)
ac7f27f fix(rl): rollout.load_format=safetensors (required for layered_summon)
ddc828b fix(rl): rollout.layered_summon=true (avoid full-base-model gather to CPU on LoRA first sync)
079c10e fix(rl): patch verl from_pretrained to use low_cpu_mem_usage (host RAM OOM)
cd065be fix(rl): free_cache_engine=false (disable vLLM sleep, was OOMing host RAM)
03c7b47 fix(rl): un-nest ref/fsdp from rollout (broke RolloutConfig schema)
88dbb59 fix(rl): correct YAML indentation under actor_rollout_ref.actor
d2c1161 fix(rl): disable FSDP CPU offload + cap vLLM gpu_mem to 0.6 (host RAM OOM during wake_up)
549a3db fix(rl): patch Triton ptx_get_version to accept CUDA 13.0 (NGC mismatch)
38abb35 fix(rl): enforce_eager=true (Triton 3.x doesn't recognize NGC CUDA 13.0)
ca631e2 fix(rl): patch vLLM lora_model_runner_mixin to skip buggy dummy LoRA warmup
cfbc9b7 fix(rl): target_modules=all-linear (q/k/v split breaks vLLM Qwen3MoE PackedLoRA dummy)
190fc23 fix(rl): vllm.fully_sharded_loras=true to fix dummy lora None on Qwen3 qkv fused proj
8dd82f2 fix(rl): set vllm.lora_extra_vocab_size=0 (dummy LoRA profile run crashes on q/k/v split)
2c07c27 fix(rl): disable VLLM_ALLREDUCE_USE_SYMM_MEM (cuMem rendezvous fails under FSDP colocate)
```

## What didn't change

- `data/tau2/...` — domain JSON + user simulator guidelines are still
  baked into the image via `COPY` (already in place).
- The SFT pipeline (`Dockerfile.sft`, `sft_job.yaml`, `axolotl`
  config) is untouched; this work was scoped to RL.
- `wandb_lora_upload.py` is unchanged; its contract still works for
  both the SFT path (axolotl-saved LoRA dir) and the new RL path
  (verl-saved LoRA dir staged by the runscript).

# tau2-bench RL on-prem pipeline — autopilot run summary

End-to-end RL pipeline (pull SFT → GRPO with KL anchor → upload RL LoRA →
Weave leaderboard) running on 1×8 H100 on-prem via `art.local.LocalBackend`.

**Final image**: `ghcr.io/kaiii66/tau2-art:f02d7e2-fix21-sftpin`
**Final passing run**: `tau2-art-rl-05141922` (pod Succeeded, all 4 stages clean)
**Leaderboard**: https://wandb.ai/kwt/tau2-ART-distill-05111101/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1

---

## Result

40 validation tasks × 3 trials per row, evaluated via W&B Inference.

| Row                | success.mean | task_reward.mean | Δ vs SFT |
|--------------------|--------------|------------------|----------|
| base Qwen3-30B     | **9.2%**     | 0.233            | —        |
| SFT @ step 16      | **27.5%**    | 0.557            | baseline |
| **RL @ step 17**   | **38.3%**    | **0.624**        | **+10.8 pp** |

RL improvement is one GRPO update on top of SFT (step 16 → step 17) with
`kl_penalty_coef=0.04` and `learning_rate=1e-7`. KL anchor was live throughout
(`loss/kl_policy_ref ≠ 0`, `loss/grad_norm` stayed under 2.0).

---

## What broke and how it was fixed

The journey took 21 image rebuilds (`fix1`…`fix21`). Pattern: every problem
was a small wiring/config issue, not a fundamental architecture flaw — once
all of them were stacked, the pipeline went green end-to-end.

### 1. Build-time

| Bug | Fix |
|---|---|
| `uv sync --frozen` failed because the local `tau2` package wants `README.md` and `src/` which weren't yet in the layer | Split into two: `uv sync --frozen --no-dev --no-install-project` (deps only) then `uv pip install --no-deps .` after `COPY . .` |
| First `uv pip install openpipe-art[backend]` worked, but a later `uv sync` wiped it (uv.lock didn't list vllm/unsloth/torch) | Use `uv pip install --no-deps .` for the project, never run `uv sync` after the backend install |
| `transformers` import-time check rejected `huggingface-hub==1.14.0` (it wants `<1.0`) | Pin `huggingface-hub>=0.34.0,<1.0` in the same install command |
| `weave` import failed: `TransportConnectionFailed` not in installed `gql==3.5.3` | openpipe-art[backend] pins `gql<4`, but `weave` needs `>=4.0`. Force-override with `uv pip install --no-deps --upgrade "gql>=4.0.0"` in a separate layer |

### 2. Runtime — process management

| Bug | Fix |
|---|---|
| `uv run python ...` in subprocess calls re-syncs the venv against `uv.lock` and clobbers the pinned `huggingface-hub`/`gql` versions, breaking imports | Use `python` directly (venv is on `$PATH` from the Dockerfile). Set `UV_RUN: list[str] = []` in `run_pipeline_local.py` |
| Each rl-stage process never exited cleanly — `LocalBackend._monitor_openai_server` is a `while True: await asyncio.sleep(30)` task with no cancellation; vLLM EngineCore + multiprocessing.resource_tracker reparent to init and keep the parent's stdout-pipe open; `subprocess.wait()` blocks forever; upload_rl + leaderboard stages never ran | (a) `start_new_session=True` when orchestrator launches the rl stage so killing the child group doesn't take down the orchestrator. (b) At end of `train_tau2_local.py`'s `__main__`, write a `.rl_complete` sentinel file then `os.killpg(getpgrp(), SIGKILL)` + `os._exit(0)`. (c) In `run_pipeline_local.py`, treat `rc == -9` as success **iff** the sentinel exists |
| Kubelet served stale cached image despite `imagePullPolicy: Always` (same tag `f02d7e2` rebuilt locally, but pod kept getting the first push) | Always re-tag for new pushes: `f02d7e2-fix1`, `-fix2`, ... `-fix21` |

### 3. Runtime — memory / config

| Bug | Fix |
|---|---|
| Default `max_model_len=262144` (Qwen3 advertised context) made vLLM try to reserve 24 GiB KV cache, OOMing on a single H100 | Set `engine_args.max_model_len` explicitly. Final value: `24576` |
| `gpu_memory_utilization=0.85 + max_model_len=49152` left 8 GiB for trainer → first 44K-token batch OOMed → vLLM EngineCore went permanently dead → every subsequent batch failed with "EngineCore encountered an issue" but the loop kept churning (0 successful gradient steps across 97 batches in one run) | Re-balance to **0.82 + 24576** + `max_orchestrator_steps=40` + `max_tokens=1024`. Math: 60 GiB model + 2.5 GiB KV + 1 GiB cuda-graphs ≈ 64 GiB → vLLM gets 0.82·79 = 65 GiB. Trainer gets ~14 GiB |
| `gpu_memory_utilization=0.78` left only 0.73 GiB for KV cache, can't even fit `max_model_len=32768` (needs 3 GiB) | (Same — settled on 0.82) |
| ART's vLLM dedicated-mode path (`inference_gpu_ids=[7]`) crashed at subprocess startup, log gone with the pod | Stayed single-GPU; never resolved (see "Suggestions") |
| Stuck rollout retried 30× with 60s backoff = 30 min stalled per failed call | `tau2_art_helpers.py` fail-fast on deterministic 400s (`context length`, `input_tokens`, `model ID is invalid`); other 400s (e.g. "Already borrowed" vLLM LoRA concurrency) still retry |
| `max_orchestrator_steps=30` truncates tau2 conversations before the agent can solve a task → val collapses from 70% to 15% | Set to 40 (sweet spot with 24K context) |

### 4. Auth / artifacts / leaderboard

| Bug | Fix |
|---|---|
| `tau2_art_helpers.ARTAgent` hard-coded `inference_api_key = WANDB_API_KEY`. LocalBackend's local vLLM server expects key `"default"`, so the agent got 401 from its own local engine | Prefer `model.inference_api_key` (set by the backend during `prepare_backend_for_training`) and fall back to env: `inference_api_key = model.inference_api_key or os.getenv("WANDB_API_KEY")` |
| tau2 data dir was looked up at `Path(__file__).parents[3]/data` which resolves to `/workspace/.venv/lib/python3.12/data` (doesn't exist) when the package is installed | K8s template sets `TAU2_DATA_DIR=/workspace/data` |
| `upload_rl_lora.py` uploaded artifacts without `wandb.base_model` metadata → W&B Inference returned 400 `model ID is invalid: model ID must be given or included in artifact metadata` | Add `metadata={"wandb.base_model": base_model, ...}` + `storage_region="coreweave-us"` (mirroring `art/utils/deployment/wandb.py`) |
| Leaderboard's base row crashed with `'NoneType' object has no attribute 'inference_api_key'` because `agent_llm = config.get("agent_llm", default)` returns `None` (not the default) when yaml has `agent_llm: null` | `agent_llm = config.get("agent_llm") or f"wandb/{base_model}"` |
| Leaderboard's SFT row looked for `step16` in the **per-run** isolated collection (`...-rl-05141922`), where step 16 doesn't exist — it lives in the original SFT collection (`...-20260511-1101`) | Construct a separate `art.TrainableModel` for the SFT row using `config["sft_source"]["name"]` |

---

## What works now

- Full pipeline: `pull_sft → rl (with KL anchor) → upload_rl → leaderboard`,
  reaches the published Weave leaderboard end-to-end with non-null scores
  for all three rows.
- Run isolation via the `-rl-<MMDDHHMM>` suffix on `model_name` — each pipeline
  invocation gets its own `.art/` checkpoint dir and W&B sub-collection without
  any manual cleanup between runs.
- KL anchor reliably non-zero throughout training; grad-norm spikes recover.
- Early-stopping (3 consecutive vals without improvement) terminates within
  ~1.5h and writes `.best_rl_step` to the snapshot directory.
- The `.rl_complete` sentinel + `start_new_session=True` combination makes
  the rl stage cleanly hand off to upload_rl + leaderboard, instead of
  hanging the pod forever on the orphan vLLM child.
- The k8s `Job` template (`onprem/k8s/art-rl-job.yaml.template`) + `k8s_submit.py`
  give a one-command submission flow: `uv run python local/k8s_submit.py --image-tag …`.

---

## Suggestions / next steps

### Reproducibility (mission item 3)
- AUTOPILOT.md asks for RL > SFT in **at least 2 independent runs**. We have 1.
  One more clean run with the current image (~3h) would confirm.

### Throughput / GPU utilisation
- **Only 1 of 8 H100s is doing real work right now.** The other 7 are reserved
  by the Job's `nvidia.com/gpu: 8` request and sit idle. Two ways to reclaim:
  1. Reduce `nvidia.com/gpu` in the template to 1 so other workloads can
     share the node.
  2. Get ART's dedicated multi-GPU vLLM subprocess working (trainer on GPUs
     0-3 with FSDP, inference on GPU 7). The subprocess crashed at startup in
     my attempts; the truncated `vllm-dedicated.log` was the blocker. Reproduce
     once with the `fix10-debuglogs` trap actually firing (see bug below)
     so the log survives pod death, then debug.

### Robustness
- **The trap-save in `run_art_rl.sh` never fires** because `exec python …` replaces
  the bash that owns the trap. Either drop the `exec` (let bash stay alive and
  shell out), or move the debug-log copy into the Python entrypoint (where
  `atexit` is reliable).
- The 3-retry trainer block in `train_tau2_local.py` doesn't restart vLLM —
  one OOM bricks the engine for the whole run. Either reinitialize the backend
  on `OutOfMemoryError`, or trip a hard exit so k8s reschedules.

### Score quality
- Current best val is 22.5%; the conservative `max_orchestrator_steps=40` and
  `max_tokens=1024` cut off agent reasoning. With dedicated mode + multi-GPU
  trainer we could afford a longer context window (say 49K) and 100 turns again,
  which historically produced val/success up to 90.9% (`tau2-art-rl-05140142`).

### Tidy-up
- Nothing committed yet — everything is on `feature/rl` uncommitted. Recommend
  committing the working state of:
  - `onprem/Dockerfile.art-rl`
  - `onprem/scripts/run_art_rl.sh`
  - `onprem/k8s/art-rl-job.yaml.template`
  - `local/*.py`
  - `train_config_local.yaml`
  - `tau2_art_helpers.py` (1-line fix for `inference_api_key`)
  - `create_leaderboard_shaped_reward.py` (`agent_llm` fallback + SFT pin)
  Then the journey above can be reproduced from a single commit.

### Other observations
- The per-run prefilter band keeps a wildly different fraction of tasks each
  time (5–66 of 74). Consider widening to `[0.05, 0.95]` or using a
  smaller-batch warmup before the band cutoff so we don't end up training on
  only 5 tasks on unlucky days.
- "Already borrowed" vLLM concurrency 400s are still common during eval. The
  retry-with-backoff handles them, but they slow leaderboard runs down. Worth
  asking the ART team / W&B Inference whether there's a way to disable
  intra-LoRA concurrency.

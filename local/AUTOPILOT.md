# Local RL Autopilot — tau2-bench

## What this is

Local GRPO training on 8×H100 using `art.local.LocalBackend` with a KL anchor
against an existing SFT checkpoint. Fixes the SFT regression caused by the
serverless backend silently ignoring `kl_beta`.

All commands run from `/home/coder/art/`.

---

## Existing SFT checkpoint (starting point)

| Field | Value |
|---|---|
| W&B collection | `kwt/tau2-ART-distill-05111101/tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260511-1101` |
| Step | 16 (confirmed) |
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |

---

## Files created

```
art/
  .dockerignore                   Excludes .venv/, .art/, pipeline_runs/, .env from build context
  train_config_local.yaml         LocalBackend config (kl_penalty_coef, sft_source, etc.)
  local/
    __init__.py
    pull_sft_lora.py              Stage 1: download SFT LoRA from W&B into .art/
    train_tau2_local.py           Stage 2: GRPO with LocalBackend + KL anchor
    upload_rl_lora.py             Stage 3: upload best RL checkpoint back to W&B
    run_pipeline_local.py         Orchestrator: pull_sft → rl → upload_rl → leaderboard
    k8s_submit.py                 Render + kubectl apply the ART RL Job on the cluster
    AUTOPILOT.md                  This file
  onprem/
    Dockerfile.art-rl             Image: nvidia/cuda:12.8.1 + uv + tau2 deps + openpipe-art[backend]
    scripts/
      run_art_rl.sh               Container entrypoint → run_pipeline_local.py
    k8s/
      art-rl-job.yaml.template    Job template (8×GPU, tau2 namespace, tau2-artifacts PVC)
```

Existing files are **untouched**: `train_tau2.py`, `run_pipeline.py`,
`create_leaderboard_shaped_reward.py`, `train_config.yaml`.

## Run isolation — every run starts fresh from SFT step 16

`run_pipeline_local.py` automatically appends the timestamp suffix to `model_name`
in the snapshot config, giving each run its own isolated `.art/` directory:

```
.art/tau2-ART-distill-05111101/models/
  tau2-distill-...-20260511-1101-rl-05121452/   ← run 1
    checkpoints/0016/   (seeded from SFT)
    checkpoints/0017/   (RL step 1)
    checkpoints/0018/   (RL step 2)

  tau2-distill-...-20260511-1101-rl-05131030/   ← run 2 (next day)
    checkpoints/0016/   (seeded from SFT — fresh start)
    checkpoints/0017/   (RL step 1)
```

`sft_source` in the config always points to the original SFT collection, so
`pull_sft_lora.py` downloads the same step-16 baseline every time regardless
of which run is in progress. **No manual cleanup is needed between runs.**

---

## Docker build setup & autopilot troubleshooting

### Prerequisites (check once before first build)

```bash
# Confirm Docker can reach the GPU base image
docker pull nvidia/cuda:12.8.1-devel-ubuntu24.04

# Confirm GHCR credentials are active
echo $CR_PAT | docker login ghcr.io -u kaiii66 --password-stdin
# CR_PAT = GitHub Personal Access Token with write:packages scope
# If not set, create one at https://github.com/settings/tokens
# and store it: echo "export CR_PAT=ghp_..." >> ~/.bashrc

# Confirm kubectl can reach the tau2 namespace
kubectl get pods -n tau2
```

### Build command (run from art/)

```bash
cd /home/coder/art
IMAGE_TAG=$(git rev-parse --short HEAD)
docker build \
    --progress=plain \
    -f onprem/Dockerfile.art-rl \
    -t ghcr.io/kaiii66/tau2-art:$IMAGE_TAG \
    . 2>&1 | tee /tmp/docker-build.log
```

`--progress=plain` shows every `RUN` step's stdout so failures are obvious.
The full log is in `/tmp/docker-build.log`.

### Build layers and expected durations

| Layer | What happens | ~Time (cold) |
|---|---|---|
| `apt-get install` | python3.12, git, curl, build-essential | 2–4 min |
| `curl astral.sh` (uv install) | Downloads + installs uv binary | 30 sec |
| `uv sync --frozen` | Installs ~40 tau2 deps from uv.lock | 3–5 min |
| `uv pip install openpipe-art[backend]` | **vllm==0.17.0 + unsloth + torch** | **20–40 min** |
| `COPY . .` | Copies source code | 10 sec |

Total cold build: ~30–45 min. Subsequent builds reuse cached layers.

### Common build failures and fixes

**`apt-get install` fails with `404 Not Found`**
```
# Ubuntu package mirror is stale in the base image — add --no-cache to bust it:
docker build --no-cache ...
```

**`uv sync --frozen` fails: "lockfile is not up to date"**
```bash
# Regenerate the lock file locally, commit, then rebuild:
cd /home/coder/art
uv lock
git add uv.lock && git commit -m "regenerate uv.lock"
```

**`uv pip install openpipe-art[backend]` fails with torch wheel not found**
```
# vllm==0.17.0 requires torch>=2.8.0 which ships on PyPI for Linux/CUDA.
# If the PyPI wheel is absent for the CUDA version, add the torch index:
# Edit onprem/Dockerfile.art-rl, change the pip install line to:
RUN uv pip install "openpipe-art[backend]==0.5.17" \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

**`uv pip install` fails with unsloth compilation error**
```
# Unsloth compiles CUDA kernels — needs build-essential + python3.12-dev.
# Both are already in the Dockerfile. If the error mentions a missing header,
# try upgrading the base image tag:
#   FROM nvidia/cuda:12.8.1-devel-ubuntu24.04   ← already correct (devel, not runtime)
# If it's a triton / flash-attn compile error, pin unsloth to the version in
# openpipe-art[backend] exactly (2026.3.3) — do NOT upgrade unsloth independently.
```

**`docker push` fails: "denied: permission_denied"**
```bash
# Re-login to GHCR:
echo $CR_PAT | docker login ghcr.io -u kaiii66 --password-stdin
docker push ghcr.io/kaiii66/tau2-art:$IMAGE_TAG
```

**`docker push` fails: "unknown blob" or hangs**
```bash
# Network / registry issue — retry push (it resumes from the last pushed layer):
docker push ghcr.io/kaiii66/tau2-art:$IMAGE_TAG
```

**k8s Job stuck in `ContainerCreating` (image pull error)**
```bash
kubectl describe pod -n tau2 -l app.kubernetes.io/component=art-rl
# Look for "ImagePullBackOff" or "ErrImagePull".
# Fix: ensure the 'ghcr' imagePullSecret exists in the tau2 namespace:
kubectl get secret ghcr -n tau2
# If missing, recreate it:
kubectl create secret docker-registry ghcr \
    --docker-server=ghcr.io \
    --docker-username=kaiii66 \
    --docker-password=$CR_PAT \
    -n tau2
```

**k8s Job pod OOMs immediately (exit code 137)**
```
# Increase memory request in the Job YAML template or reduce batch size.
# Edit onprem/k8s/art-rl-job.yaml.template:
#   memory: 400Gi  →  try reducing rollouts_per_group in train_config_local.yaml first.
```

**`loss/kl_policy_ref` is always 0 in W&B**
```
# The KL anchor is not being applied. Check that:
# 1. train_config_local.yaml has  kl_penalty_coef: 0.04  (not 0)
# 2. local/train_tau2_local.py passes kl_penalty_coef to backend.train()
# 3. The .sft_endpoint_step sidecar file exists in the pipeline_runs snapshot
```

### Autopilot agent instructions for Docker build

When asked to "autopilot the Docker build":

1. Read `/home/coder/art/local/AUTOPILOT.md` (this file) for full context.
2. Run the build with `--progress=plain` and capture stderr to `/tmp/docker-build.log`.
3. If the build fails, search the log for the **first** `ERROR` line; that is the
   root cause. Apply the matching fix from the table above.
4. Re-run the build after each fix. Repeat until the build succeeds.
5. Once the build succeeds, run `docker push` and confirm the image is in GHCR.
6. Run `uv run python local/k8s_submit.py --image-tag ... --dry-run` to validate
   the rendered Job YAML before submitting.
7. Submit the Job and confirm the pod reaches `Running` state within 5 minutes.
8. Tail logs until either `=== done ===` appears (success) or an error is printed.

---

## Run commands

### On-cluster via Kubernetes (production path)

```bash
cd art

# 1. Build and push the image (do once per code change)
IMAGE_TAG=$(git rev-parse --short HEAD)
docker build -f onprem/Dockerfile.art-rl \
             -t ghcr.io/kaiii66/tau2-art:$IMAGE_TAG .
docker push ghcr.io/kaiii66/tau2-art:$IMAGE_TAG

# 2. Submit the Job (renders YAML + kubectl apply)
uv run python local/k8s_submit.py \
    --image-tag ghcr.io/kaiii66/tau2-art:$IMAGE_TAG

# 3. Monitor (Job name is tau2-art-rl-<MMDDHHMM>)
kubectl get job -n tau2 -l app.kubernetes.io/component=art-rl -w
kubectl logs -f job/tau2-art-rl-<SUFFIX> -n tau2

# Dry-run: print rendered YAML without submitting
uv run python local/k8s_submit.py --image-tag ... --dry-run
```

### Locally (dev / smoke test — GPU node or interactive session)

```bash
cd art

# Full pipeline
uv run python local/run_pipeline_local.py

# Smoke test first (4 tasks, skip upload + leaderboard)
uv run python local/run_pipeline_local.py --num-tasks 4 --skip upload_rl leaderboard

# Re-run leaderboard only after a completed run
uv run python local/run_pipeline_local.py \
    --resume pipeline_runs/<MMDDHHMM> --skip pull_sft rl upload_rl

# Pull SFT LoRA only (dry-run)
uv run python local/pull_sft_lora.py --dry-run
```

---

## How the stages connect

```
pipeline_runs/<MMDDHHMM>/
  train_config_local.yaml    copy of config for this run
  .sft_endpoint_step         written by pull_sft_lora.py  → KL reference step + leaderboard SFT pin
  .last_trained_model        written by pull_sft_lora.py  → leaderboard model name
  .best_rl_step              written by train_tau2_local.py on val improvement → upload + leaderboard RL pin
  01-pull_sft.log
  02-rl.log
  03-upload_rl.log
  04-leaderboard.log
```

`create_leaderboard_shaped_reward.py` auto-discovers all three sidecar files and
produces three Weave leaderboard rows: **base**, **sft @ step16**, **rl @ best step**.

---

## Leaderboard metrics

| Metric | Meaning |
|---|---|
| `success.mean` | Binary headline — did the agent solve the task? |
| `task_reward.mean` | Shaped continuous score (diagnostic) |

**Target**: RL row must beat SFT row on `success.mean` consistently across runs.

---

## Key hyperparameters to tune

All in `train_config_local.yaml`:

| Param | Default | Guidance |
|---|---|---|
| `kl_penalty_coef` | `0.04` | Raise to `0.1` if policy drifts; lower to `0.01` if learning stalls |
| `learning_rate` | `1e-7` | Conservative; try `3e-7` if KL is stable and reward is flat |
| `rollouts_per_group` | `16` | Reduce to `8` to cut GPU time on smoke tests |
| `rl_prefilter_keep_band` | `[0.10, 0.90]` | Widen to `[0.05, 0.95]` if too few tasks survive the filter |
| `early_stop_patience_evals` | `3` | Stop before regression; increase to `5` for longer runs |

---

## W&B metrics to watch

| Metric | What it tells you |
|---|---|
| `loss/kl_policy_ref` | **Must be non-zero** — confirms KL anchor is live |
| `loss/grad_norm` | Should stay below ~2.0; spike = policy collapse |
| `train/reward` | Training shaped reward; should trend up |
| `val/reward` | Validation shaped reward; drives early-stop and best-step |
| `train/success` | Binary success on training tasks |
| `data/step_num_groups_trainable` | Should be > 0; if 0 every step, widen prefilter band |

---

## GPU layout for 30B MoE on 8×H100

`LocalBackend` uses Unsloth (single-process) by default. If the first run OOMs
or hangs at rollout, split trainer vs. inference GPUs in `local/train_tau2_local.py`:

```python
model = art.TrainableModel(
    name=model_name,
    project=config["project"],
    base_model=config["base_model"],
    _internal_config={
        "trainer_gpu_ids": [0, 1, 2, 3],
        "inference_gpu_ids": [4, 5, 6, 7],
    }
)
```

---

## Autopilot mission

0. **First-time image build** — build and push `ghcr.io/kaiii66/tau2-art:<sha>`.
   Only needed when `Dockerfile.art-rl` or `uv.lock` changes.

1. **Smoke test** — submit a k8s Job with `--num-tasks 4 --skip upload_rl leaderboard`
   via `SKIP_STAGES` env (or run locally). Confirm `.art/` checkpoints appear
   on the PVC and `loss/kl_policy_ref` is non-zero in W&B. Fix any OOM / import
   errors first.

2. **Full run** — submit without SKIP_STAGES. Confirm RL row beats SFT on
   `success.mean` in Weave leaderboard under project `tau2-ART-distill-05111101`.

3. **Iterate** — tune hyperparameters in `train_config_local.yaml`, rebuild the
   image if needed, and submit a new Job. Each k8s Job submission generates its
   own PIPELINE_SUFFIX, so runs are always isolated. Compare in W&B under
   `tau2-ART-distill-05111101`. Target: RL beats SFT on `success.mean` in at
   least 2 independent runs.

4. **Lock in** — update `train_config_local.yaml` with the winning values and
   add a comment explaining each tuned value (match the comment style already in
   that file).

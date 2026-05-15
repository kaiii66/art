# tau2-bench end-to-end runbook

Two pipelines, run in order. Skip step 1 if SFT already exists in W&B.

```
[teacher data] ──► [SFT job] ──► SFT LoRA on W&B  ──► [Local RL pipeline] ──► Leaderboard
                                  (step 16)              (this repo)
```

---

## 0. One-time setup

Required on your dev pod (the box where you run these commands).

```bash
# 1. Env vars (in /home/coder/art/.env)
WANDB_API_KEY=wandb_v1_...     # personal W&B token; must have access to entity 'kwt'
HF_TOKEN=hf_...                # HuggingFace token (Qwen3-30B-A3B-Instruct-2507 weights)
WANDB_ENTITY=kwt

# 2. GHCR auth (for docker push)
echo $CR_PAT | docker login ghcr.io -u kaiii66 --password-stdin

# 3. K8s secrets in tau2 namespace (one-time)
kubectl get secret wandb hf ghcr -n tau2   # should exist already
```

---

## 1. SFT (one-time per dataset / base model)

Produces a LoRA checkpoint at W&B path
`kwt/<project>/<sft-collection-name>:step16`.

If you already have one (e.g. `tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260511-1101:step16`),
**skip this section**.

To run SFT from scratch, use the existing cloud pipeline:

```bash
cd /home/coder/art

# Uploads teacher trajectories → runs SFT → stops before RL/leaderboard.
# This produces the SFT artifact your RL pipeline will start from.
uv run python run_pipeline.py --skip rl leaderboard
```

After completion, note the **collection name** that was created (in
`pipeline_runs/<MMDDHHMM>/.last_trained_model`) — you'll point the RL pipeline
at that name in step 2.

---

## 2. Local on-prem RL pipeline

Runs: `pull_sft → rl (GRPO with KL anchor) → upload_rl → leaderboard`.

### 2a. Update `train_config_local.yaml`

Edit one block to point at the SFT collection from step 1:

```yaml
project: "tau2-ART-distill-05111101"
base_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"
model_name: "tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260511-1101"

sft_source:
  entity: "kwt"
  project: "tau2-ART-distill-05111101"
  name:   "tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260511-1101"
  step:   "latest"   # or a specific integer like 16
```

The remaining hyperparameters in that file are the validated ones from this
work — leave them alone unless you know why you're changing them
(see `RUN_SUMMARY.md` for the rationale).

### 2b. Build & push the container image

```bash
cd /home/coder/art

IMAGE_TAG=ghcr.io/kaiii66/tau2-art:$(git rev-parse --short HEAD)
docker build --progress=plain -f onprem/Dockerfile.art-rl -t "$IMAGE_TAG" . \
  2>&1 | tee /tmp/docker-build.log
docker push "$IMAGE_TAG"
```

⚠️ kubelet caches images by tag and ignores `imagePullPolicy: Always` when the
tag is unchanged. If you rebuilt without bumping the git SHA (uncommitted
changes), append a fresh suffix:

```bash
IMAGE_TAG=ghcr.io/kaiii66/tau2-art:$(git rev-parse --short HEAD)-$(date +%s)
docker build ...; docker push ...
```

### 2c. Smoke test (optional, ~30 min)

Run 4 tasks only, skip the upload + leaderboard stages.

```bash
uv run python local/k8s_submit.py \
    --image-tag "$IMAGE_TAG" \
    --skip-stages "upload_rl leaderboard" \
    --num-tasks 4

# Tail logs (job name will be tau2-art-rl-MMDDHHMM)
kubectl get job -n tau2 -l app.kubernetes.io/component=art-rl -w
kubectl logs -f job/tau2-art-rl-<MMDDHHMM> -n tau2
```

Success criteria:
- A `[best-step] new best val/reward=…` line appears
- `loss/kl_policy_ref` is non-zero (visible in the wandb run linked from
  the log)
- Pod ends with `phase=Succeeded`

### 2d. Full run (~3 h)

```bash
uv run python local/k8s_submit.py --image-tag "$IMAGE_TAG"

# Monitor
kubectl get job -n tau2 -l app.kubernetes.io/component=art-rl -w
kubectl logs -f job/tau2-art-rl-<MMDDHHMM> -n tau2
```

The run prints stage markers (`=== stage: rl ===`, `=== stage: upload_rl ===`,
`=== stage: leaderboard ===`, `=== done ===`). When you see
`Leaderboard published: ObjectRef(…)` the job is essentially done.

Final outputs:

| Output | Where |
|--------|-------|
| RL LoRA checkpoints | W&B artifacts: `kwt/<project>/<model_name>-rl-<MMDDHHMM>:step{N}` |
| Best step + val reward | Printed to log; sidecar `pipeline_runs/<MMDDHHMM>/.best_rl_step` (inside the pod, lost on pod death) |
| Weave leaderboard | `https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1` |
| W&B run | `https://wandb.ai/kwt/<project>/runs/<id>` (link printed in log) |

---

## 3. Re-running just the leaderboard

If training finished but you want to re-eval (e.g. after improving the
leaderboard script), the only supported path right now is a **fresh full
run** — because the snapshot dir lives inside the pod's local FS and is
lost on pod death.

Workaround if your RL LoRA is already on W&B at a known step:

```bash
# Edit train_config_local.yaml to add:
#   leaderboard_trained_model_name: tau2-distill-...-rl-<SUFFIX>
#   leaderboard_trained_model_step: 17    # the step you want to eval

# Then submit with --skip-stages "pull_sft rl upload_rl" and a custom suffix:
uv run python local/k8s_submit.py \
    --image-tag "$IMAGE_TAG" \
    --skip-stages "pull_sft rl upload_rl" \
    --suffix 05151200
```

(There's a TODO to make this nicer — see RUN_SUMMARY.md.)

---

## 4. Debugging a failing run

The k8s pod prints to stdout (visible via `kubectl logs`). When training
fails, three places to look:

1. **`kubectl logs -f job/<name> -n tau2`** — main stream.
2. Inside the running pod (while alive):
   ```bash
   POD=$(kubectl get pod -n tau2 -l job-name=<name> -o jsonpath='{.items[0].metadata.name}')
   kubectl exec -n tau2 $POD -- bash -c 'cat /workspace/pipeline_runs/<suffix>/02-rl.log | tail -100'
   ```
3. **W&B run** linked from the log — has metrics over time
   (`loss/kl_policy_ref`, `loss/grad_norm`, `train/reward`, `val/reward`).

Common failure modes and their causes are documented in `RUN_SUMMARY.md`
("What broke and how it was fixed").

---

## Quick reference

```bash
# Full pipeline, fresh image, all stages:
cd /home/coder/art
IMAGE=ghcr.io/kaiii66/tau2-art:$(git rev-parse --short HEAD)
docker build -f onprem/Dockerfile.art-rl -t $IMAGE . && docker push $IMAGE
uv run python local/k8s_submit.py --image-tag $IMAGE

# Smoke test (4 tasks, no leaderboard, ~30 min):
uv run python local/k8s_submit.py --image-tag $IMAGE \
    --skip-stages "upload_rl leaderboard" --num-tasks 4

# Watch:
kubectl get job -n tau2 -l app.kubernetes.io/component=art-rl -w
kubectl logs -f job/tau2-art-rl-<MMDDHHMM> -n tau2

# Leaderboard URL when done:
echo https://wandb.ai/kwt/tau2-ART-distill-05111101/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1
```

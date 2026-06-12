# tau2-bench end-to-end runbook

Two pipelines, run in order. Skip step 1 if SFT already exists in W&B.

```
[teacher data] ──► [SFT job] ──► SFT LoRA on W&B ──► [RL pipeline] ──► leaderboard ──► leaderboard_crn (CRN × 3) ──► paired_analysis + upload
                                   (serverless)         (on-prem GPU)
```

The single entry-point `run_full_pipeline.py` handles all stages. It supports two
submission backends for the RL job: **vanilla Kubernetes** (`--k8s`, the original path)
and **SUNK / Slurm-on-Kubernetes** (`--slurm`, the current production path on the
`training` cluster).

---

## 0. One-time setup

### 0a. Environment variables (`.env`)

```bash
# /home/coder/art/.env
WANDB_API_KEY=wandb_v1_...   # W&B token; must have write access to entity 'kwt'
HF_TOKEN=hf_...              # HuggingFace token (for Qwen3-30B-A3B-Instruct-2507)
WANDB_ENTITY=kwt
GHCR_USER=kaiii66            # GitHub Container Registry username

# Frontier-baseline leaderboard rows (optional but recommended)
OPENAI_API_KEY=sk-...        # adds gpt-4.1-mini row to every leaderboard
GEMINI_API_KEY=...           # adds gemini-3.5-flash row to every leaderboard
```

`OPENAI_API_KEY` and `GEMINI_API_KEY` are **optional** — the pipeline runs fine
without them, but the leaderboard will only show base / sft / rl rows. Add them
to get the frontier-baseline comparison rows automatically.

### 0b. GHCR auth (for docker push + enroot image pull)

The Docker image must be pushed to GHCR and pulled on the cluster by enroot.
Both require a base64-encoded auth entry in `~/.docker/config.json` — **not** a
credential helper (`credsStore`).

```bash
# One-time login (writes base64 auth to ~/.docker/config.json):
echo $GHCR_PAT | docker login ghcr.io -u kaiii66 --password-stdin

# Verify it wrote a base64 auth (not a credsStore pointer):
python3 -c "import json; d=json.load(open('$HOME/.docker/config.json')); print(d['auths']['ghcr.io'])"
# Should print: {'auth': 'a2Fp...'}  NOT {'credsStore': '...'}
```

### 0c. KUBECONFIG

```bash
# SUNK (training) cluster — use this for --slurm runs:
export KUBECONFIG=/home/coder/.kube/training
kubectl config use-context training

# Original ray cluster — use this for vanilla K8s runs:
export KUBECONFIG=/home/coder/.kube/config-cwb607-ray
```

### 0d. SUNK cluster: bootstrap `tau2` namespace (one-time per cluster)

The SUNK cluster needs PVCs and secrets created once before the first run:

```bash
export KUBECONFIG=/home/coder/.kube/training

# Namespace + PVCs
kubectl create namespace tau2
kubectl apply -n tau2 -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tau2-artifacts
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: shared-vast
  resources:
    requests:
      storage: 500Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: tau2-data
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: shared-vast
  resources:
    requests:
      storage: 100Gi
EOF

# Secrets
source /home/coder/art/.env
kubectl create secret generic wandb --from-literal=api="$WANDB_API_KEY" -n tau2
kubectl create secret generic hf    --from-literal=token="$HF_TOKEN"    -n tau2
python3 -c "
import json, base64
d = json.load(open('$HOME/.docker/config.json'))
auth = d['auths']['ghcr.io']['auth']
user, _, token = base64.b64decode(auth).decode().partition(':')
print(token.strip())
" | xargs -I{} kubectl create secret docker-registry ghcr \
    --docker-server=ghcr.io --docker-username=kaiii66 --docker-password={} -n tau2
```

### 0e. Vanilla K8s cluster: bootstrap `tau2` namespace (one-time)

Same as above but apply to the `ray` cluster context. The `wandb`, `hf`, and `ghcr`
secrets must exist in the `tau2` namespace before submitting a job.

---

## 1. Full pipeline — one command

### SUNK / Slurm (current production path)

```bash
cd /home/coder/art
export KUBECONFIG=/home/coder/.kube/training

SUFFIX=$(TZ=America/Los_Angeles date +%m%d%H%M)
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --tail \
    2>&1 | tee pipeline_runs/full_run_$SUFFIX.log
```

This runs all stages end-to-end (~4–6 h total):
1. Generate val split (idempotent, <1 s)
2. Upload train/val/test/base datasets to W&B + Weave
3. SFT via serverless backend (~1 h)
4. Patch `train_config_local.yaml` with SFT artifact coordinates
5. Docker build + push to GHCR (~30 min)
6. Submit Slurm batch job via the login pod (`sbatch` through `kubectl exec`)
7. On the cluster: pull SFT LoRA → GRPO RL training → upload RL checkpoint → leaderboard → CRN leaderboard (3 seeds, sft+rl, paired analysis) → upload paired-analysis artifact to W&B

When it finishes, results are at:
```
https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1
https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-crn-v1
```

### Vanilla Kubernetes (original path, `ray` cluster)

```bash
cd /home/coder/art
export KUBECONFIG=/home/coder/.kube/config-cwb607-ray

SUFFIX=$(TZ=America/Los_Angeles date +%m%d%H%M)
uv run python run_full_pipeline.py --suffix $SUFFIX --tail \
    2>&1 | tee pipeline_runs/full_run_$SUFFIX.log
```

---

## 2. Resume points (reuse the same `$SUFFIX`)

After any failure, resume from the right stage rather than restarting from scratch:

```bash
# SFT done — fix RL/infra/code, then rebuild image and resubmit:
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --skip-sft --tail

# SFT done + image already correct (e.g. fixing Slurm config only):
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --skip-sft --skip-build --tail

# Dry-run: render the sbatch script without submitting:
uv run python local/slurm_submit.py \
    --image-tag ghcr.io/kaiii66/tau2-art:<sha>-$SUFFIX \
    --suffix $SUFFIX --dry-run
```

---

## 3. Slurm-specific flags

All `--slurm-*` flags have sensible defaults for this cluster. Override only when needed:

| Flag | Default | When to change |
|------|---------|----------------|
| `--slurm-namespace` | `tenant-slurm` | Different Slurm deployment namespace |
| `--slurm-login-pod` | `slurm-login-0` | Login pod was recreated with a different name |
| `--slurm-login-container` | `sshd` | Container name inside the login pod |
| `--nfs-base` | `/mnt/data/kai` | Your username/path on the shared NFS |
| `--slurm-partition` | `h100` | Different GPU partition |
| `--slurm-time-limit` | `08:00:00` | Longer for large task sets (use `16:00:00` to be safe) |

Example override:
```bash
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX \
    --nfs-base /mnt/data/myname \
    --slurm-time-limit 16:00:00 \
    --tail
```

---

## 4. Monitoring

### Slurm job status
```bash
export KUBECONFIG=/home/coder/.kube/training

# Queue
kubectl exec -n tenant-slurm slurm-login-0 -c sshd -- squeue

# Live log (JOBID is printed by slurm_submit and saved in k8s_manifest.json)
kubectl exec -n tenant-slurm slurm-login-0 -c sshd -- \
    tail -f /mnt/data/kai/logs/tau2-$SUFFIX-<JOBID>.log

# Or read JOBID from the manifest:
JOBID=$(python3 -c "import json; print(json.load(open('pipeline_runs/$SUFFIX/k8s_manifest.json'))['slurm_job_id'])")
kubectl exec -n tenant-slurm slurm-login-0 -c sshd -- \
    tail -f /mnt/data/kai/logs/tau2-$SUFFIX-$JOBID.log
```

### W&B / Weave
```
https://wandb.ai/kwt/<project>                                          ← W&B run metrics
https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1
https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-crn-v1   ← CRN leaderboard
```

### Vanilla K8s job status
```bash
kubectl get job tau2-art-rl-$SUFFIX -n tau2
kubectl logs -f job/tau2-art-rl-$SUFFIX -n tau2
```

---

## 5. Known issues and fixes

### Fused MoE LoRA load failure
**Symptom:** `ValueError: Target module ModuleList(... Qwen3MoeMLP ...) is not supported`
during the `pull_sft` or vLLM startup stage.

**Cause:** The SFT checkpoint's `adapter_config.json` has `"experts"` in `target_modules`
(added by Unsloth for Qwen3-MoE). Standard PEFT cannot load a fused MoE adapter without
first converting it.

**Fix** (already applied in `local/pull_sft_lora.py`): after download, the script calls
`art.utils.convert_moe_lora.convert_checkpoint_if_needed()` and strips `"experts"` from
`target_modules`. This is a no-op for non-MoE checkpoints. No action needed unless you
see the symptom — it means the fix didn't apply for some reason.

### Slurm wall-clock timeout
**Symptom:** job status `TIMEOUT`; RL was still training.

**Fix:** resubmit with `--slurm-time-limit 16:00:00`. A full run (RL up to 100 steps
+ leaderboard) fits comfortably in 16 h on 8× H100.

### GHCR image pull failure in the compute pod
**Symptom:** `[ERROR] URL https://ghcr.io/token returned error code: 401 Unauthorized`
in the Slurm job log.

**Fix:** the enroot credential file must be on the NFS share (not the login pod's local
filesystem). Re-run with `--skip-sft --skip-build` — `slurm_submit.py` rewrites the
credential file to `$NFS_BASE/.config/enroot/.credentials` on every submit.

---

## 6. Output artifacts

| Artifact | Location |
|----------|----------|
| SFT LoRA checkpoints | `W&B: kwt/<project>/<model_name>:step{N}` |
| RL LoRA checkpoints | `W&B: kwt/<project>/<model_name>-rl-<SUFFIX>:step{N}` |
| RL best step | `pipeline_runs/$SUFFIX/.best_rl_step` (on NFS inside the pod) |
| Weave leaderboard | `https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1` |
| CRN Weave leaderboard | `https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-crn-v1` |
| Paired-analysis artifact | `W&B: kwt/<project>/paired-analysis-<SUFFIX>` (type: paired-analysis; contains log + CRN sim JSONs) |
| Run audit trail | `pipeline_runs/$SUFFIX/AUTOPILOT_NOTES.md` |
| Docker image | `ghcr.io/kaiii66/tau2-art:<git-sha>-<SUFFIX>` |

---

## 7. Quick reference

```bash
# ── SUNK full run ──────────────────────────────────────────────────────────
export KUBECONFIG=/home/coder/.kube/training
cd /home/coder/art
SUFFIX=$(TZ=America/Los_Angeles date +%m%d%H%M)
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --tail \
    2>&1 | tee pipeline_runs/full_run_$SUFFIX.log

# ── Resume after failure (SFT done) ───────────────────────────────────────
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --skip-sft --tail

# ── Watch the Slurm job ────────────────────────────────────────────────────
JOBID=$(python3 -c "import json; print(json.load(open('pipeline_runs/$SUFFIX/k8s_manifest.json'))['slurm_job_id'])")
kubectl exec -n tenant-slurm slurm-login-0 -c sshd -- \
    tail -f /mnt/data/kai/logs/tau2-$SUFFIX-$JOBID.log

# ── Leaderboard URL ────────────────────────────────────────────────────────
python3 -c "import json; p=json.load(open('pipeline_runs/$SUFFIX/k8s_manifest.json'))['wandb_project']; print(f'https://wandb.ai/kwt/{p}/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1')"
```

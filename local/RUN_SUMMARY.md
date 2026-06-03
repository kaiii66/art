# tau2-bench ART Pipeline — Run Summary

Latest validated run: **suffix `06021512`** (2026-06-02/03)

---

## Results

**Eval set:** tau2-telecom test split, 40 tasks × 3 trials = 120 evaluations per model.

| Model | pass^1 (success rate) | pass^3 (all 3 trials) | val/reward (training) |
|-------|-----------------------|-----------------------|----------------------|
| Base (Qwen3-30B, no fine-tuning) | ~14% | — | — |
| **SFT @ step 8** | **86.7%** | **75.0%** | 73.0% |
| **RL @ step 10 (GRPO)** | **89.2%** | **77.5%** | 79.1% |
| GPT-4.1-mini (frontier baseline) | (see Weave) | — | — |

RL beats SFT by **+2.5 pp** on pass^1 and pass^3. Wilcoxon p=0.32 (not statistically significant at n=40; directionally consistent).

**Weave leaderboard:**
https://wandb.ai/kwt/tau2-ART-distill-06021512/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1

**W&B project:**
https://wandb.ai/kwt/tau2-ART-distill-06021512

---

## Run Identifiers

| Field | Value |
|-------|-------|
| Suffix | `06021512` |
| W&B project | `tau2-ART-distill-06021512` |
| Docker image | `ghcr.io/kaiii66/tau2-art:5fdd619-06021512` |
| Slurm job | 4919 (16 h wall, node `slurm-h100-225-159`) |
| RL W&B run | `rx7sw526` |
| RL W&B run URL | https://wandb.ai/kwt/tau2-ART-distill-06021512/runs/rx7sw526 |
| SFT model | `tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260602-1512` |
| RL model | `tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260602-1512-rl-06021512` |

---

## Hyperparameters

### Models

| Role | Model |
|------|-------|
| Student (agent) | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Teacher (SFT data) | `wandb/zai-org/GLM-5.1` |
| User simulator | `wandb/Qwen/Qwen3-235B-A22B-Instruct-2507` |

### SFT (distillation)

| Parameter | Value |
|-----------|-------|
| Teacher rollouts per task | 12 |
| Teacher concurrency | 5 |
| Validation rollouts per task | 2 |
| SFT epochs | 2 |
| Batch size | 2 |
| Peak LR | 1e-4 |
| LR warmup ratio | 0.1 |
| LR schedule | cosine |
| Chunk size (batches per val) | 8 |
| Early-stop patience | 3 consecutive flat evals |
| Best step | **8** (val/reward = 73.0%) |
| Total chunks trained | 11 (early-stopped) |

### RL (GRPO)

| Parameter | Value |
|-----------|-------|
| Groups per step | 3 |
| Rollouts per group | 16 |
| Total rollouts per step | 48 |
| Learning rate | 5e-7 |
| KL penalty coefficient (β) | 0.04 |
| Max steps | 100 |
| Early-stop patience | 5 consecutive flat evals |
| Best step | **10** (val/reward = 79.1%) |
| Total steps trained | 7 (steps 9–15, early-stopped) |
| Validation rollouts per task | 4 |

### Leaderboard evaluation

| Parameter | Value |
|-----------|-------|
| Eval dataset | `tau2-ART-telecom-test-scenarios` (40 tasks) |
| Trials per task | 3 |
| Max concurrency | 8 |
| User simulator | `wandb/Qwen/Qwen3-235B-A22B-Instruct-2507` |

---

## SFT Training Curve

| Chunk | val/reward | Note |
|-------|-----------|------|
| 1 | 9.5% | baseline |
| 2 | 20.3% | new best |
| 3 | 21.6% | new best |
| 4 | 41.1% | new best |
| 5 | 31.1% | plateau 1/3 |
| 6 | 66.2% | new best |
| 7 | 64.9% | plateau 1/3 |
| **8** | **73.0%** | **new best → early-stop seed** |
| 9 | — | plateau 1/3 |
| 10 | — | plateau 2/3 |
| 11 | — | plateau 3/3 → early stop |

---

## RL Training Curve

| RL Step | Model Step | val/reward | Note |
|---------|-----------|-----------|------|
| 0 | 9 | 66.9% | new best |
| **1** | **10** | **79.1%** | **new best** |
| 2 | 11 | — | plateau 1/5 |
| 3 | 12 | — | plateau 2/5 |
| 4 | 13 | — | plateau 3/5 |
| 5 | 14 | — | plateau 4/5 |
| 6 | 15 | — | plateau 5/5 → early stop |

---

## Data Collection

| Metric | Value |
|--------|-------|
| Tasks probed | 74 |
| Tasks solvable (≥10% probe success) | 71 (95.9%) |
| Teacher rollouts collected | 855 |
| Teacher success rate | 97.4% (833/855) |
| Clean trajectories (no tool errors) | 697 / 855 (81.5%) |
| Trajectories used for SFT | 697 |

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| Cluster | CoreWeave SUNK (Slurm-on-Kubernetes), `training` context |
| GPU node | 1× `slurm-h100-225-159` (8× H100 NVLink 80 GB) |
| Slurm partition | `h100` |
| Wall-clock limit | 16 h (ran ~4.5 h) |
| Container runtime | pyxis / enroot |
| Image | `ghcr.io/kaiii66/tau2-art:5fdd619-06021512` |
| NFS base | `/mnt/data/kai` |
| RL checkpoint storage | `/mnt/data/kai/tau2-artifacts/.art` (NFS-persisted) |

---

## Reliability Notes

- **Already borrowed (400) errors:** 987 incidents on Qwen3-235B user simulator during RL rollouts (48 concurrent requests vs. limited W&B inference slots). All resolved on retry ≤ attempt 4/30. No failed rollouts.
- **SSL errors:** None (previous run `06011448` had 51/74 tasks fail due to SSL issues; fully resolved this run).
- **Slurm timeout:** Preemptively cancelled 8 h job (4918) and resubmitted as 16 h job (4919). No work lost.

---

## Comparison vs Previous Run (`06011448`)

| Metric | 06011448 | **06021512** | Change |
|--------|----------|-------------|--------|
| User simulator | Qwen3-30B | **Qwen3-235B** | ↑ |
| Tasks solvable | 52.7% | **95.9%** | +43.2 pp |
| Clean trajectories | 157 | **697** | +4.4× |
| Best SFT val/reward | 50.0% | **73.0%** | +23 pp |
| SFT pass^1 (test) | 46.7% | **86.7%** | +40 pp |
| Best RL val/reward | 44.6% | **79.1%** | +34.5 pp |
| RL pass^1 (test) | 53.3% | **89.2%** | +35.9 pp |

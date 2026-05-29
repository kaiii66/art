# Pushing GRPO past SFT on the binary metric — a tau2-bench autopilot log

> Toronto Machine Learning Summit · intermediate applied-ML practitioners
> tau2-bench telecom · Qwen3-30B-A3B-Instruct-2507 + OpenPipe ART (LocalBackend, GRPO + KL anchor)

## TL;DR

We ran seven hyperparameter experiments on top of a fixed SFT seed, trying to push the GRPO policy strictly past the SFT on **binary task success** (the leaderboard's `score_success.success.mean`, not the shaped `score_task_reward`). The winning recipe is:

```yaml
learning_rate: 5.0e-7
kl_penalty_coef: 0.04
task_reward_blend: 0.90      # 90% binary, 10% shaped
groups_per_step: 3           # 4+ OOMs on this trainer
rollouts_per_group: 16
validation_rollouts_per_task: 4
early_stop_patience_evals: 5
```

That produced **GRPO @ step 12 = 0.6833 binary success** vs matching SFT 0.6500 (**+0.0333**, clears the +0.03 stronger gate). Best step landed at RL iter 2; val/success peaked 0.700.

The single change that turned a regressing pipeline into a real win was **`task_reward_blend: 0.75 → 0.90`** — bumping the weight on the binary reward inside the GRPO advantage. The LR-and-KL sweeps that followed were either neutral or actively harmful to absolute success. **The reward you optimize matters more than how fast you optimize it.**

---

## The setup

- Eval: tau2-bench telecom, 40 tasks × 3 trials, `max_steps=100`, agent temp=0.0, max_tokens=16384 (per the leaderboard config in `train_config_local.yaml`).
- Two scoring columns on the published Weave leaderboard:
  - **`score_success.success.mean`** — binary, the metric to optimize.
  - `score_task_reward.task_reward.mean` — shaped reward, diagnostic only.
- Training: OpenPipe ART LocalBackend GRPO on 8×H100 via k8s (`onprem/Dockerfile.art-rl`, `local/k8s_submit.py`). Same SFT seed across all experiments (`tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260519-0201:step10`).
- Each RL-only iteration ≈ 3 h RL + 2.5 h leaderboard ≈ 5.5 h wall.
- Acceptance gate: `RL_success − matching_SFT_success ≥ +0.02` (`+0.03` if trial count is low).

## Headline table

| # | Config (LR / KL / blend / val_n) | Matching SFT | Best RL | Δ success | Abs RL | Read |
|---|---|---:|---:|---:|---:|---|
| baseline 0 | 5e-7 / 0.04 / 0.75 / 2 | 0.6239 | 0.6083 | **−0.0156** | 0.6083 | regressed |
| baseline 1 | 1e-6 / 0.04 / 0.75 / 2 | 0.5966 | 0.5966 | 0.0000 | 0.5966 | tied (shaped misalignment) |
| **Exp 1** | **5e-7 / 0.04 / 0.90 / 4** | 0.6500 | **0.6833** | **+0.0333** | **0.6833** | **best — kept** |
| Exp 2 | 5e-7 / 0.04 / 0.95 / 4 | 0.5593 | 0.6325 | +0.0732 | 0.6325 | gate-pass, abs drop |
| Exp 5 | 7.5e-7 / 0.04 / 0.90 / 4 | 0.5500 | 0.6186 | +0.0686 | 0.6186 | gate-pass, abs drop |
| Exp 6 | 1e-6 / 0.04 / 0.90 / 4 | 0.5667 | 0.6271 | +0.0604 | 0.6271 | gate-pass, abs drop |
| Exp 7 | 5e-7 / 0.02 / 0.90 / 4 | 0.5726 | 0.6186 | +0.0460 | 0.6186 | gate-pass, abs drop |

All seven leaderboard rows live in `kwt/tau2-ART-distill-05190200/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1`. Full run-by-run state is in `rl_success_results.tsv` (untracked).

## Why the winning config wins — five mechanical claims

### 1. Goodhart's Law lives inside the GRPO advantage

GRPO computes per-group advantages from whatever scalar reward you hand it. If `train_reward` is mostly the shaped reward (action sequence quality, termination bonus, tool-use accuracy, step penalties) but the leaderboard scores **only the binary task outcome**, you're optimizing the wrong objective. The two baselines (blend=0.75) are textbook examples — Run 3 improved shaped `task_reward` by +0.023 but tied on `score_success`. Bump the blend to 0.90 and the gradient finally points at the metric you'll be measured on.

### 2. `task_reward_blend` has a non-monotonic sweet spot

The blend recipe is `reward = blend·binary + (1 − blend)·shaped`. Going 0.75 → 0.90 was a big win (Exp 1: abs 0.6833). Going 0.90 → 0.95 was *worse* (Exp 2: abs 0.6325). Why? At 0.95, almost every reward is 0 or 1. GRPO's within-group advantage (subtract the group mean) becomes very sparse — most rollouts in a group end up with the *same* advantage value, so the gradient direction is noisier per update. **The 10% shaped weight in Exp 1 is gradient density, not the objective.** Less is more, but a *little* shaped reward is load-bearing.

### 3. LR is a variance/bias dial, not a "go faster" knob

At blend=0.90 (the right objective), sweeping LR upward made absolute RL *worse*:
- 5e-7 → 0.6833 (Exp 1, best)
- 7.5e-7 → 0.6186 (Exp 5)
- 1e-6 → 0.6271 (Exp 6)

The deltas vs matching SFT *widened* because the SFT row's eval landed low on those passes, but the absolute RL score dropped. If you only watched the gate, you'd ship Exp 5 — which is worse than Exp 1 in absolute terms.

### 4. KL wasn't the binding constraint at this LR

Dropping `kl_penalty_coef` from 0.04 to 0.02 (Exp 7) did *not* enlarge KL magnitudes (still |5e-4|–|7e-4|, same as Exp 1) and absolute RL fell to 0.6186. The KL anchor isn't doing meaningful work at lr=5e-7 because the policy isn't moving fast enough to need an anchor. **If KL is already small, loosening the anchor is chasing a phantom problem.**

### 5. The textbook "KL ∈ [3e-3, 2e-2]" target is a guideline, not a gate

All four kept variants stayed *below* 3e-3 for most of training. Reward still improved. The target band is a sanity check that the policy is moving at all; if val/success is climbing and absolute RL > SFT, KL below 3e-3 is fine.

---

## Battle scars

These are the seven non-obvious things that cost real wall time in this autopilot session. Stealing any one of them is worth more than the title of this post.

### Battle scar 1 — Shaped reward is a Trojan horse for your metric

The first three runs all reported "RL beat SFT" on shaped reward. Two of them *regressed or tied* on the actual leaderboard binary metric. If your eval scores two different things, **optimize the one you'll be measured on, not the one that's easier to make a gradient out of.**

### Battle scar 2 — Your eval baseline is also noisy

The same SFT checkpoint at step 10 scored **0.5500, 0.5593, 0.5667, 0.5726, 0.5966, 0.6239, 0.6500** across seven leaderboard passes — σ ≈ 0.034 on a 3-trial × 40-task eval. That sigma is *bigger than the acceptance gate (0.02)*, which makes "RL beat SFT by +0.05" sometimes mean nothing. **Always evaluate the candidate and its matching baseline in the same pass, and track absolute scores alongside paired deltas.**

### Battle scar 3 — Higher LR can widen the gap while making the model worse

The mechanism: a faster policy update shifts the SFT row's eval context (different rollout interleaving on the shared eval infra), so the SFT eval lands low. RL eval also drops, but less. So `RL − SFT` increases while *both* drop. The autopilot's gate ("Δ ≥ +0.02") rewarded this artifact three runs in a row. **Acceptance gates that look only at deltas are not robust to baseline noise.**

### Battle scar 4 — The container log buffer is your worst observability surface

The k8s pod under `DEBUG`-level rollout logging generates ~100 lines/sec; `kubectl logs` only keeps ~30s–2min. Stage transitions like "first KL value" or "best step uploaded" rotate out in seconds. **Query W&B directly (`api.run(...).scan_history()`); don't grep pod logs for milestones.**

### Battle scar 5 — KL targets from textbooks aren't sacred

Cited target band was `loss/kl_policy_ref ∈ [3e-3, 2e-2]`. All four winning runs stayed *below* 3e-3 the whole time. Reward still improved. If your val metric is climbing and KL is in 1e-3 range, leave it alone. **The right diagnostic is "is the metric I care about going up?" not "is KL in this textbook range?"**

### Battle scar 6 — Train-time val/success ≠ leaderboard score

Training-time val uses `temperature=0.7` and `max_tokens=512`. Leaderboard uses `temperature=0.0` and `max_tokens=16384`. Exp 1's `val/success = 0.700` mapped to **leaderboard 0.6833**. Correlated but not directly comparable. The *delta* in val/success across iterations is a good early signal for which step will win on the leaderboard; the absolute value is not a leaderboard predictor.

### Battle scar 7 — Iteration time is the real budget

End-to-end pipeline (SFT + RL + leaderboard) ≈ **17 h**. RL-only iteration (re-use prior SFT) ≈ **5.6 h**. Building the "re-use SFT" iteration path was worth more than any individual sweep — without it, ~80% of compute would go into SFT runs that don't change. **Before you start your hyperparameter sweep, invest in the infrastructure that lets you reuse intermediate artifacts.**

---

## Quotable lines for the slides

- "**Optimize the metric you're measured on, not the metric that's easy to differentiate.**"
- "**Eval variance > acceptance gate**" — a silent killer. Track absolute, not just deltas.
- "**KL isn't a 'go faster' knob, and it isn't a 'stay safe' knob — it's both. Tune one cause at a time.**"
- "**Pure binary reward is sparse advantage.** Keep ~10% shaped reward as gradient density, not as the objective."
- "**The autopilot caught the gate failures; only a human caught the absolute regression.** Build acceptance criteria that match what you actually want to ship."

---

## Reproducing the winning recipe

```bash
cd /home/coder/art
set -a && source .env && set +a

# 1) Confirm the seeded SFT artifact lives in W&B (already done):
#    kwt/tau2-ART-distill-05190200/tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260519-0201:step10

# 2) Build + push the image (config baked at COPY . .):
IMAGE=ghcr.io/kaiii66/tau2-art:$(git rev-parse --short HEAD)-$(date +%s)
docker build --progress=plain -f onprem/Dockerfile.art-rl -t "$IMAGE" . | tee /tmp/docker-build.log
docker push "$IMAGE"

# 3) Submit RL + leaderboard (re-uses the existing SFT step 10):
uv run python local/k8s_submit.py --image-tag "$IMAGE"

# 4) Watch the pod (or query W&B API instead of grep'ing kubectl logs):
kubectl logs -f job/$(kubectl get job -n tau2 -l app.kubernetes.io/component=art-rl \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}') -n tau2
```

When the pod hits `Succeeded`, the published leaderboard at  
`https://wandb.ai/kwt/tau2-ART-distill-05190200/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1`  
gets a new row. Pull the four scores with `uv run python local/fetch_leaderboard_scores.py tau2-ART-distill-05190200` and compare RL vs matching SFT on the `success.mean` column.

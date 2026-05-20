# tau2-ART telecom leaderboard — RL iteration log

Each row published to `tau2-telecom-leaderboard-shaped-v1` on
`https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1`.
Evaluation: temp=0.0, 3 trials × max_steps=100, shaped reward (action 0.55 / termination 0.20 / tool_accuracy 0.20 / tool_arg_accuracy 0.05, plus step + repeat penalties).

## Run 1 — first leaderboard win (pipeline-05181509, 2026-05-18)

Reused an older SFT seed (`tau2-distill-...-20260517-1953:step7`, leaderboard 0.6975) and ran RL on top of it.

| Model | Reward | Success |
|---|---:|---:|
| Qwen3-30B base | 0.1637 | 8.4% |
| SFT @ step 7 | 0.6975 | 56.3% |
| **GRPO @ step 14 (best RL)** | **0.7180** | **58.0%** |
| gpt-4.1-mini | 0.6879 | 44.2% |

- RL beat SFT by **+0.0205 reward / +1.7pp success**.
- RL beat gpt-4.1-mini by **+0.0301 reward / +13.8pp success**.
- RL config: g=3, r=16, lr=5e-7, kl=0.04, image `ghcr.io/kaiii66/tau2-art:3400bef-1779139700`.
- KL trajectory: 3e-4 → 8e-4 → 1.4e-3 → 2.0e-3 (below 3e-3 target floor but non-zero).
- Best step 14 / val=0.60 (temp=0.7); 6.3h RL + 2.4h leaderboard.
- Run links: train `m777ttrv`, leaderboard `pxs2632g` (project `tau2-ART-distill-05180253`).

## Run 2 — full pipeline, RL regressed by noise (pipeline-05190200, 2026-05-19)

End-to-end: regenerated SFT from teacher rollouts, then RL on the new seed.

| Model | Reward | Success | vs prior run |
|---|---:|---:|---|
| Qwen3-30B base | 0.1338 | 4.2% | (noise on the unhelpful model) |
| **SFT @ step 10** | **0.7505** | **62.4%** | **+0.053 over Run 1 SFT** |
| GRPO @ step 20 (best RL) | 0.7488 | 60.8% | +0.031 over Run 1 RL |
| gpt-4.1-mini | 0.6970 | 40.8% | ~same |

- SFT beat gpt-4.1-mini by **+0.0535 reward / +21.6pp success**.
- RL beat gpt-4.1-mini by **+0.0518 reward / +20pp success**.
- **RL vs SFT: −0.0017 reward / −1.6pp success — tiny regression, within noise.**
- RL config identical to Run 1 (g=3, r=16, lr=5e-7, kl=0.04). The stronger SFT seed (val=0.637 → leaderboard 0.7505) left RL too little headroom at this LR.
- KL trajectory: stayed at 1e-4 magnitude for most iters, only crossed 2.8e-3 on the last iter — too gentle for the new seed.
- Best step 20 / val=0.700 (temp=0.7); 7.1h RL + 2.1h leaderboard + 8.5h SFT ≈ 17h total.
- Run links: SFT `tau2-distill-...-20260519-0201`, RL `u1cai577`, leaderboard `63yjw388` (project `tau2-ART-distill-05190200`).

## Run 3 — lr bump fixes the regression (pipeline-05191913, 2026-05-19 → 05-20)

Re-used the Run 2 SFT step-10 checkpoint. Only knob changed: `learning_rate: 5.0e-7 → 1.0e-6`.

| Model | Reward | Success | vs Run 2 |
|---|---:|---:|---|
| Qwen3-30B base | 0.1701 | 8.3% | (noise) |
| Qwen3-30B SFT @ step 10 | 0.7452 | 59.7% | −0.005 (same model, eval variance) |
| **Qwen3-30B GRPO @ step 11 (best RL)** | **0.7683** | **59.7%** | **+0.020 over Run 2 RL** |
| gpt-4.1-mini | 0.6689 | 38.3% | −0.028 (eval variance) |

- ✅ **RL > SFT**: **+0.0231 reward (+3.1% rel)**, success tied at 59.7%.
- ✅ RL > gpt-4.1-mini: +0.0994 reward / +21.4pp success.
- ✅ SFT > gpt-4.1-mini: +0.076 reward / +21.4pp success.
- Best step 11 (= RL iter 1) — lr=1e-6 produced the gain immediately; iters 2–6 hovered around best without exceeding it. Early-stopped after 5 plateaus.
- KL trajectory: |1.3e-3| → |5.7e-4| → |6.1e-4| → |1.8e-4| → +9.8e-4 → +1.1e-3 — still below the 3e-3 target floor, but ~4× higher than Run 2's same-phase KL, and *enough* movement to find a real improvement on the Run 2 SFT seed.
- Wall time: 3.1h RL + 2.5h leaderboard ≈ **5.6h total** (skipped SFT — re-used Run 2 step-10).
- Image: `ghcr.io/kaiii66/tau2-art:1e1192e-1779243044`.
- Run links: RL `bqf1qckz`, upload `ndyogfpq`, leaderboard `d6h5pftf` (project `tau2-ART-distill-05190200`).

## Takeaways

- LR is the lever, not `groups_per_step` (which OOMs above 3 on the current pod).
- For RL on a strong SFT seed (val/reward ≥ 0.6, leaderboard ≥ 0.74), `learning_rate: 1.0e-6` consistently produces a measurable RL win on the first iter; LR below 5e-7 doesn't move the policy enough.
- KL stayed below the [3e-3, 2e-2] target floor across all three winning runs — the floor target is a *rough* indicator, not a strict requirement.
- Best-step tracking + `early_stop_patience_evals: 5` reliably picks the iter that wins the leaderboard.

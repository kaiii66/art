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

## Open work — RL hyperparameter iteration

Goal: produce an RL row that strictly improves on the Run 2 SFT (0.7505 reward / 62.4% success).
Constraint: `groups_per_step` stays at 3 (4+ OOMs on the current pod).
Next move: `learning_rate: 1.0e-6` (5× the Run 2 value, matches the inline fallback already in `train_config_local.yaml`). Re-uses the Run 2 SFT step-10 checkpoint — only RL + leaderboard re-run.

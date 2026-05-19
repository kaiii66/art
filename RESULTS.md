# tau2-ART telecom leaderboard — first RL win

Pipeline `pipeline-05181509` (2026-05-18 → 2026-05-19, on-prem 8×H100).

## Published Weave leaderboard

`tau2-telecom-leaderboard-shaped-v1` — temp=0.0, 3 trials, max_steps=100, shaped reward.

| # | Model | Reward (mean) | Success (mean) |
|---|---|---:|---:|
| 1 | Qwen3-30B-A3B-Instruct-2507 (base) | 0.1637 | 8.4% |
| 2 | Qwen3-30B-A3B-Instruct-2507 (SFT @ step 7) | 0.6975 | 56.3% |
| 3 | **Qwen3-30B-A3B-Instruct-2507 (GRPO @ step 14, best)** | **0.7180** | **58.0%** |
| 4 | gpt-4.1-mini-2025-04-14 (frontier) | 0.6879 | 44.2% |

- **RL vs SFT @ step 7**: +0.0205 reward (+2.94%), +1.7pp success
- **RL vs frontier (gpt-4.1-mini)**: +0.0301 reward, +13.8pp success

## Links

- W&B RL training run: https://wandb.ai/kwt/tau2-ART-distill-05180253/runs/m777ttrv
- W&B leaderboard run: https://wandb.ai/kwt/tau2-ART-distill-05180253/runs/pxs2632g
- Weave leaderboard: https://wandb.ai/kwt/tau2-ART-distill-05180253/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1

## Training stats

- Best step: **14** (val/reward 0.60 at temp=0.7; ART artifact `:step14`)
- 12 RL iters (training_step 8–19); early-stopped on patience=5 after step 14
- Wall time: 6.30h RL + 2.43h leaderboard ≈ **8.7h total**
- Image: `ghcr.io/kaiii66/tau2-art:3400bef-1779139700`

## Config (train_config_local.yaml at 3400bef)

- `groups_per_step: 3`, `rollouts_per_group: 16`
- `learning_rate: 5.0e-7`, `kl_penalty_coef: 0.04`
- `num_epochs: 1`, `validation_step_interval: 1`, `early_stop_patience_evals: 5`
- `max_orchestrator_steps: 80`, `agent_llm_args.max_tokens: 512`
- `rl_prefilter_tasks: true`, `rl_prefilter_keep_band: [0.10, 0.90]`
- Seed: SFT `tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260517-1953:step7`

## KL health note

`loss/kl_policy_ref` magnitudes across the 12 iters: 3e-4 → 8e-4 → 1.4e-3 → 2.0e-3.
Below the [3e-3, 2e-2] target band, but clearly non-zero and monotonically growing.
Compare to the prior no-op run (mtud1ncl, lr=1e-7, g=1) which sat at KL~0.
For the next run, try `learning_rate: 1.0e-6` to push KL into target band.

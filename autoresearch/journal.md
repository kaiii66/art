# Autoresearch Journal — tau2-bench RL improvement loop

Append-only log of every iteration of the autoresearch loop defined in
`program.md`. One section per iteration; oldest at the top, newest at the
bottom. The agent reads this file in tandem with `BEST.md` to maintain
context about what's already been tried.

Schema for each iteration entry:

```
## Iteration <N>  —  <YYYY-MM-DD HH:MM PT>
- snapshot      : pipeline_runs/<MMDDHHMM>
- wandb_group   : pipeline-<MMDDHHMM>
- decision      : KEPT | DISCARDED
- sft_success   : <0.0–1.0>      (best val/success across SFT validations)
- rl_best_step  : <int>          (step at rl/best_val_reward)
- rl_best_reward: <float>        (rl/best_val_reward)
- delta_vs_best : <+/-float>     (rl_best_reward − previous BEST.md reward)
- diff:
    <unified diff applied this iteration>
- hypothesis (this iter):
    <one sentence; what the diff was supposed to accomplish>
- mcp diagnosis:
    <one paragraph; what the W&B/Weave evidence actually showed>
- hypothesis (next iter):
    <one sentence; what the next iteration should try and why>
```

---

(no iterations recorded yet; the first run of the autoresearch loop will
append below this line)

## Iteration 2  —  2026-04-25 01:50 UTC
- snapshot      : pipeline_runs/04241419
- wandb_group   : pipeline-04241419
- decision      : KEPT
- sft_success   : 1.000
- rl_best_step  : 41
- rl_best_reward: 0.200
- delta_vs_best : +0.075 (vs iter 1 best 0.125)
- diff:
    diff --git a/train_distill_config.yaml b/train_distill_config.yaml
    index 9817916..74bb62b 100644
    --- a/train_distill_config.yaml
    +++ b/train_distill_config.yaml
    @@ -31,7 +31,7 @@ user_llm_args:
      # Always uses the full train split from the loaded artifact (74 telecom tasks
      # at the time of writing). To bound size for a smoke test, pass
      # `--num-tasks N` on the train_tau2_distill.py CLI (kept for ad-hoc use).
    -teacher_rollouts_per_task: 3
    +teacher_rollouts_per_task: 5
      num_validation_tasks: 4
      # Bumped 30 -> 100 to match leaderboard.max_steps in train_config.yaml and
      # tau2's own default. Avoids train/eval mismatch where tasks needing 31-100
- hypothesis (this iter):
    Increasing teacher_rollouts_per_task from 3 to 5 will produce a larger, more
    diverse SFT corpus, yielding a stronger final SFT checkpoint that places more
    tasks in the RL-trainable [0.10, 0.90] prefilter band and prevents the early
    reward-variance collapse seen in iteration 1.
- mcp diagnosis:
    With teacher_rollouts_per_task=5, SFT expanded from 18 to 32 chunks (2 full
    epochs). Val success peaked at 100% (chunk 8/32) and settled at 50% for the
    final checkpoint, confirming a stronger student but with residual oscillation.
    The RL prefilter retained 40/74 tasks (54.1%, vs 6.8% in iter 1), validating
    that a better SFT seed directly enlarges the trainable band. GRPO training
    showed meaningful reward variance (train/reward_std≈0.286) throughout, unlike
    iter 1's collapse to std=0.003 at step 2. Val/reward improved steeply in
    epoch 0: 0.125 (baseline, step 33) → 0.175 (step 39) → 0.200 (step 41).
    Epoch 1 then oscillated—0.150, 0.200, 0.175—before early-stopping at 3
    consecutive non-improvements; the model appears to overfit the 40-task
    training set during the second epoch, erasing part of the epoch-0 gains.
- hypothesis (next iter):
    Since RL reward peaked at step 41 (epoch 0, step 8) and regressed throughout
    epoch 1, reducing num_epochs from 3 to 1 in train_config.yaml should lock in
    the epoch-0 gains before the model overfits the 40-task training distribution.

---

## Iteration 1  —  2026-04-24 21:16 UTC
- snapshot      : pipeline_runs/04241151
- wandb_group   : pipeline-04241151
- decision      : KEPT
- sft_success   : 1.000
- rl_best_step  : 23
- rl_best_reward: 0.125
- delta_vs_best : +inf (first iteration, previous best was -inf)
- diff:
    (none — pristine HEAD recipe, no edits applied)
- hypothesis (this iter):
    Establish pristine baseline using the unmodified feature/autoresearch HEAD recipe.
- mcp diagnosis:
    RL prefilter retained only 5/74 tasks (6.8%) with probe reward ~0.017 and ~34-token
    rollouts, indicating the final SFT model (collection step 18) was weak despite peaking
    at 100% val success at SFT chunk 4. GRPO reward variance collapsed to std=0.003 at
    training step 2 (from 0.196 at step 1), leaving near-zero gradient signal for
    subsequent steps. Val/reward climbed 0.100 at step 19 → 0.125 at step 23 before
    early-stopping at patience=3. SFT validation oscillated wildly (0%→100%→50%→50%→
    25%→0%) over the 4-task held-out set, indicating high noise and possible overfitting
    by the final epoch; W&B confirms final SFT val/success=0. Teacher probe success rate
    was 68.9% and rollout success 75%, so the teacher collected adequate data, but the
    student SFT did not retain it to the final checkpoint.
- hypothesis (next iter):
    Increasing teacher_rollouts_per_task from 3 to 5 will produce a larger, more diverse
    SFT corpus, yielding a stronger final SFT checkpoint that places more tasks in the
    RL-trainable [0.10, 0.90] prefilter band and prevents the early reward-variance
    collapse seen in this run.

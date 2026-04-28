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

## Iteration 9  —  2026-04-28 07:06 UTC
- snapshot      : pipeline_runs/04271957
- wandb_group   : pipeline-04271957
- decision      : DISCARDED
- sft_success   : 1.000
- rl_best_step  : 35
- rl_best_reward: 0.150
- delta_vs_best : -0.675 (0.150 vs 0.825 iter-8 best)
- diff:
    diff --git a/train_tau2.py b/train_tau2.py
    --- a/train_tau2.py
    +++ b/train_tau2.py
    @@ -479,6 +479,7 @@
                         result = await backend.train(
                             model,
                             finished_groups,
                             learning_rate=learning_rate,
                             ppo=True,
                             epsilon=0.2,
    +                        importance_sampling_level="sequence",
                         )
- hypothesis (this iter):
    Adding sequence-level importance sampling (GSPO) on top of PPO clipping
    would reweight IS ratios at the sequence level rather than token level,
    producing a tighter policy gradient signal and pushing val/reward above 0.825.
- mcp diagnosis:
    SFT was stronger than iter 8 (100% val success, sft_endpoint_step=28), and
    the RL prefilter retained more tasks (40/74 = 54.1% vs 30/74 = 40.5%).
    Despite the better SFT seed and larger trainable set, RL performance collapsed:
    val/reward trajectory was 0.000 → 0.050 → 0.125 → 0.150 → 0.100, peaking at
    0.150 (step 35) and early-stopping after 3 non-improvements. Sequence-level IS
    computes log-prob ratios over entire sequences (product of all per-token ratios),
    making the IS weight far smaller than 1 for long rollouts and effectively
    discounting nearly all gradient updates. With ppo=True already epsilon-clipping
    updates, the combined effect appears to have starved GRPO of usable gradient
    magnitude, causing the policy to freeze near its initial SFT distribution.
    SFT quality is NOT the limiting factor here — the constraint is in the RL update
    rule itself.
- hypothesis (next iter):
    Campaign complete — this was the final requested iteration.

---

## Iteration 8  —  2026-04-28 02:54 UTC
- snapshot      : pipeline_runs/04271414
- wandb_group   : pipeline-04271414
- decision      : KEPT
- sft_success   : 1.000
- rl_best_step  : 37
- rl_best_reward: 0.825
- delta_vs_best : +0.600 (vs iter 3 best 0.225)
- diff:
    diff --git a/train_tau2.py b/train_tau2.py
    --- a/train_tau2.py
    +++ b/train_tau2.py
    @@ -479,6 +479,8 @@
                         result = await backend.train(
                             model,
                             finished_groups,
                             learning_rate=learning_rate,
    +                        ppo=True,
    +                        epsilon=0.2,
                         )
- hypothesis (this iter):
    Adding PPO clipping (ppo=True, epsilon=0.2) to backend.train() constrains
    GRPO policy updates within a trust region, preventing large destructive
    updates and stabilizing the advantage signal for more consistent improvement.
- mcp diagnosis:
    SFT converged strongly this run (best val/success=1.000, endpoint_step=30),
    giving RL a rich starting point. RL prefilter retained 30/74 tasks (40.5%).
    PPO clipping produced a remarkable val/reward trajectory: 0.600 (step 33) →
    0.575 (step 35, early-stop 1/3, counter reset on next improvement) → 0.825
    (step 37). Best val/reward=0.825 is 3.7× the previous BEST of 0.225. The
    early-stop counter reset when step 37 surpassed step 33's 0.600, indicating
    PPO's trust-region constraint enabled genuine continued improvement past the
    first local peak rather than oscillating. This is the largest single-iteration
    gain in the campaign.
- hypothesis (next iter):
    Add importance_sampling_level="sequence" (GSPO sequence-level IS) on top of
    PPO clipping to compute importance weights at the sequence level rather than
    token level, potentially producing a tighter policy gradient signal and pushing
    val/reward above 0.825.

---

## Iteration 6  —  2026-04-27 04:16 UTC
- snapshot      : pipeline_runs/04262116
- wandb_group   : pipeline-04262116
- decision      : DISCARDED
- sft_success   : 0.750
- rl_best_step  : 22
- rl_best_reward: 0.150
- delta_vs_best : -0.075 (vs iter 3 best 0.225)
- diff:
    diff --git a/train_distill_config.yaml b/train_distill_config.yaml
    --- a/train_distill_config.yaml
    +++ b/train_distill_config.yaml
    @@ -47,7 +47,7 @@
    -sft_epochs: 2
    +sft_epochs: 1
- hypothesis (this iter):
    Reducing sft_epochs from 2 to 1 should stop the SFT before epoch 2 can oscillate
    the val/success backward, producing a more consistently strong SFT checkpoint and
    a larger RL prefilter pool — mirroring the num_epochs 3→1 fix that worked for RL.
- mcp diagnosis:
    sft_epochs=1 cut SFT training to half the gradient steps: only 5 validation chunks
    (vs 9 with 2 epochs), best SFT val/success=0.750 (vs 1.000 in iter 3), and RL
    prefilter fell from 51/74 to 40/74 tasks (54.1%). The weaker SFT starting point
    propagated directly into RL: best val/reward peaked at only 0.150 (step 22) with
    a shallow trajectory (0.050→0.125→0.125(1/3)→0.150→0.125(1/3)). The epoch-2
    oscillation was not actually the problem in prior runs — the variation in SFT
    val/success was stochastic, and epoch 2 provided net-positive gradient steps when
    the run converged well (as in iter 3). Removing it reliably weakened the student.
    DISCARDED — reverted sft_epochs to 2.
- hypothesis (next iter):
    Increase early_stop_patience_evals from 3 to 5: iter 5 showed val/reward 0.125→0.225
    in consecutive steps right at early-stop 2/3, suggesting genuine mid-dip recovery
    was being cut off — 2 extra patience slots should allow the RL to complete such
    recoveries.

---

## Iteration 7  —  2026-04-27 18:48 UTC
- snapshot      : pipeline_runs/04270927
- wandb_group   : pipeline-04270927
- decision      : DISCARDED
- sft_success   : 0.000
- rl_best_step  : 31
- rl_best_reward: 0.075
- delta_vs_best : -0.150 (vs iter 3 best 0.225)
- diff:
    diff --git a/train_config.yaml b/train_config.yaml
    --- a/train_config.yaml
    +++ b/train_config.yaml
    @@ -96,7 +96,7 @@
    -early_stop_patience_evals: 3
    +early_stop_patience_evals: 5
- hypothesis (this iter):
    Increasing early_stop_patience_evals from 3 to 5 should allow RL to recover from
    mid-training dips (as seen in iter 5's 0.125→0.225 trajectory) before stopping,
    potentially reaching a higher peak reward.
- mcp diagnosis:
    Another bad-SFT draw: final SFT val/success=0.000 (model oscillated 1.000→0.000→
    0.250→0.000 across 40 validation chunks), so RL prefilter retained only 5/74 tasks
    (6.8%) — identical to the worst iter 1 outcome. With near-zero GRPO signal from
    just 5 trainable tasks, only one RL validation fired (val/reward=0.075 at step 31).
    early_stop_patience_evals played no role: the RL collapsed before the patience
    logic could matter. The bottleneck remains SFT stochasticity, not RL patience.
    DISCARDED — reverted early_stop_patience_evals to 3.
- hypothesis (next iter):
    Increase groups_per_step from 4 to 8 in train_config.yaml: more groups per step
    covers a broader slice of trainable tasks per epoch, giving RL denser coverage
    of the prefilter pool and potentially a more stable reward signal per step.

---

## Iteration 5  —  2026-04-26 07:08 UTC
- snapshot      : pipeline_runs/04260008
- wandb_group   : pipeline-04260008
- decision      : DISCARDED
- sft_success   : 0.750
- rl_best_step  : 41
- rl_best_reward: 0.225
- delta_vs_best : +0.000 (ties iter 3 best 0.225 — not strictly greater, DISCARD)
- diff:
    diff --git a/train_config.yaml b/train_config.yaml
    --- a/train_config.yaml
    +++ b/train_config.yaml
    @@ -73,7 +73,7 @@
    -rollouts_per_group: 16
    +rollouts_per_group: 24
- hypothesis (this iter):
    With LR confirmed optimal at 5e-7, increasing rollouts_per_group from 16 to 24
    gives each GRPO step a larger, lower-variance advantage estimate, producing
    cleaner gradient signal and potentially pushing past the 0.225 ceiling.
- mcp diagnosis:
    Rollouts_per_group=24 matched but did not exceed iter 3's best: final val/reward
    reached exactly 0.225 at step 41 (last step of the single epoch). RL prefilter
    retained only 44/74 tasks (59.5%, down from 51/74 in iter 3), suggesting a weaker
    SFT seed this run (best SFT val/success=0.750 vs 1.000 in iter 3). The trajectory
    was erratic early — 0.150→0.150(1/3)→0.175→0.175(1/3)→0.125(2/3) — before
    recovering to 0.225 on the final validation. The larger batch gave no net lift:
    the extra statistical power per step was offset by fewer trainable tasks. It is
    unclear whether the SFT weakness or the rollout change was the limiting factor.
    DISCARDED — reverted rollouts_per_group to 16; BEST remains iter 3 (0.225).
- hypothesis (next iter):
    N/A — iteration budget exhausted

---

## Iteration 4  —  2026-04-26 03:03 UTC
- snapshot      : pipeline_runs/04252003
- wandb_group   : pipeline-04252003
- decision      : DISCARDED
- sft_success   : 1.000
- rl_best_step  : 43
- rl_best_reward: 0.125
- delta_vs_best : -0.100 (vs iter 3 best 0.225)
- diff:
    diff --git a/train_config.yaml b/train_config.yaml
    --- a/train_config.yaml
    +++ b/train_config.yaml
    @@ -80,7 +80,7 @@
    -learning_rate: 5.0e-7
    +learning_rate: 2.0e-7
- hypothesis (this iter):
    Halving learning_rate from 5e-7 to 2e-7 should smooth the early val/reward dip
    (0.125→0.050 in iter 3) by reducing GRPO step size and keep the policy closer
    to the SFT manifold throughout training.
- mcp diagnosis:
    Lower LR badly hurt RL training: prefilter retained only 46/74 tasks (62.2%, down
    from 51/74 in iter 3), and val/reward peaked at just 0.125 (step 43) vs 0.225 in
    iter 3. The trajectory showed two phases: an initial 0.075 peak that collapsed to
    0.000/0.025 before recovering to 0.125, then decay to 0.100/0.075 as the epoch
    ended. The smaller gradient steps kept the policy so close to the SFT point that
    GRPO lacked enough push to reach higher-reward territory; the early dip from iter 3
    was smoothed but at the cost of the eventual peak. SFT training was stable (best
    val/success=1.000, 10 validation chunks), so the problem is purely in the RL phase.
    DISCARDED — reverted learning_rate to 5.0e-7.
- hypothesis (next iter):
    With LR confirmed at 5e-7 as optimal, try increasing rollouts_per_group from 16 to
    24 to give each GRPO step a larger, lower-variance advantage estimate, which should
    produce cleaner gradient signal and potentially push past the 0.225 ceiling.

---

## Iteration 3  —  2026-04-25 02:38 UTC
- snapshot      : pipeline_runs/04241938
- wandb_group   : pipeline-04241938
- decision      : KEPT
- sft_success   : 1.000
- rl_best_step  : 43
- rl_best_reward: 0.225
- delta_vs_best : +0.025 (vs iter 2 best 0.200)
- diff:
    diff --git a/train_config.yaml b/train_config.yaml
    --- a/train_config.yaml
    +++ b/train_config.yaml
    @@ -66,7 +66,7 @@
     # Bumped 1 -> 3: re-use the small set of trainable groups (post-prefilter)
     # multiple times so GRPO sees more update steps from the same data.
    -num_epochs: 3
    +num_epochs: 1
- hypothesis (this iter):
    Since RL reward peaked at step 41 (epoch 0) in iter 2 then regressed through
    epoch 1, reducing num_epochs from 3 to 1 should lock in the epoch-0 gains before
    the model overfits the 40-task training distribution.
- mcp diagnosis:
    With num_epochs=1 the RL prefilter retained 51/74 tasks (68.9%, up from 40/74 in
    iter 2), reflecting a stronger SFT seed from the same recipe but different stochastic
    teacher rollouts. Val/reward showed an early dip (0.125→0.050→0.075, triggering
    early-stop 1/3 then 2/3 before counter reset at 0.150) then a clean monotonic climb:
    0.150→0.175→0.225. Best val/reward=0.225 at step 43; epoch ended naturally when the
    next validation also returned 0.225 (early-stop 1/3). No regression as seen in iter 2's
    second epoch—removing epochs 2-3 prevented the oscillation. SFT validation oscillated
    (best 1.000, final 1.000) across 9 validation chunks.
- hypothesis (next iter):
    The early val/reward dip (0.125→0.050 at steps 33-35) before recovery suggests initial
    GRPO updates push the policy away from the SFT optimum; halving learning_rate from 5e-7
    to 2e-7 should smooth early optimization and potentially reach a higher peak without the
    initial regression.

---

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

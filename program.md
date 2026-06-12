# tau2 RL success program

This program is for an autonomous coding agent improving RL-over-SFT on
`kwt/tau2-ART-distill-05190200`.

The goal is not to maximize shaped reward. The goal is to make the RL checkpoint
beat its matching SFT checkpoint on binary leaderboard success:

```text
score_success.success.mean
```

Use `score_task_reward.task_reward.mean` only as a diagnostic. A run that
improves shaped reward but ties or loses on `score_success` is not a success.

## Current evidence

From the existing Weave leaderboard evaluations in
`kwt/tau2-ART-distill-05190200`:

```text
pipeline-05190200:
  SFT step 10:  score_success=0.6239, score_task_reward=0.7505
  RL step 20:   score_success=0.6083, score_task_reward=0.7488

pipeline-05191913:
  SFT step 10:  score_success=0.5966, score_task_reward=0.7452
  RL step 11:   score_success=0.5966, score_task_reward=0.7683
```

The latest RL run was operationally healthy and found a best validation
checkpoint early, but it did not prove an RL-over-SFT improvement on
`score_success`. It improved shaped `score_task_reward` while only tying SFT on
binary success.

## Key files

Read these before changing anything:

- `README.md` for the pipeline overview.
- `local/RUNBOOK.md` for Kubernetes/local RL operations.
- `train_config_local.yaml` for the active RL hyperparameters.
- `local/train_tau2_local.py` for the LocalBackend GRPO loop.
- `tau2_art_helpers.py` for reward shaping, `tau2_rollout`,
  `score_success`, and `score_task_reward`.
- `create_leaderboard_shaped_reward.py` for the leaderboard semantics.
- `local/fetch_leaderboard_scores.py` for reading Weave leaderboard summaries.

Important code facts:

- Local RL uses `kl_penalty_coef`; serverless `kl_beta` is not the same thing.
- `task_reward_blend` is applied in `tau2_rollout` when shaped reward is on:
  `reward = blend * binary_reward + (1 - blend) * shaped_reward`.
- Validation logs `val/success`, `val/reward`, and `val/task_reward`, and for
  local RL these are binary success/task reward values.
- The shaped leaderboard deliberately has two columns:
  `success.mean` is headline binary success, and `task_reward.mean` is shaped
  diagnostic reward.

## Working directory and setup

Work from:

```bash
cd /home/coder/art
```

Check the environment before launching expensive jobs:

```bash
test -f .env
kubectl get secret wandb hf ghcr -n tau2
kubectl get pod -n tau2
```

Do not assume local GPU training is available. The full RL pipeline runs through
Kubernetes using the existing submission scripts.

## What you may change

You may make narrowly scoped changes to:

- `train_config_local.yaml`
- small helper scripts for reporting experiment results
- reward weighting or logging code, if needed to make `score_success` selection
  clearer
- comments/docs that prevent future confusion between success and shaped reward

Prefer changing config over changing training code. Do not refactor unrelated
pipeline code during the experiment loop.

Do not change:

- dataset definitions
- leaderboard scorer semantics
- `score_success`
- evaluation max steps or temperature unless explicitly running a controlled
  evaluation-ablation
- SFT artifacts or source checkpoints unless you are starting a separate SFT
  program

## Metrics and acceptance rule

Primary metric:

```text
score_success.success.mean
```

Acceptance gate:

```text
RL score_success >= matching SFT score_success + 0.02
```

Use a 0.02 absolute improvement as the minimum practical win. Prefer 0.03+ if
leaderboard trial count is low.

Secondary diagnostics:

```text
score_task_reward.task_reward.mean
val/success
val/reward
train/success
train/task_reward
train/shaped_reward
loss/kl_policy_ref
loss/train
loss/entropy
loss/grad_norm
```

Never keep a change solely because `score_task_reward` improved.

## Results log

Create and maintain an untracked TSV file:

```text
rl_success_results.tsv
```

Header:

```text
timestamp	suffix	branch_or_commit	variant	status	sft_success	rl_success	success_delta	sft_task_reward	rl_task_reward	best_rl_step	val_success	kl_policy_ref	notes
```

Rules:

- Use tab separators, not commas.
- Use `status=keep`, `discard`, `crash`, or `inconclusive`.
- Use `inconclusive` when the run finishes but the leaderboard is missing or too
  noisy to compare.
- Do not commit `rl_success_results.tsv` unless the human explicitly asks.

## Baseline and calibration

Before starting new experiments, record the existing baselines from W&B/Weave:

```text
pipeline-05190200 SFT vs RL step 20
pipeline-05191913 SFT vs RL step 11
```

If budget allows, rerun leaderboard evaluation with `leaderboard.num_trials: 5`
for the current SFT and RL candidates. Use that result to calibrate the amount
of binary success variance.

Useful commands:

```bash
uv run python local/fetch_leaderboard_scores.py tau2-ART-distill-05190200
```

To rerun leaderboard only for an uploaded RL artifact, follow the workaround in
`local/RUNBOOK.md`: add `leaderboard_trained_model_name` and
`leaderboard_trained_model_step` to `train_config_local.yaml`, then submit with
`--skip-stages "pull_sft rl upload_rl"`.

## First experiment

Run the first new experiment as a conservative reward-alignment variant:

```yaml
learning_rate: 5.0e-7
kl_penalty_coef: 0.04
task_reward_blend: 0.90
rollouts_per_group: 16
validation_rollouts_per_task: 4
early_stop_patience_evals: 5
```

Keep `groups_per_step: 3` initially. Existing config comments say larger values
have OOM risk on the current pod.

Rationale: the observed failure mode is not obvious under-training. It is
misalignment between shaped progress and binary task completion. Fix reward
alignment before trying larger policy updates.

## Experiment queue

Try reward alignment before update aggression. Start with the success-heavy
variant; use the lower-variance control only if you need to isolate whether the
extra validation rollouts changed the result.

1. Success-heavy:
   `learning_rate=5e-7`, `kl_penalty_coef=0.04`, `task_reward_blend=0.90`,
   `validation_rollouts_per_task=4`.
2. Very success-heavy:
   `learning_rate=5e-7`, `kl_penalty_coef=0.04`, `task_reward_blend=0.95`,
   `validation_rollouts_per_task=4`.
3. Control with lower variance:
   `learning_rate=5e-7`, `kl_penalty_coef=0.04`, `task_reward_blend=0.75`,
   `validation_rollouts_per_task=4`.
4. Binary-only short check:
   `shaped_reward=false` or `task_reward_blend=1.0`, short run if supported.

Only if the success-heavy variants improve `score_success`, try controlled
update-strength variants:

```text
learning_rate=7.5e-7, kl_penalty_coef=0.04, task_reward_blend=0.90
learning_rate=1e-6,   kl_penalty_coef=0.04, task_reward_blend=0.90
learning_rate=7.5e-7, kl_penalty_coef=0.02, task_reward_blend=0.90
```

Do not jump directly to `2e-6`. Existing evidence shows `1e-6` improved shaped
reward without improving success.

## Running experiments

For local pipeline orchestration:

```bash
uv run python local/run_pipeline_local.py --project-suffix <MMDDHHMM>
```

To resume a snapshot or rerun only a subset:

```bash
uv run python local/run_pipeline_local.py --resume pipeline_runs/<MMDDHHMM>
uv run python local/run_pipeline_local.py --resume pipeline_runs/<MMDDHHMM> --skip pull_sft rl upload_rl
```

For full Kubernetes handoff through the wrapper:

```bash
uv run python run_full_pipeline.py --suffix <MMDDHHMM> --skip-sft --tail
```

For smoke testing:

```bash
uv run python run_full_pipeline.py --suffix <MMDDHHMM> --skip-sft --smoke --tail
```

If using `local/k8s_submit.py` directly, build and push an image first as
described in `local/RUNBOOK.md`.

Monitor:

```bash
kubectl get job -n tau2 -l app.kubernetes.io/component=art-rl -w
kubectl logs -f job/tau2-art-rl-<MMDDHHMM> -n tau2
```

Treat a run as valid only if:

- the RL stage completes,
- an RL artifact is uploaded,
- the leaderboard publishes or can be rerun,
- you can extract SFT and RL `score_success` from Weave.

## Analysis loop

After each run:

1. Record run suffix, W&B run id, best RL step, final val success, and
   leaderboard SFT/RL scores in `rl_success_results.tsv`.
2. Compare RL vs matching SFT on `score_success`.
3. If RL wins by at least 0.02, keep the change and consider a small follow-up
   sweep around it.
4. If RL ties or loses while `score_task_reward` improves, treat that as reward
   misalignment. Increase `task_reward_blend` or reduce shaped components.
5. If both success and shaped reward regress, discard the change or reduce
   update strength.
6. If KL remains far below roughly `0.003` after reward alignment is fixed,
   then test slightly higher LR or lower KL.
7. If KL spikes, loss becomes unstable, or success regresses, restore the last
   kept config.

## Per-task regression analysis

When a run is inconclusive or shaped reward improves without success:

1. Use Weave `Evaluation.evaluate` child traces for the matching SFT and RL
   evaluations.
2. Bucket examples into RL wins, RL losses, and ties by `score_success`.
3. For RL losses, inspect shaped submetrics from `tau2_art_helpers.py`:
   `action_fraction`, `termination_bonus`, `tool_accuracy`,
   `tool_arg_accuracy`, `max_step_penalty`, and `repeat_message_penalty`.
4. If failed tasks have high shaped scores, adjust the shaped reward rather
   than increasing LR.

## Crash handling

For simple bugs, fix and rerun the same idea. For OOM, timeout, missing artifact,
or infrastructure failure:

- record `status=crash` or `status=inconclusive`,
- preserve enough log/W&B links in notes,
- revert experimental config changes that caused the failure,
- continue with the next lower-risk idea.

Do not repeatedly retry the same infrastructure failure without new evidence.

## Autonomous loop

Loop until manually stopped:

1. Read current results and identify the best kept variant.
2. Choose exactly one new hypothesis.
3. Patch only the minimum necessary config/code.
4. Run the experiment or, if it is too expensive, run a smoke test first.
5. Wait for completion and collect leaderboard `score_success`.
6. Append one row to `rl_success_results.tsv`.
7. Keep, discard, or mark inconclusive based on the acceptance rule.
8. Write a short note in the results row explaining the next hypothesis.
9. Continue without asking whether to proceed.

The agent should only pause for human input if credentials are missing, cluster
access is unavailable, or continuing would risk deleting data or overwriting
unrelated work.

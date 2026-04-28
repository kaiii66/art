# Autoresearch Program — tau2-bench RL Improvement Loop

You are an AI research agent inside the `test/art/` workspace. Your job is to
**iteratively improve the RL-trained tau2-bench model relative to a single
SFT-then-RL baseline**, using a budget of N iterations of the existing
`run_pipeline.py`. Each iteration runs the FULL pipeline (upload → SFT → RL →
leaderboard); you propose one focused, well-scoped edit per iteration based on
evidence gathered from W&B/Weave via the `user-wandb` MCP, run the pipeline,
diagnose, then KEEP or DISCARD the edit. State is committed to
`autoresearch/journal.md` (append-only) and `autoresearch/BEST.md` (current
best, atomically replaced).

## Goal

Maximize `rl/best_val_reward` (validation reward of the best RL checkpoint)
on tau2-bench `telecom`, where the only allowed change between iterations is
to the SFT or RL training recipe — NOT to the dataset, the W&B project name,
the leaderboard scaffolding, or the base model.

## Hard guardrails (NEVER violate)

1. **Do not change `project:` in `train_config.yaml` or `train_distill_config.yaml`.**
   The project is pinned to keep one growing leaderboard. Changing it
   silently invalidates every previous iteration's metrics and breaks the
   `BEST.md` reference.
2. **Do not run `--force` on `upload_dataset_to_wandb.py`.** The dataset is
   the experimental control. Re-uploading bumps `:vN` and shifts every prior
   iteration's "what was trained on" footprint.
3. **Do not delete the `tau2-leaderboard-shaped-base-evaluated` artifact.**
   The base row is part of the leaderboard's identity; re-evaluating it
   adds duplicate rows.
4. **Do not edit `create_leaderboard_shaped_reward.py` or
   `upload_dataset_to_wandb.py`** — these are pipeline plumbing, not a
   research surface. Editable surface is listed below.
5. **Do not skip stages.** Each iteration runs the full pipeline so RL
   trains from a freshly-distilled SFT checkpoint, not from a contaminated
   collection that mixes prior iterations' RL steps. The upload stage
   auto-skips when the dataset is already in W&B; the leaderboard stage
   auto-skips the base row after the first iteration.

## Editable surface (THE research knobs)

You may modify any of:

- `train_distill_config.yaml`  — SFT recipe
  - `teacher_rollouts_per_task`, `prefilter_tasks`, `filter_successful_only`,
    `sft_epochs`, `sft_batch_size`, `sft_peak_lr`, `sft_warmup_ratio`,
    `sft_schedule_type`, `sft_chunk_size`, `validation_every_n_chunks`
- `train_config.yaml`          — RL recipe
  - `groups_per_step`, `rollouts_per_group`, `learning_rate`, `num_epochs`,
    `validation_step_interval`, `early_stop_patience_evals`, `shaped_reward`,
    `shaped_reward_weights`, `rl_prefilter_tasks`, `rl_prefilter_k`,
    `rl_prefilter_keep_band`, `max_orchestrator_steps`, `agent_llm_args`
- `tau2_art_helpers.py`        — reward shaping internals (advanced)
- `train_tau2.py` / `train_tau2_distill.py` — control flow (advanced)

## The loop (run this every iteration)

### 1. Read the current best

```
cat autoresearch/BEST.md
```

This file holds the snapshot tag, `rl/best_val_reward`, the cumulative diff
relative to the pristine `feature/autoresearch` HEAD, and the W&B group view
URL. Treat its `rl/best_val_reward` as the bar to beat.

### 2. Form a hypothesis (use the MCP, not just intuition)

Use the `user-wandb` MCP to look at the W&B project
`tau2-ART-autoresearch-telecom`. Tools (most useful first):

- `summarize_evaluation_tool` on the latest leaderboard run — what's the
  shape of the gap between sft and rl?
- `get_run_history_tool` on the BEST iteration's `rl-<suffix>-...` run —
  where does `val/reward` peak? Is there overfitting after the peak?
- `diagnose_run_tool` on the LAST attempted iteration's RL run — flag
  pathologies (reward collapse, NaN loss, stalled training).
- `compare_runs_tool` on the BEST `rl-<suffix>-...` vs the LAST attempted
  `rl-<suffix>-...` — diff the configs (which knob you changed) against the
  resulting val/reward trajectory.
- `query_wandb_tool` for ad-hoc filtered run lists, e.g. "rl runs in
  pipeline-* groups, sorted by `summary.rl/best_val_reward`".

Write a one-sentence hypothesis. Examples:
- "RL collapses after step 12 because LR is 5× the SFT LR; halving lr to
  1e-5 should keep the policy near the SFT manifold longer."
- "The shaped action_fraction reward is being maxed out without success;
  rebalancing weights toward `termination_bonus` should refocus exploration."

### 3. Make ONE focused edit

Edit exactly one of the editable surface files. Do not bundle unrelated
changes — the loop's value is in causal attribution, which one-knob-per-iter
preserves. Save the diff for the journal entry.

### 4. Run the pipeline

```
cd test/art
uv run python run_pipeline.py
```

This creates `pipeline_runs/<MMDDHHMM>/`, snapshots the configs, writes
`group: pipeline-<MMDDHHMM>` into them, runs upload (auto-skip),
sft, rl, leaderboard. The full run is long (hours); do not interrupt unless
a stage fails fatally.

### 5. Diagnose post-run with the MCP

After the pipeline returns, use the MCP again on the new
`pipeline-<MMDDHHMM>` group to extract:

- `rl/best_val_reward` and `rl/best_step` from the rl run summary
- `train/reward` trajectory shape from `get_run_history_tool` (rising?
  flat? collapsing?)
- `val/reward` curve shape — does it improve monotonically or peak early?
- The leaderboard rows for this iteration via `summarize_evaluation_tool`
- Any anomalies via `diagnose_run_tool` (CUDA OOMs, timeouts, low GPU util)

### 6. KEEP or DISCARD

- **KEEP** if `rl/best_val_reward` strictly exceeds `BEST.md`'s value.
  - Append the iteration to `autoresearch/journal.md`.
  - Atomically replace `autoresearch/BEST.md` with the new best. Include
    snapshot tag, `rl/best_val_reward`, `rl/best_step`, the cumulative diff
    against pristine HEAD, and the W&B group view URL.
- **DISCARD** otherwise.
  - Append the iteration to `autoresearch/journal.md` with `decision: DISCARDED`
    and the qualitative diagnosis (why the change hurt or did nothing).
  - **Revert the editable file** so the next iteration starts from the
    current BEST recipe, not from the discarded one:
    ```
    git checkout HEAD -- <editable file>          # if BEST is unchanged HEAD
    git checkout <BEST snapshot's commit> -- <file>  # if BEST is mid-loop
    ```
    (The `BEST.md` cumulative diff is the source of truth for which file
    contents are "current best".)

### 7. Commit + push (so the loop is visible on GitHub)

After steps 1–6 finalize the iteration on disk, persist the trail to the
remote so a human can monitor progress at
`https://github.com/kaiii66/art/commits/feature/autoresearch` while the loop
runs unattended.

What to add to the commit (the `.gitignore` already excludes the noise so
`git add -A` is safe, but be explicit so a stale untracked file from an
earlier session can't sneak in):

```bash
cd /home/coder/test/art

git add autoresearch/journal.md autoresearch/BEST.md \
        pipeline_runs/<MMDDHHMM>/manifest.json \
        pipeline_runs/<MMDDHHMM>/train_config.yaml \
        pipeline_runs/<MMDDHHMM>/train_distill_config.yaml \
        pipeline_runs/<MMDDHHMM>/02-upload.log \
        pipeline_runs/<MMDDHHMM>/03-sft.log \
        pipeline_runs/<MMDDHHMM>/04-rl.log \
        pipeline_runs/<MMDDHHMM>/05-leaderboard.log \
        pipeline_runs/<MMDDHHMM>/.last_trained_model \
        pipeline_runs/<MMDDHHMM>/.sft_endpoint_step \
        pipeline_runs/<MMDDHHMM>/.best_rl_step
```

(Some of those `pipeline_runs/<MMDDHHMM>/.*` files only exist on KEEP runs
that actually produced an SFT/RL artifact; `git add` silently skips missing
paths so this list is safe to use unchanged. If you also reverted an
editable file in the DISCARD branch of step 6, include that file here too.)

Commit message format — one commit per iteration so the GitHub commit log
IS the iteration log. The first line is the summary line and must include
decision + iter # + reward; the body links the W&B group:

```bash
git commit -m "$(cat <<'EOF'
autoresearch iter <N>: <KEPT|DISCARDED> rl_best=<X.XXX> step=<S> snapshot=<MMDDHHMM>

hypothesis: <one sentence>
diagnosis : <one sentence>
wandb     : https://wandb.ai/<entity>/tau2-ART-autoresearch-telecom?groupBy=group&groupBys=group
group     : pipeline-<MMDDHHMM>
EOF
)"
```

Push:

```bash
git push origin feature/autoresearch
```

Failure handling for the push step:
- **Network/auth flake**: retry once after 30s. If it still fails, leave
  the commit local, log the failure inside the journal entry, and
  continue. Subsequent iterations' pushes will batch-deliver any backlog.
- **Conflict / non-fast-forward**: STOP the loop and surface the conflict
  to the human. The autoresearch branch is single-writer by design; a
  conflict means someone else is pushing to the branch and the experiment
  state is no longer trustworthy.
- **NEVER `git push --force`** for any reason. NEVER amend a previous
  iteration's commit — each iteration is its own immutable record.

### 8. Stop conditions

Stop the loop and report when ANY of:

- N iterations completed (decided up front; default N = 8).
- 3 consecutive DISCARDs (suggests the local search has plateaued).
- Pipeline failure that is NOT auto-recoverable (e.g. dataset artifact gone,
  W&B service down, base model removed). Surface the error and pause for
  human input — do NOT improvise around hard infrastructure failures.
- `rl/best_val_reward` exceeds an absolute target (set per campaign in the
  initial human prompt; default: no absolute target, just relative beats).

## Output contract

Every iteration ends with EITHER:
- An updated `BEST.md` (KEEP) AND an appended `journal.md` entry, OR
- An unchanged `BEST.md` AND an appended `journal.md` entry with
  `decision: DISCARDED`.

The journal entry MUST include: iteration number, ISO timestamp, snapshot
tag, W&B group, decision, `sft/success` (best across SFT validations),
`rl/best_val_reward`, `rl/best_step`, the diff applied this iteration
(unified), the hypothesis (one sentence), the qualitative MCP diagnosis
(one paragraph), and the next hypothesis (one sentence) for iter N+1.

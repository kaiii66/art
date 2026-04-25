# Current Best — tau2-bench RL Autoresearch

This file holds the SINGLE current-best iteration. It is atomically
overwritten when a new iteration's `rl/best_val_reward` strictly exceeds the
value below. The `cumulative diff` is the patch (relative to the pristine
`feature/autoresearch` HEAD at the start of the loop) needed to reproduce
the editable-file state of the best iteration. The next iteration's loop
applies that patch first, then layers its single new edit on top.

---

## Best so far

- snapshot         : 04241419
- wandb_group      : pipeline-04241419
- wandb_url        : https://wandb.ai/kwt/tau2-ART-autoresearch-telecom/groups/pipeline-04241419
- iteration        : 2
- recorded_at      : 2026-04-25T01:50 UTC
- sft_success      : 1.000
- rl_best_step     : 41
- rl_best_reward   : 0.200
- base_recipe_ref  : `feature/autoresearch` HEAD at loop start

## Cumulative diff (relative to base_recipe_ref)

```diff
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
```

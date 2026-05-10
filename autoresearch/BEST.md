# Current Best — tau2-bench RL Autoresearch

This file holds the SINGLE current-best iteration. It is atomically
overwritten when a new iteration's `rl/best_val_reward` strictly exceeds the
value below. The `cumulative diff` is the patch (relative to the pristine
`feature/autoresearch` HEAD at the start of the loop) needed to reproduce
the editable-file state of the best iteration. The next iteration's loop
applies that patch first, then layers its single new edit on top.

---

## Best so far

- snapshot         : 04271414
- wandb_group      : pipeline-04271414
- wandb_url        : https://wandb.ai/kwt/tau2-ART-autoresearch-telecom/groups/pipeline-04271414
- iteration        : 8
- recorded_at      : 2026-04-28T02:54 UTC
- sft_success      : 1.000
- rl_best_step     : 37
- rl_best_reward   : 0.825
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
 # `--num-tasks N` on the CLI (kept for ad-hoc use).
-teacher_rollouts_per_task: 3
+teacher_rollouts_per_task: 5
 num_validation_tasks: 4
diff --git a/train_config.yaml b/train_config.yaml
--- a/train_config.yaml
+++ b/train_config.yaml
@@ -66,7 +66,7 @@
 # Bumped 1 -> 3: re-use the small set of trainable groups (post-prefilter)
 # multiple times so GRPO sees more update steps from the same data.
-num_epochs: 3
+num_epochs: 1
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
```

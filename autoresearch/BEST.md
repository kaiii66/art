# Current Best — tau2-bench RL Autoresearch

This file holds the SINGLE current-best iteration. It is atomically
overwritten when a new iteration's `rl/best_val_reward` strictly exceeds the
value below. The `cumulative diff` is the patch (relative to the pristine
`feature/autoresearch` HEAD at the start of the loop) needed to reproduce
the editable-file state of the best iteration. The next iteration's loop
applies that patch first, then layers its single new edit on top.

---

## Best so far

- snapshot         : 04241151
- wandb_group      : pipeline-04241151
- wandb_url        : https://wandb.ai/kwt/tau2-ART-autoresearch-telecom/groups/pipeline-04241151
- iteration        : 1
- recorded_at      : 2026-04-24T21:16 UTC
- sft_success      : 1.000
- rl_best_step     : 23
- rl_best_reward   : 0.125
- base_recipe_ref  : `feature/autoresearch` HEAD at loop start

## Cumulative diff (relative to base_recipe_ref)

(empty — iteration 1 used the pristine HEAD recipe unchanged)

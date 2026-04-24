# Current Best — tau2-bench RL Autoresearch

This file holds the SINGLE current-best iteration. It is atomically
overwritten when a new iteration's `rl/best_val_reward` strictly exceeds the
value below. The `cumulative diff` is the patch (relative to the pristine
`feature/autoresearch` HEAD at the start of the loop) needed to reproduce
the editable-file state of the best iteration. The next iteration's loop
applies that patch first, then layers its single new edit on top.

---

## Best so far

- snapshot         : (none yet)
- wandb_group      : (none yet)
- wandb_url        : (none yet — first KEEP populates this)
- iteration        : 0
- recorded_at      : (none yet)
- sft_success      : (n/a)
- rl_best_step     : (n/a)
- rl_best_reward   : -inf
- base_recipe_ref  : `feature/autoresearch` HEAD at loop start

## Cumulative diff (relative to base_recipe_ref)

(empty — no iterations have been kept yet; the next iteration starts from
the pristine HEAD of `feature/autoresearch`)

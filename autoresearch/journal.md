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

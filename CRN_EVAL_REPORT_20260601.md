# CRN Eval Report — SFT vs RL (telecom, 2026-06-01)

**Method:** Common Random Numbers (CRN) — each run fixes the user-simulator seed
via `tau2 run --seed <i>`, so SFT and RL within a seed see identical random user
behaviour. Pairing by `task_id` across runs cancels per-task difficulty variance,
making the RL−SFT difference estimate far more powerful than run-level stderr bars.

**Driver:** `run_crn_eval.py --n-runs 8 --num-trials 3` (seeds 0..7), launched
2026-06-01 18:58 UTC.

---

## Models

| Role | `--agent-llm` |
|------|---------------|
| **SFT** | `wandb/wandb-artifact:///kwt/tau2-ART-distill-05190200/tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260519-0201:v10` |
| **RL** | `wandb/wandb-artifact:///kwt/tau2-ART-distill-05190200/tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260519-0201-rl-05201134:v0` |
| **Base** | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| **User simulator** (held identical → the CRN control) | `wandb/Qwen/Qwen3-235B-A22B-Instruct-2507`  (temp 0.0, max_tokens 16384) |

**Eval config:** domain `telecom`, task-split `test` (40 tasks), `--num-trials 3`,
`--max-steps 100`, `--max-concurrency 8`, agent temp 0.0 / max_tokens 16384.

---

## Run outcome — PARTIAL FAILURE

The wandb inference endpoint collapsed at ~21:40 UTC (flood of HTTP 5xx / 429).
Only **seeds 0–2** produced valid sft+rl pairs; seeds 3–7 are lost (RL = 0 sims
throughout). The master process exited 0 only because `run_tau2_cli_eval.py`
returns 0 even when individual `tau2 run` invocations abort.

| Seed | SFT sims/120 | RL sims/120 | Usable pair? |
|------|-------------:|------------:|:------------:|
| 0 | 120 | 120 | ✅ |
| 1 | 120 | 120 | ✅ |
| 2 | 119 | 119 | ✅ |
| 3 | 120 | 0 | ❌ |
| 4 | 116 | 0 | ❌ |
| 5 | 0 | 0 | ❌ |
| 6 | 0 | 0 | ❌ |
| 7 | 5 | 0 | ❌ |

Broken files (seeds 3–7) quarantined in `data/simulations/crn_failed_20260601/`
(moved, not deleted). Analysis below was run on the 3 clean seeds via `--n 3`.

---

## Results (3 valid CRN seeds)

### Run-level aggregate (`aggregate_tau2cli_runs.py --labels sft rl --n 3`)

| model | avg_reward | pass^1 | pass^3 | n |
|-------|-----------:|-------:|-------:|:-:|
| sft | 0.855 ± 0.007 | 0.856 ± 0.007 | 0.663 ± 0.038 | 3 |
| rl  | **0.891 ± 0.013** | **0.889 ± 0.012** | **0.713 ± 0.088** | 3 |

### Paired-task analysis (`paired_task_analysis.py --n 3`, 40 tasks)

**pass^1 (per-trial success rate)**
- sft 0.8556 → rl 0.8899  |  **diff +3.44pp**
- 95% bootstrap CI: **[+0.00, +6.67pp]**
- Wilcoxon (one-sided, H1: rl > sft): **p = 0.022**
- **Verdict: SIGNIFICANT — RL beats SFT on pass^1.**

**pass^3 (strict reliability)**
- sft 0.6821 → rl 0.7439  |  diff +6.18pp
- 95% bootstrap CI: [−1.95, +13.96pp]
- Wilcoxon p = 0.056
- **Verdict: NOT SIGNIFICANT — CI includes 0 (underpowered at n=3).**

**McNemar on 'reliably solved' (pass^3 == 1.0)**
- RL-only reliable: 10 tasks · SFT-only reliable: 4 tasks · concordant: 26
- McNemar one-sided p = 0.090 → not enough discordant tasks to confirm.

---

## Takeaways

1. Even with only 3/8 seeds, CRN pairing is powerful enough to confirm an **RL
   pass^1 win** (p=0.022) that run-level stderr bars alone would not separate.
2. **pass^3 is inconclusive** and needs the full 8 seeds.
3. The eval is bottlenecked by inference-endpoint stability, not by the harness.

## Recommended next step — re-run failed seeds 3–7

```bash
set -a && source .env && set +a
for s in 3 4 5 6 7; do
  .venv/bin/python run_tau2_cli_eval.py --models sft rl --num-trials 3 --seed $s
done
.venv/bin/python aggregate_tau2cli_runs.py --labels sft rl --n 8
.venv/bin/python paired_task_analysis.py --n 8
```

Open questions before relaunch:
- Confirm the wandb inference endpoint has recovered.
- Consider lowering `--max-concurrency` (was 8) to avoid overloading it again.
- One task hit a `BadRequestError` at 175k input tokens (192k ctx limit) — rare,
  but kills that trial.

---
*Generated 2026-06-02. Source data: `data/simulations/tau2cli_{sft,rl}_20260601_*.json`
(seeds 0–2); logs under `logs/crn_*` and `logs/tau2cli_*`.*

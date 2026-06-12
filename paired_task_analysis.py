"""Paired-task analysis: SFT vs RL on pass^1 and pass^3.

Pools the N most recent simulation files from data/simulations/ and reports:

  - Bootstrap 95% CI on the mean RL−SFT difference (pass^1 and pass^3)
  - Wilcoxon signed-rank test (one-sided, H1: RL > SFT)
  - McNemar discordant-task count on "reliably solved" (pass^3 == 1.0)

Two file formats are supported:
  - tau2cli_{label}_*.json    : full Results JSON written by run_tau2_cli_eval.py
  - tau2cli_lb_{label}_*.json : lightweight JSON written by
                                create_leaderboard_shaped_reward.py (list of
                                {task_id, reward, trial, seed, dropped} dicts)

This is statistically more powerful than the run-level aggregate in
aggregate_tau2cli_runs.py because it uses the task as the unit of analysis
(n=40 paired units) rather than the run (n=3), cancelling per-task difficulty
variance in the paired difference.

Usage:
    .venv/bin/python paired_task_analysis.py [--n 3] [--k 3]
    .venv/bin/python paired_task_analysis.py --n 3 --leaderboard   # lb format
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import is_successful

SIM_DIR = Path(__file__).resolve().parent / "data" / "simulations"
N_BOOT = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3,
                   help="Number of most-recent runs per label to pool (default 3)")
    p.add_argument("--k", type=int, default=3,
                   help="k for pass^k reliability metric (default 3)")
    p.add_argument("--labels", nargs=2, default=["sft", "rl"],
                   help="Two model labels to compare (default: sft rl)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for bootstrap (default 0)")
    p.add_argument("--leaderboard", action="store_true",
                   help="Read tau2cli_lb_{label}_*.json files written by "
                        "create_leaderboard_shaped_reward.py instead of "
                        "the full CLI Results format.")
    p.add_argument("--sim-dir", type=Path, default=None,
                   help="Override simulation directory (default: data/simulations/ "
                        "relative to this script, or /home/coder/art/data/simulations "
                        "on workstation).")
    return p.parse_args()


def latest_files(label: str, n: int, leaderboard: bool = False, sim_dir: Path = None) -> list[Path]:
    directory = sim_dir or SIM_DIR
    pattern = f"tau2cli_lb_{label}_*.json" if leaderboard else f"tau2cli_{label}_*.json"
    files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[:n]


def load_task_trials(label: str, n: int, leaderboard: bool = False, sim_dir: Path = None) -> dict[str, list[int]]:
    """Return {task_id: [0/1, ...]} pooled over the n most-recent runs.

    When leaderboard=True, reads the lightweight tau2cli_lb_{label}_*.json
    format written by create_leaderboard_shaped_reward.py (list of
    {task_id, reward, trial, seed, dropped} dicts).
    """
    out: dict[str, list[int]] = defaultdict(list)
    files = latest_files(label, n, leaderboard=leaderboard, sim_dir=sim_dir)
    if not files:
        fmt = "leaderboard (tau2cli_lb_)" if leaderboard else "CLI (tau2cli_)"
        directory = sim_dir or SIM_DIR
        raise FileNotFoundError(
            f"No {fmt} simulation files found for label '{label}' in {directory}"
        )
    for f in files:
        if leaderboard:
            records = json.loads(f.read_text())
            for r in records:
                if not r.get("dropped", False):
                    # Prefer the binary `success` field written by current
                    # create_leaderboard_shaped_reward.py. Fall back to
                    # is_successful(reward) for older files that only stored the
                    # (shaped) reward — note those older files under-report success
                    # because a shaped 0.99 != 1.0.
                    if "success" in r:
                        out[r["task_id"]].append(int(round(float(r["success"]))))
                    else:
                        out[r["task_id"]].append(int(is_successful(r["reward"])))
        else:
            df = Results.load(f).to_df()
            df["succ"] = df["reward"].apply(is_successful).astype(int)
            for task_id, group in df.groupby("task_id"):
                out[task_id].extend(group["succ"].tolist())
    return dict(out)


def pass_hat_k(successes: int, n_trials: int, k: int) -> float:
    """Unbiased pass^k estimator (from tau2 paper).

    Probability that at least k out of k randomly chosen trials succeed.
    Returns nan if n_trials < k.
    """
    if n_trials < k:
        return float("nan")
    return math.comb(successes, k) / math.comb(n_trials, k)


def bootstrap_ci(
    diffs: np.ndarray, rng: np.random.Generator, n_boot: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI."""
    means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    return diffs.mean(), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    label_a, label_b = args.labels  # a=baseline (SFT), b=candidate (RL)

    sim_dir = args.sim_dir
    fmt = "leaderboard" if args.leaderboard else "CLI"
    print(f"\nLoading last {args.n} run(s) per label: {label_a} vs {label_b} (format: {fmt})")
    trials_a = load_task_trials(label_a, args.n, leaderboard=args.leaderboard, sim_dir=sim_dir)
    trials_b = load_task_trials(label_b, args.n, leaderboard=args.leaderboard, sim_dir=sim_dir)

    tasks = sorted(set(trials_a) & set(trials_b))
    if not tasks:
        print("ERROR: no overlapping task_ids found between the two labels.")
        return 1

    n_trials_per_task = set(len(trials_a[t]) for t in tasks) | set(len(trials_b[t]) for t in tasks)
    k = args.k

    print(f"Tasks: {len(tasks)}  |  Trials per task pooled: {n_trials_per_task}  |  k={k}")

    # Per-task statistics
    rate_a, rate_b = [], []
    pk_a, pk_b = [], []

    for t in tasks:
        ta, tb = trials_a[t], trials_b[t]
        rate_a.append(sum(ta) / len(ta))
        rate_b.append(sum(tb) / len(tb))
        pk_a.append(pass_hat_k(sum(ta), len(ta), k))
        pk_b.append(pass_hat_k(sum(tb), len(tb), k))

    rate_a = np.array(rate_a)
    rate_b = np.array(rate_b)
    pk_a = np.array(pk_a)
    pk_b = np.array(pk_b)

    d_rate = rate_b - rate_a
    d_pk = pk_b - pk_a

    print("\n" + "=" * 70)
    print(f"PAIRED TASK ANALYSIS  ({label_a} = baseline,  {label_b} = candidate)")
    print("=" * 70)

    for metric_name, baseline, candidate, diffs in [
        ("pass^1  (per-trial success rate)", rate_a, rate_b, d_rate),
        (f"pass^{k} (strict reliability)   ", pk_a, pk_b, d_pk),
    ]:
        m_a = float(np.nanmean(baseline))
        m_b = float(np.nanmean(candidate))
        mean_d, lo, hi = bootstrap_ci(diffs[~np.isnan(diffs)], rng, N_BOOT)

        try:
            w = stats.wilcoxon(diffs[~np.isnan(diffs)], alternative="greater")
            w_p = w.pvalue
        except ValueError:
            w_p = float("nan")

        print(f"\n  {metric_name}")
        print(f"    {label_a} mean : {m_a:.4f}  ({m_a*100:.1f}%)")
        print(f"    {label_b} mean : {m_b:.4f}  ({m_b*100:.1f}%)")
        print(f"    mean diff    : {mean_d:+.4f}  ({mean_d*100:+.2f}pp)")
        print(f"    95% boot CI  : [{lo:+.4f}, {hi:+.4f}]  ({lo*100:+.2f}pp to {hi*100:+.2f}pp)")
        print(f"    Wilcoxon p   : {w_p:.4f}  (one-sided H1: {label_b} > {label_a})")

        if lo > 0:
            verdict = f"SIGNIFICANT — CI entirely above 0 (p={w_p:.3f})"
        elif hi < 0:
            verdict = f"SIGNIFICANT REGRESSION — CI entirely below 0"
        else:
            verdict = f"NOT SIGNIFICANT — CI includes 0"
        print(f"    verdict      : {verdict}")

    # McNemar on "reliably solved" tasks (pass^k == 1.0)
    print(f"\n  McNemar on 'reliably solved' (pass^{k} == 1.0 across all pooled trials)")
    rel_a = pk_a == 1.0
    rel_b = pk_b == 1.0
    b = int((rel_a & ~rel_b).sum())   # baseline reliable, candidate not
    c = int((~rel_a & rel_b).sum())   # candidate reliable, baseline not
    concordant = int((rel_a == rel_b).sum())
    print(f"    Concordant tasks        : {concordant}")
    print(f"    {label_b}-only reliable  : c = {c}  ← {label_b} wins these tasks")
    print(f"    {label_a}-only reliable  : b = {b}  ← {label_a} wins these tasks")
    if b + c > 0:
        binom = stats.binomtest(c, b + c, 0.5, alternative="greater")
        print(f"    McNemar p (one-sided)   : {binom.pvalue:.4f}")
        if binom.pvalue < 0.05:
            print(f"    → {label_b} reliably solves more tasks than {label_a} (p<0.05)")
        else:
            print(f"    → not enough discordant tasks to confirm a win")
    else:
        print("    → no discordant tasks (models identical on reliability)")

    # Per-task detail for discordant pass^k tasks
    discordant_tasks = [
        (tasks[i], float(pk_a[i]), float(pk_b[i]), float(d_pk[i]))
        for i in range(len(tasks))
        if not np.isnan(d_pk[i]) and abs(d_pk[i]) > 1e-6
    ]
    if discordant_tasks:
        print(f"\n  Per-task pass^{k} discordant detail (top 10 by |diff|):")
        discordant_tasks.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"    {'task_id':<65} {label_a:>6}  {label_b:>6}  {'diff':>7}")
        for task_id, pa, pb, d in discordant_tasks[:10]:
            short_id = task_id[:64]
            print(f"    {short_id:<65} {pa:>6.3f}  {pb:>6.3f}  {d:>+7.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

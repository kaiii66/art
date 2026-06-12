"""Aggregate the N most recent tau2cli_<label>_*.json files per label and
report mean ± stderr for avg_reward and pass^k.

Usage:
    .venv/bin/python aggregate_tau2cli_runs.py --labels base sft rl --n 3
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics

SIM_DIR = Path("/home/coder/art/data/simulations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["base", "sft", "rl"])
    p.add_argument("--n", type=int, default=3, help="Most-recent runs per label")
    p.add_argument(
        "--since",
        type=float,
        default=None,
        help="Only include files with mtime >= this unix timestamp",
    )
    return p.parse_args()


def gather(labels: list[str], n: int, since: float | None) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in SIM_DIR.glob("tau2cli_*.json"):
        # tau2cli_<label>_<YYYYMMDD>_<HHMMSS>.json
        stem = f.stem  # tau2cli_<label>_<date>_<time>
        parts = stem.split("_")
        if len(parts) < 4 or parts[0] != "tau2cli":
            continue
        # label may contain hyphens (e.g. "gpt-4.1-mini") but not underscores
        label = "_".join(parts[1:-2])
        if label not in labels:
            continue
        if since is not None and f.stat().st_mtime < since:
            continue
        groups[label].append(f)
    for label, files in groups.items():
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        groups[label] = files[:n]
    return groups


def fmt(vals: list[float | None]) -> str:
    clean = [v for v in vals if v is not None]
    if not clean:
        return "—"
    if len(clean) == 1:
        return f"{clean[0]:.4f}"
    mean = statistics.mean(clean)
    se = statistics.stdev(clean) / math.sqrt(len(clean))
    return f"{mean:.4f} ± {se:.4f}"


def main() -> int:
    args = parse_args()
    groups = gather(args.labels, args.n, args.since)

    metrics = ["avg_reward", "pass^1", "pass^2", "pass^3"]

    print(f"\nAggregating last {args.n} runs per label from {SIM_DIR}\n")
    rows: dict[str, list[dict]] = {}
    for label in args.labels:
        files = groups.get(label, [])
        runs: list[dict] = []
        for f in files:
            try:
                results = Results.load(f)
                m = compute_metrics(results)
            except Exception as e:
                print(f"  [warn] {f.name}: {e}")
                continue
            run = {"file": f.name, "avg_reward": m.avg_reward}
            for k, v in m.pass_hat_ks.items():
                run[f"pass^{k}"] = v
            run["completed"] = len(results.simulations)
            run["planned"] = len(results.tasks) * results.info.num_trials
            runs.append(run)
        rows[label] = runs

    # Per-run detail
    print("Per-run detail:")
    for label in args.labels:
        print(f"\n[{label}]")
        for r in rows.get(label, []):
            line = f"  {r['file']}  ({r['completed']}/{r['planned']})  "
            line += "  ".join(
                f"{m}={r.get(m):.4f}" for m in metrics if r.get(m) is not None
            )
            print(line)

    # Aggregate table
    print("\n" + "=" * 78)
    print("MEAN ± STDERR")
    print("=" * 78)
    header_cells = ["model", *metrics, "n"]
    widths = [10, 18, 18, 18, 18, 3]
    print(" | ".join(c.ljust(w) for c, w in zip(header_cells, widths)))
    print("-+-".join("-" * w for w in widths))
    for label in args.labels:
        runs = rows.get(label, [])
        vals = {m: [r.get(m) for r in runs] for m in metrics}
        cells = [label, *[fmt(vals[m]) for m in metrics], str(len(runs))]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

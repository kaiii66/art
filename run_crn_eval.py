"""Common Random Numbers (CRN) eval driver for SFT vs RL.

Runs `run_tau2_cli_eval.py --models sft rl` N times. Run i fixes the
user-simulator seed to `i` via --seed, so SFT and RL within that run see
identical random user behaviour (the CRN property). Pooling the N runs and
pairing by task_id in `paired_task_analysis.py` then estimates the RL-SFT
difference with the shared user-simulator noise removed.

Each run streams to its own log file under logs/ so individual runs can be
inspected independently.

Usage:
    .venv/bin/python run_crn_eval.py --n-runs 8 --num-trials 3

Dry run (prints the per-run commands without executing):
    .venv/bin/python run_crn_eval.py --n-runs 3 --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/coder/art")
PYBIN = REPO / ".venv/bin/python"
EVAL_SCRIPT = REPO / "run_tau2_cli_eval.py"
LOG_DIR = REPO / "logs"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-runs", type=int, default=8,
                   help="Number of CRN runs; seeds 0..N-1 (default 8)")
    p.add_argument("--num-trials", type=int, default=3,
                   help="--num-trials forwarded to run_tau2_cli_eval.py (default 3)")
    p.add_argument("--models", nargs="+", default=["sft", "rl"],
                   help="Model rows to compare under CRN (default: sft rl)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print each per-run command without executing")
    # Remaining args are forwarded verbatim to run_tau2_cli_eval.py, so callers
    # can pass --num-tasks/--task-split/--config/etc. without duplicating them here.
    args, extra = p.parse_known_args()
    args.extra = extra
    return args


def build_run_command(seed: int, args: argparse.Namespace) -> list[str]:
    cmd = [
        str(PYBIN),
        str(EVAL_SCRIPT),
        "--models",
        *args.models,
        "--num-trials",
        str(args.num_trials),
        "--seed",
        str(seed),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd += args.extra
    return cmd


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    rc = 0
    for seed in range(args.n_runs):
        cmd = build_run_command(seed, args)
        print(f"\n{'=' * 70}\n# CRN run {seed + 1}/{args.n_runs}  (seed={seed})\n{'=' * 70}")
        print("CMD:")
        print("  " + " ".join(cmd))

        if args.dry_run:
            # Forward to the eval script's own --dry-run so the constructed
            # tau2 commands (with the seed flag) are printed for inspection.
            subprocess.run(cmd, cwd=str(REPO))
            continue

        log_path = LOG_DIR / f"crn_seed{seed}_{utc_stamp()}.log"
        print(f"log: {log_path}")
        with open(log_path, "w", buffering=1) as f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            proc.wait()
        if proc.returncode != 0:
            print(f"[warn] CRN run seed={seed} exited {proc.returncode}")
            rc = proc.returncode

    return rc


if __name__ == "__main__":
    sys.exit(main())

"""
Orchestrate `tau2 run` (tau2-bench CLI) across 5 model rows and print per-run
statistics + a final comparison table.

Smoke test (no API calls):
    .venv/bin/python run_tau2_cli_eval.py --models base --num-trials 1 \\
        --num-tasks 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv("/home/coder/art/.env", override=True)

from create_leaderboard_shaped_reward import load_config  # noqa: E402

REPO = Path("/home/coder/art")
PYBIN = REPO / ".venv/bin/python"
TAU2_BIN = REPO / ".venv/bin/tau2"
LOG_DIR = REPO / "logs"

MODEL_KEYS = ["base", "sft", "rl", "gpt-4.1-mini", "gemini"]

USER_LLM = "wandb/Qwen/Qwen3-235B-A22B-Instruct-2507"
USER_LLM_ARGS = {"temperature": 0.0, "max_tokens": 16384}
AGENT_LLM_ARGS = {"temperature": 0.0, "max_tokens": 16384}

# Termination reasons mirrored from tau2.data_model.simulation.TerminationReason.
# Listed explicitly so the report shows zeros even when a category never fires.
TERMINATION_REASONS = [
    "agent_stop",
    "user_stop",
    "max_steps",
    "too_many_errors",
    "agent_error",
    "user_error",
]

# Categories for log-file error tally. Order matters: the first match wins.
ERROR_PATTERNS = [
    ("rate_limit", re.compile(r"rate_limit|RateLimitError|\b429\b", re.IGNORECASE)),
    ("timeout", re.compile(r"\bTimeout\b|APITimeoutError", re.IGNORECASE)),
    ("connection", re.compile(r"APIConnectionError|ConnectionError", re.IGNORECASE)),
    (
        "server_error",
        re.compile(r"ServiceUnavailable|InternalServerError|\b5\d\d\b", re.IGNORECASE),
    ),
    ("other_litellm", re.compile(r"litellm\.\w*Error", re.IGNORECASE)),
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="train_config_local.yaml")
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=MODEL_KEYS,
        help="Subset of model rows to run",
    )
    p.add_argument("--num-trials", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--task-ids", nargs="+", default=None)
    p.add_argument(
        "--task-split",
        default="test",
        help="tau2 task-split-name. tau2 telecom has small/train/test/full/base; "
        "the ART 'validation' set (validation_scenarios.json, 40 tasks) matches 'test'.",
    )
    p.add_argument("--sft-version", default="v10")
    p.add_argument("--rl-version", default="v0")
    p.add_argument("--sft-ref", default=None, help="Full --agent-llm override for SFT")
    p.add_argument("--rl-ref", default=None, help="Full --agent-llm override for RL")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix user-simulator seed for CRN. Same seed is passed to every model "
        "in this invocation so SFT and RL see identical random user behaviour.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_agent_llms(cfg: dict, args: argparse.Namespace) -> dict[str, str]:
    """Return {model_key: --agent-llm string} for every requested row."""
    base_model = cfg["base_model"]  # e.g. Qwen/Qwen3-30B-A3B-Instruct-2507
    sft_src = cfg["sft_source"]
    entity = sft_src["entity"]
    sft_project = sft_src["project"]
    sft_name = sft_src["name"]
    rl_name = cfg.get("leaderboard_trained_model_name")
    rl_project = sft_project  # RL artifact lives under the same W&B project.

    sft_ref = args.sft_ref or (
        f"wandb/wandb-artifact:///{entity}/{sft_project}/{sft_name}:{args.sft_version}"
    )
    if args.rl_ref:
        rl_ref = args.rl_ref
    elif rl_name:
        rl_ref = (
            f"wandb/wandb-artifact:///{entity}/{rl_project}/{rl_name}:{args.rl_version}"
        )
    else:
        rl_ref = None

    return {
        "base": f"wandb/{base_model}",
        "sft": sft_ref,
        "rl": rl_ref,
        "gpt-4.1-mini": "openai/gpt-4.1-mini-2025-04-14",
        "gemini": "gemini/gemini-3.5-flash",
    }


def build_command(
    *,
    save_to: str,
    agent_llm: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        str(TAU2_BIN),
        "run",
        "--domain",
        "telecom",
        "--agent-llm",
        agent_llm,
        "--agent-llm-args",
        json.dumps(AGENT_LLM_ARGS),
        "--user-llm",
        USER_LLM,
        "--user-llm-args",
        json.dumps(USER_LLM_ARGS),
        "--num-trials",
        str(args.num_trials),
        "--max-steps",
        str(args.max_steps),
        "--max-concurrency",
        str(args.max_concurrency),
        "--task-split-name",
        args.task_split,
        "--save-to",
        save_to,
    ]
    if args.num_tasks is not None:
        cmd += ["--num-tasks", str(args.num_tasks)]
    if args.task_ids:
        cmd += ["--task-ids", *args.task_ids]
    # CRN: the same seed is threaded into every model row in this invocation so
    # the user simulator produces identical random sequences for each (task, trial).
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return cmd


def stream_to_console_and_log(proc: subprocess.Popen, log_path: Path) -> None:
    """Forward proc.stdout line-by-line to both this process's stdout and log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", buffering=1) as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)


def find_results_json(save_to: str) -> Optional[Path]:
    """Locate the saved tau2 results file. tau2 writes to <DATA_DIR>/simulations/."""
    candidates = [
        REPO / "data" / "simulations" / f"{save_to}.json",
        REPO / "data" / "tau2" / "simulations" / f"{save_to}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def tally_log_errors(log_path: Path) -> dict[str, int]:
    counts: Counter = Counter()
    if not log_path.exists():
        return dict(counts)
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            for label, pattern in ERROR_PATTERNS:
                if pattern.search(line):
                    counts[label] += 1
                    break
    return {label: counts.get(label, 0) for label, _ in ERROR_PATTERNS}


def report_run(
    *,
    label: str,
    save_to: str,
    log_path: Path,
    exit_code: int,
    num_trials: int,
) -> dict:
    """Print the per-run stats block and return summary fields for the final table."""
    print(f"\n{'=' * 60}\nRUN REPORT — {label}\n{'=' * 60}")
    print(f"save_to:  {save_to}")
    print(f"log:      {log_path}")
    print(f"exit:     {exit_code}")

    # Import lazily so --dry-run never touches tau2 imports.
    from tau2.data_model.simulation import Results
    from tau2.metrics.agent_metrics import compute_metrics

    results_path = find_results_json(save_to)
    if results_path is None:
        print("results:  <not written>")
    else:
        print(f"results:  {results_path}")

    results = None
    if results_path is not None:
        try:
            results = Results.load(results_path)
        except Exception as e:
            print(f"  [warn] could not parse results: {e}")

    # 1. Completion check
    if results is not None:
        planned = len(results.tasks) * results.info.num_trials
        completed = len(results.simulations)
    else:
        planned = 0
        completed = 0
    missing = planned - completed
    print(f"\nCompletion: {completed}/{planned} (missing {missing})")

    # 2. Termination-reason breakdown
    term_counts = {k: 0 for k in TERMINATION_REASONS}
    if results is not None:
        for sim in results.simulations:
            reason = getattr(sim.termination_reason, "value", str(sim.termination_reason))
            if reason in term_counts:
                term_counts[reason] += 1
            else:
                term_counts[reason] = term_counts.get(reason, 0) + 1
    print("Termination reasons:")
    for k in TERMINATION_REASONS:
        print(f"  {k:<18} {term_counts.get(k, 0)}")
    extras = [k for k in term_counts if k not in TERMINATION_REASONS]
    for k in extras:
        print(f"  {k:<18} {term_counts[k]}  (unknown)")

    # 3. Log-file error tally
    err_counts = tally_log_errors(log_path)
    rate_limit_total = err_counts.get("rate_limit", 0)
    print(f"API/log tally (rate_limited total: {rate_limit_total}):")
    for label_, _ in ERROR_PATTERNS:
        print(f"  {label_:<18} {err_counts.get(label_, 0)}")

    # 4. One-line verdict
    if exit_code == 0 and missing == 0 and results is not None:
        verdict = "OK"
        print(f"\n✓ run OK")
    else:
        verdict = f"ABORT(exit={exit_code})"
        print(
            f"\n✗ run ABORTED (exit {exit_code}) — "
            f"{rate_limit_total} rate-limit hits, {missing} sims missing otherwise"
        )

    # 5. Metrics (only if results exist).
    avg_reward: Optional[float] = None
    pass_hat_ks: dict[int, float] = {}
    if results is not None and len(results.simulations) > 0:
        try:
            m = compute_metrics(results)
            avg_reward = m.avg_reward
            pass_hat_ks = m.pass_hat_ks
            print(f"\nMetrics: avg_reward = {avg_reward:.4f}")
            for k in sorted(pass_hat_ks):
                print(f"  pass^{k} = {pass_hat_ks[k]:.4f}")
        except Exception as e:
            print(f"  [warn] compute_metrics failed: {e}")

    return {
        "label": label,
        "avg_reward": avg_reward,
        "pass_hat_ks": pass_hat_ks,
        "completed": completed,
        "planned": planned,
        "rate_limit_hits": rate_limit_total,
        "verdict": verdict,
    }


def print_final_table(rows: list[dict], num_trials: int) -> None:
    headers = ["model", "avg_reward", "pass^1", f"pass^{num_trials}", "completed", "rate_limit_hits", "verdict"]
    fmt = lambda v: "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    lines = [headers]
    for r in rows:
        p1 = r["pass_hat_ks"].get(1)
        pk = r["pass_hat_ks"].get(num_trials)
        lines.append([
            r["label"],
            fmt(r["avg_reward"]),
            fmt(p1),
            fmt(pk),
            f"{r['completed']}/{r['planned']}",
            str(r["rate_limit_hits"]),
            r["verdict"],
        ])

    widths = [max(len(row[i]) for row in lines) for i in range(len(headers))]
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for i, row in enumerate(lines):
        print(" | ".join(c.ljust(widths[j]) for j, c in enumerate(row)))
        if i == 0:
            print("-+-".join("-" * w for w in widths))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    print(f"[config]   project:           {cfg.get('project')}")
    print(f"[config]   base_model:        {cfg.get('base_model')}")
    print(f"[config]   sft_source:        {cfg.get('sft_source')}")
    print(f"[config]   rl trained name:   {cfg.get('leaderboard_trained_model_name')}")
    print(f"[config]   rl trained step:   {cfg.get('leaderboard_trained_model_step')}")

    agent_llms = resolve_agent_llms(cfg, args)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label in args.models:
        agent_llm = agent_llms.get(label)
        if agent_llm is None:
            print(f"\n[skip] {label}: no --agent-llm could be resolved")
            continue

        stamp = utc_stamp()
        save_to = f"tau2cli_{label}_{stamp}"
        log_path = LOG_DIR / f"{save_to}.log"
        cmd = build_command(save_to=save_to, agent_llm=agent_llm, args=args)

        print(f"\n{'#' * 60}\n# {label}  →  {agent_llm}\n{'#' * 60}")
        print("CMD:")
        print("  " + " ".join(_shell_quote(c) for c in cmd))

        if args.dry_run:
            continue

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Stream from a thread so we could parallelize later if desired;
        # for now we just wait on this one process.
        t = threading.Thread(target=stream_to_console_and_log, args=(proc, log_path))
        t.start()
        proc.wait()
        t.join()

        rows.append(
            report_run(
                label=label,
                save_to=save_to,
                log_path=log_path,
                exit_code=proc.returncode,
                num_trials=args.num_trials,
            )
        )

    if not args.dry_run and rows:
        print_final_table(rows, args.num_trials)

    return 0


def _shell_quote(s: str) -> str:
    if not s or any(ch in s for ch in " \t\"'$`\\{}[]<>|&;()*?#"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    sys.exit(main())

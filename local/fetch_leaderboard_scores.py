"""Fetch the 4-row leaderboard scores from Weave for a specific run.

Usage:  uv run python local/fetch_leaderboard_scores.py <project>
        e.g.  uv run python local/fetch_leaderboard_scores.py tau2-ART-distill-05162346
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv("/home/coder/art/.env")

import weave  # noqa: E402
from weave.trace.weave_client import WeaveClient  # noqa: E402

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "tau2-ART-distill-05162346"
ENTITY = os.environ.get("WANDB_ENTITY", "kwt")

client: WeaveClient = weave.init(f"{ENTITY}/{PROJECT}")

calls = list(
    client.get_calls(
        filter={"op_names": [f"weave:///{ENTITY}/{PROJECT}/op/Evaluation.evaluate:*"]},
        include_costs=False,
        include_feedback=False,
    )
)

print(f"Project: {ENTITY}/{PROJECT}")
print(f"Total Evaluation.evaluate calls: {len(calls)}")
print()

rows = []
for c in calls:
    display = (c.display_name or "").strip()
    if not display:
        continue
    summary = (c.summary or {}).get("output") or c.output or {}
    s_success = (summary.get("score_success") or {}).get("success", {}).get("mean")
    s_taskr = (summary.get("score_task_reward") or {}).get("task_reward", {}).get("mean")
    if s_success is None and s_taskr is None:
        s_success = (summary.get("success") or {}).get("mean")
        s_taskr = (summary.get("task_reward") or {}).get("mean")
    rows.append((display, s_success, s_taskr, c.started_at))

rows.sort(key=lambda r: r[3] or 0)
print(f"{'model':<70s} {'success.mean':>14s} {'task_reward.mean':>18s}")
print("-" * 105)
for display, s, t, _ in rows:
    s_str = f"{s:.4f}" if s is not None else "<missing>"
    t_str = f"{t:.4f}" if t is not None else "<missing>"
    print(f"{display:<70s} {s_str:>14s} {t_str:>18s}")

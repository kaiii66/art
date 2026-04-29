"""
Convert art.Trajectory objects produced by Phase A of train_tau2_distill.py
into the chat JSONL format Axolotl consumes (`type: chat_template` datasets).

Each output line is one JSON object:
    {
      "messages": [{"role": "system",    "content": "..."},
                   {"role": "user",      "content": "..."},
                   {"role": "assistant", "content": "...",
                                          "tool_calls": [...]},
                   {"role": "tool",      "tool_call_id": "...", "content": "..."}],
      "tools":    [<openai-tool-schema>, ...]
    }

`art.Trajectory.messages_and_choices` is a heterogeneous list: some entries
are plain dicts (system/user/tool/assistant-without-LLM), others are OpenAI
`Choice` objects (assistant turns produced by the LLM during the rollout).
This script normalizes both into plain OpenAI chat dicts, then writes one
JSONL line per trajectory.

Usage:
    # As a library:
    from onprem.scripts.trajectory_to_jsonl import write_trajectories_jsonl
    n = write_trajectories_jsonl(trajectories, out_path="/data/sft.jsonl")

    # As a CLI (loads a list of trajectories from a pickle / W&B artifact):
    python -m onprem.scripts.trajectory_to_jsonl \
        --trajectories-pickle /tmp/teacher_trajectories.pkl \
        --out /data/sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def _choice_to_assistant_dict(choice: Any) -> dict:
    """Turn an OpenAI `Choice` object into an assistant message dict."""
    msg = choice.message
    out: dict = {"role": "assistant", "content": msg.content or ""}
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    if raw_tool_calls:
        out["tool_calls"] = []
        for tc in raw_tool_calls:
            # `tc.function.arguments` is already a JSON string per the OpenAI spec.
            out["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return out


def _normalize_messages(messages_and_choices: Iterable[Any]) -> list[dict]:
    """Coerce mixed-type ART message list to OpenAI chat dicts."""
    out: list[dict] = []
    for item in messages_and_choices:
        if isinstance(item, dict):
            out.append(item)
            continue
        # Best-effort: anything else we treat as an OpenAI Choice.
        if hasattr(item, "message"):
            out.append(_choice_to_assistant_dict(item))
            continue
        raise TypeError(
            f"Unexpected item in messages_and_choices: {type(item).__name__}. "
            "Expected dict or OpenAI Choice."
        )
    return out


def trajectory_to_record(traj: Any) -> dict:
    """Convert one `art.Trajectory` to one JSON-serializable dict."""
    messages = _normalize_messages(traj.messages_and_choices)
    tools = getattr(traj, "tools", None) or []
    return {
        "messages": messages,
        "tools": tools,
        # Keep a small metadata blob so downstream tooling can filter / weight.
        "metadata": {
            "task_id": traj.metadata.get("task_id") if traj.metadata else None,
            "domain": traj.metadata.get("domain") if traj.metadata else None,
            "reward": traj.reward,
            "success": traj.metrics.get("success") if traj.metrics else None,
        },
    }


def write_trajectories_jsonl(
    trajectories: Iterable[Any],
    out_path: str | Path,
    *,
    filter_successful_only: bool = True,
) -> int:
    """Write trajectories to JSONL. Returns the number of records written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for traj in trajectories:
            if filter_successful_only:
                success = (traj.metrics or {}).get("success", 0.0)
                if success < 1.0:
                    continue
            record = trajectory_to_record(traj)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories-pickle",
        required=True,
        help="Path to a pickle file containing a list of art.Trajectory objects.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include trajectories with success<1.0 (default: drop).",
    )
    args = parser.parse_args()

    with open(args.trajectories_pickle, "rb") as fp:
        trajectories = pickle.load(fp)

    n = write_trajectories_jsonl(
        trajectories,
        out_path=args.out,
        filter_successful_only=not args.include_failed,
    )
    print(f"wrote {n} records -> {args.out}")


if __name__ == "__main__":
    _cli()

"""
Generate a reproducible 'val' split for a tau2 domain and write it into
the domain's split_tasks.json.

The val split is drawn from  full - base  (tasks not already in train or
test) using stratified random sampling that mirrors the category distribution
of the existing 'test' split.  The result is fully disjoint from train, test,
and base, giving a clean held-out validation set for checkpoint selection
during RL / SFT training — so 'test' can serve as a true leaderboard holdout.

This script is idempotent: if 'val' already exists in split_tasks.json it
prints the current counts and exits unless --force is passed.

Usage
-----
    # Telecom domain (default):
    uv run python scripts/generate_val_split.py

    # Explicit domain + config:
    uv run python scripts/generate_val_split.py --domain telecom

    # Preview without writing:
    uv run python scripts/generate_val_split.py --dry-run

    # Regenerate with a different seed (expert use):
    uv run python scripts/generate_val_split.py --force --seed 123

    # Require at least 2 faults per task (matches test's minimum fault count):
    uv run python scripts/generate_val_split.py --force --min-faults 2

Reproducibility
---------------
The default seed=42 with the default size (matching test distribution)
produces the val split that is checked into the repo. Re-running with the
same --seed and --size produces identical output as long as split_tasks.json
has not changed.

When --min-faults N is specified, the available pool is pre-filtered to only
tasks with >= N faults before stratified sampling. If a category has fewer
tasks than the target after filtering, all available tasks in that category
are taken and a warning is printed (rather than crashing).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Resolve repo root relative to this script (art/scripts/ -> art/)
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
DATA_DIR = REPO_ROOT / "data" / "tau2" / "domains"


def get_category(task_id: str) -> str:
    """Extract the category prefix from a task ID like '[category]...'."""
    if task_id.startswith("[") and "]" in task_id:
        return task_id.split("]")[0][1:]
    return "unknown"


def count_faults(task_id: str) -> int:
    """Count the number of faults in a task ID.

    Task IDs encode faults as pipe-separated segments in the body, e.g.
    '[cat]fault1|fault2[PERSONA:X]' has 2 faults.  Single-fault tasks have
    no pipe character in the body segment.
    """
    # Strip leading '[category]' and trailing '[PERSONA:...]'
    body = task_id
    if "]" in body:
        body = body[body.index("]") + 1:]
    body = body.split("[PERSONA")[0]
    return body.count("|") + 1


def find_split_tasks_file(domain: str) -> Path:
    path = DATA_DIR / domain / "split_tasks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"split_tasks.json not found for domain '{domain}' at {path}. "
            "Make sure the tau2 data directory is set up correctly."
        )
    return path


def generate_val_split(
    domain: str,
    seed: int = 42,
    size: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    min_faults: int = 0,
) -> int:
    split_file = find_split_tasks_file(domain)
    with split_file.open() as f:
        splits: dict[str, list[str]] = json.load(f)

    required = {"full", "base", "train", "test"}
    missing = required - set(splits.keys())
    if missing:
        print(f"ERROR: split_tasks.json is missing required splits: {missing}", file=sys.stderr)
        return 1

    # Guard: already exists
    if "val" in splits and not force:
        cats = Counter(get_category(t) for t in splits["val"])
        print(f"'val' split already exists in {split_file.name} ({len(splits['val'])} tasks).")
        print("Category breakdown:", dict(sorted(cats.items())))
        print("Use --force to regenerate.")
        return 0

    base = set(splits["base"])
    full = set(splits["full"])
    test = splits["test"]
    available = full - base

    # Optionally filter to tasks with >= min_faults faults
    if min_faults > 0:
        before = len(available)
        available = {t for t in available if count_faults(t) >= min_faults}
        removed = before - len(available)
        if removed:
            print(f"Filtered out {removed} tasks with < {min_faults} fault(s) from available pool ({before} -> {len(available)}).")

    # Stratify by category — match test distribution by default
    test_cat_counts = Counter(get_category(t) for t in test)
    avail_by_cat: dict[str, list[str]] = {}
    for t in available:
        avail_by_cat.setdefault(get_category(t), []).append(t)
    # Sort within each category for determinism before sampling
    for cat in avail_by_cat:
        avail_by_cat[cat].sort()

    if size is None:
        target_counts = dict(test_cat_counts)
    else:
        # Scale test proportions to the requested total size
        total_test = sum(test_cat_counts.values())
        target_counts = {
            cat: max(1, round(count / total_test * size))
            for cat, count in test_cat_counts.items()
        }
        # Adjust rounding error on largest category
        delta = size - sum(target_counts.values())
        if delta != 0:
            largest = max(target_counts, key=target_counts.get)
            target_counts[largest] += delta

    print(f"Domain        : {domain}")
    print(f"split file    : {split_file}")
    print(f"available pool: {len(available)} tasks (full - base{', min_faults>=' + str(min_faults) if min_faults > 0 else ''})")
    print(f"seed          : {seed}")
    if min_faults > 0:
        print(f"min_faults    : {min_faults}")
    print()
    print("Sampling plan (category -> requested | available):")
    shortfalls = []
    for cat, need in sorted(target_counts.items()):
        have = len(avail_by_cat.get(cat, []))
        flag = " [SHORTFALL — taking all available]" if have < need else ""
        print(f"  {cat}: {need} requested | {have} available{flag}")
        if have < need:
            shortfalls.append((cat, need, have))

    if shortfalls:
        print()
        print("WARNING: pool exhausted for some categories after fault-count filtering.")
        print("Taking all available tasks for those categories (val will be smaller than target):")
        for cat, need, have in shortfalls:
            print(f"  {cat}: wanted {need}, taking {have}")

    random.seed(seed)
    val: list[str] = []
    for cat, need in sorted(target_counts.items()):
        have = len(avail_by_cat.get(cat, []))
        actual = min(need, have)
        val.extend(random.sample(avail_by_cat.get(cat, []), actual))

    val_set = set(val)
    assert len(val_set) == len(val), "Duplicate task IDs in val split — bug in generation logic"
    assert val_set.isdisjoint(base), "val overlaps with base!"
    assert val_set.isdisjoint(set(splits["train"])), "val overlaps with train!"
    assert val_set.isdisjoint(set(splits["test"])), "val overlaps with test!"

    print()
    print(f"Generated {len(val)} val tasks:")
    cat_counts = Counter(get_category(t) for t in val)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    if dry_run:
        print("\n[dry-run] Not writing. Pass without --dry-run to save.")
        return 0

    # Insert "val" right after "train" for logical ordering in the JSON.
    # Skip any existing "val" key so --force doesn't let the old value
    # overwrite the newly generated list during iteration.
    new_splits: dict[str, list[str]] = {}
    for k, v in splits.items():
        if k == "val":
            continue  # old val skipped; new val inserted after "train" below
        new_splits[k] = v
        if k == "train":
            new_splits["val"] = val
    if "val" not in new_splits:
        new_splits["val"] = val

    with split_file.open("w") as f:
        json.dump(new_splits, f, indent=4)

    print(f"\nWritten to {split_file}")
    print("Keys:", list(new_splits.keys()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domain",
        default="telecom",
        help="tau2 domain name (default: telecom)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Total number of val tasks to sample (default: same total as test, "
            "preserving category proportions)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing split_tasks.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing 'val' split",
    )
    parser.add_argument(
        "--min-faults",
        type=int,
        default=0,
        dest="min_faults",
        help=(
            "Only include tasks with at least this many faults in the sampling "
            "pool (default: 0, no filtering). Use 2 to match test's fault-count "
            "floor and avoid biasing checkpoint selection towards easier tasks."
        ),
    )
    args = parser.parse_args()
    return generate_val_split(
        domain=args.domain,
        seed=args.seed,
        size=args.size,
        dry_run=args.dry_run,
        force=args.force,
        min_faults=args.min_faults,
    )


if __name__ == "__main__":
    sys.exit(main())

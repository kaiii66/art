"""Build rLLM `Dataset` rows for tau2-bench RL training.

The verl backend reads training tasks via `Dataset.get_verl_data_path()`,
which expects a parquet that `DatasetRegistry.register_dataset` writes
under `~/.cache/rllm` (or wherever the registry's `_DATASET_DIR` points).
Each row is a dict; the `extra_info` field is what gets passed to the
workflow's `run(task=..., uid=...)` call (see
`AgentWorkflowEngine.execute_tasks_verl`), so our row shape just needs
`{"task_id": ..., "domain": ...}`.

Tries the existing W&B `training_dataset_artifact` first (so the SFT and
RL pipelines share a fixed task ordering and the leaderboard remains
comparable), and falls back to `tau2.run.get_tasks` if W&B isn't reachable
(e.g. local smoke tests outside the K8s pod).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rllm.data import Dataset, DatasetRegistry  # noqa: E402

logger = logging.getLogger(__name__)


def _load_tasks_from_artifact(
    artifact_name: str,
    domain: str,
    json_filename: str = "training_scenarios.json",
):
    """Try loading task rows from a W&B artifact. Returns list of dicts or None."""
    try:
        import json as _json
        import wandb
    except Exception:
        logger.info("wandb not available; falling back to tau2.run.get_tasks")
        return None

    run = wandb.run
    api_key = os.environ.get("WANDB_API_KEY")
    if run is None and not api_key:
        return None

    try:
        if run is not None:
            art = run.use_artifact(artifact_name)
        else:
            art = wandb.Api().artifact(artifact_name)
        download_dir = art.download()
        path = Path(download_dir) / json_filename
        if not path.exists():
            logger.warning("artifact %s missing %s", artifact_name, json_filename)
            return None
        with path.open() as f:
            rows = _json.load(f)
    except Exception as exc:
        logger.warning("could not load artifact %s: %s", artifact_name, exc)
        return None

    if not rows:
        return []

    out = []
    for r in rows:
        task_id = r.get("task_id") or r.get("id")
        if not task_id:
            continue
        out.append({"task_id": task_id, "domain": r.get("domain", domain)})
    return out


def _load_tasks_from_tau2(domain: str, split: str, num_tasks: Optional[int]) -> list[dict]:
    from tau2.run import get_tasks
    tasks = get_tasks(task_set_name=domain, task_split_name=split, num_tasks=num_tasks)
    return [{"task_id": t.id, "domain": domain} for t in tasks]


def build_dataset_rows(
    domain: str,
    split: str = "train",
    num_tasks: Optional[int] = None,
    artifact_name: Optional[str] = None,
) -> list[dict]:
    """Resolve task rows for the given split.

    Order of attempts:
      1. W&B artifact (`artifact_name`) -- preserves the SFT data ordering.
      2. tau2 default split loader -- offline / smoke fallback.
    """
    if artifact_name:
        rows = _load_tasks_from_artifact(artifact_name, domain)
        if rows is not None:
            if num_tasks is not None:
                rows = rows[:num_tasks]
            logger.info("loaded %d %s tasks from W&B artifact %s", len(rows), split, artifact_name)
            return rows
    rows = _load_tasks_from_tau2(domain, split, num_tasks)
    logger.info("loaded %d %s tasks from tau2.run.get_tasks (domain=%s, split=%s)",
                len(rows), split, domain, split)
    return rows


def register_dataset(
    name: str,
    rows: list[dict],
    split: str = "train",
) -> Dataset:
    """Register a Dataset in rllm's DatasetRegistry and return the handle.

    Idempotent: re-registering with the same (name, split) overwrites the
    parquet cleanly, which is what we want for repeatable training runs.
    """
    if not rows:
        raise ValueError(f"register_dataset({name!r}, split={split!r}): no rows to register")
    return DatasetRegistry.register_dataset(name=name, data=rows, split=split)


def build_and_register(
    domain: str,
    project: str,
    artifact_name: Optional[str] = None,
    val_artifact_name: Optional[str] = None,
    val_split: str = "test",
    num_train_tasks: Optional[int] = None,
    num_val_tasks: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> tuple[Dataset, Dataset | None]:
    """Build + register train / val datasets in one call. Returns (train, val)."""
    dataset_name = dataset_name or f"tau2-{domain}"

    train_rows = build_dataset_rows(
        domain=domain,
        split="train",
        num_tasks=num_train_tasks,
        artifact_name=artifact_name,
    )
    train_ds = register_dataset(dataset_name, train_rows, split="train")

    val_ds = None
    if val_artifact_name or val_split:
        val_rows = build_dataset_rows(
            domain=domain,
            split=val_split,
            num_tasks=num_val_tasks,
            artifact_name=val_artifact_name,
        )
        if val_rows:
            val_ds = register_dataset(dataset_name, val_rows, split=val_split)

    return train_ds, val_ds


__all__ = [
    "build_and_register",
    "build_dataset_rows",
    "register_dataset",
]

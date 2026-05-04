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

Curriculum prefilter (optional)
--------------------------------
`prefilter_tasks()` runs k_probe rollouts of the SFT model (probe_llm) on
each training task, keeps only tasks whose empirical success rate falls in
[low, high] (default [0.10, 0.90]), and caches results so reruns skip the
probe. Tasks with 0% success have no learnable signal; tasks with 100%
success have no variance for GRPO. Focusing on the trainable middle band is
the single highest-leverage change to fix RL regressing below SFT.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

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


def _derive_probe_llm_from_lora_uri(lora_uri: str | None) -> str | None:
    """Convert a wandb-artifact URI to a litellm W&B Inference model name.

    wandb-artifact:///entity/project/name:alias -> wandb/name
    This is the model name that W&B Inference exposes for LoRA serving.
    """
    if not lora_uri:
        return None
    try:
        body = lora_uri.removeprefix("wandb-artifact:///")
        # entity/project/name:alias
        artifact_name = body.split("/")[2].split(":")[0]
        return f"wandb/{artifact_name}"
    except Exception:
        return None


def _probe_task_once(
    task_row: dict,
    env_kwargs: dict,
    probe_llm: str,
    probe_llm_args: dict,
    system_prompt: str,
    tool_parser: Any,
    max_steps: int,
) -> bool:
    """Run a single probe rollout of `probe_llm` on one task. Returns True on success.

    Uses Tau2Env for the environment side and litellm.completion for the
    policy side (no vLLM required -- the probe calls the SFT model via W&B
    Inference). Observation shapes follow the Tau2Env contract:
      {"user_content": str}  -- user turn
      {"tool_outputs": {...}} -- tool call results
    """
    import litellm

    from onprem.scripts.tau2_rl_env import Tau2Env

    env = Tau2Env(**env_kwargs)
    obs, info = env.reset(task=task_row)

    if env.done:
        return float(info.get("success", 0.0)) >= 1.0

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    user_content = obs.get("user_content") if isinstance(obs, dict) else str(obs)
    if user_content:
        messages.append({"role": "user", "content": user_content})

    for _ in range(max_steps):
        try:
            response = litellm.completion(model=probe_llm, messages=messages, **probe_llm_args)
            reply_text: str = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("probe litellm call failed for task %s: %s", task_row.get("task_id"), exc)
            return False

        # Parse Qwen3 <tool_call> markers from the reply text.
        try:
            parsed_calls = tool_parser.parse(reply_text)
        except Exception:
            parsed_calls = []

        if parsed_calls:
            action: Any = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": c.get("name", ""),
                        "arguments": (
                            json.dumps(c["arguments"])
                            if isinstance(c.get("arguments"), dict)
                            else str(c.get("arguments", "{}"))
                        ),
                    },
                }
                for c in parsed_calls
            ]
        else:
            action = {"_assistant_text": reply_text}

        messages.append({"role": "assistant", "content": reply_text})

        obs, _reward, done, info = env.step(action)
        if done:
            return float(info.get("success", 0.0)) >= 1.0

        # Append env feedback: user text or tool results.
        if isinstance(obs, dict):
            if obs.get("user_content"):
                messages.append({"role": "user", "content": obs["user_content"]})
            for tc_id, tc_result in obs.get("tool_outputs", {}).items():
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": tc_result})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

    return False


def prefilter_tasks(
    rows: list[dict],
    *,
    domain: str,
    probe_llm: str | None,
    probe_llm_args: dict | None = None,
    k_probe: int = 4,
    keep_band: tuple[float, float] = (0.10, 0.90),
    env_kwargs: dict | None = None,
    cache_path: str | Path | None = None,
    concurrency: int = 8,
) -> list[dict]:
    """Filter `rows` to tasks where the probe model succeeds [low, high] of the time.

    Args:
        rows:          Task rows from build_dataset_rows().
        domain:        tau2 domain (e.g. "telecom").
        probe_llm:     litellm model string for the SFT LoRA (e.g.
                       "wandb/tau2-sft-Qwen3-30B-..."). Auto-derived from
                       STARTING_LORA_URI if not provided; skips prefilter if
                       still None.
        probe_llm_args: Extra litellm args (temperature, max_tokens, ...).
        k_probe:       Number of rollouts per task.
        keep_band:     (low, high) success-rate band to keep.
        env_kwargs:    Passed directly to Tau2Env (domain, user_llm, ...).
        cache_path:    JSON file to read/write cached probe results. Skips
                       probing for tasks already in the cache. Pass None to
                       disable caching.
        concurrency:   Thread pool size for parallel task probing.

    Returns:
        Filtered subset of `rows`. If the probe fails or probe_llm is
        unavailable, returns the original rows unchanged with a warning.
    """
    if probe_llm is None:
        probe_llm = _derive_probe_llm_from_lora_uri(os.environ.get("STARTING_LORA_URI"))

    if probe_llm is None:
        logger.warning(
            "prefilter_tasks: no probe_llm configured and STARTING_LORA_URI not set; "
            "skipping prefilter (using all %d tasks)",
            len(rows),
        )
        return rows

    probe_llm_args = {
        "temperature": 0.7,
        "max_tokens": 4096,
        **(probe_llm_args or {}),
    }
    low, high = keep_band

    # Load probe cache if available.
    cache: dict[str, float] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            logger.info("prefilter: loaded %d cached probe results from %s", len(cache), cache_file)
        except Exception as exc:
            logger.warning("prefilter: could not read cache %s: %s", cache_file, exc)

    # Build agent setup (system prompt + parser) once -- cheap, no LLM call.
    try:
        from rllm.parser import get_tool_parser
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
        from tau2.registry import registry as tau2_registry

        env_constructor = tau2_registry.get_env_constructor(domain)
        _tmp_env = env_constructor()
        domain_policy: str = _tmp_env.get_policy()
        tools_openai: list[dict] = [t.openai_schema for t in _tmp_env.get_tools()]
        del _tmp_env

        parser_cls = get_tool_parser("qwen")
        tool_parser = parser_cls()
        tools_schema_json = json.dumps(tools_openai, indent=2)
        tools_prompt = tool_parser.get_tool_prompt(tools_schema_json)
        system_prompt = (
            SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)
            + "\n\n"
            + tools_prompt
        )
    except Exception as exc:
        logger.warning("prefilter: agent setup failed (%s); skipping prefilter", exc)
        return rows

    base_env_kwargs = {
        "domain": domain,
        "max_steps": 50,
        **(env_kwargs or {}),
    }

    # Probe tasks not yet in cache.
    to_probe = [r for r in rows if r["task_id"] not in cache]
    if to_probe:
        logger.info(
            "prefilter: probing %d tasks with %s (k=%d, band=[%.2f,%.2f], concurrency=%d)",
            len(to_probe), probe_llm, k_probe, low, high, concurrency,
        )

        def _probe_task(row: dict) -> tuple[str, float]:
            task_id = row["task_id"]
            successes = sum(
                _probe_task_once(
                    row, base_env_kwargs, probe_llm, probe_llm_args,
                    system_prompt, tool_parser, base_env_kwargs["max_steps"],
                )
                for _ in range(k_probe)
            )
            rate = successes / k_probe
            logger.debug("prefilter: task %s success_rate=%.2f (%d/%d)", task_id, rate, successes, k_probe)
            return task_id, rate

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_probe_task, r): r for r in to_probe}
                for fut in as_completed(futures):
                    try:
                        task_id, rate = fut.result()
                        cache[task_id] = rate
                    except Exception as exc:
                        row = futures[fut]
                        logger.warning("prefilter: probe failed for task %s: %s", row.get("task_id"), exc)
        except Exception as exc:
            logger.warning("prefilter: probing failed (%s); skipping prefilter", exc)
            return rows

        # Persist updated cache.
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(cache, indent=2))
                logger.info("prefilter: saved probe cache to %s", cache_file)
            except Exception as exc:
                logger.warning("prefilter: could not write cache %s: %s", cache_file, exc)

    # Filter.
    filtered = [r for r in rows if low <= cache.get(r["task_id"], 0.5) <= high]
    logger.info(
        "prefilter: kept %d/%d tasks in band [%.2f, %.2f]; dropped %d "
        "(too easy: %d, too hard: %d)",
        len(filtered), len(rows), low, high,
        len(rows) - len(filtered),
        sum(1 for r in rows if cache.get(r["task_id"], 0.5) > high),
        sum(1 for r in rows if cache.get(r["task_id"], 0.5) < low),
    )

    if not filtered:
        logger.warning(
            "prefilter: no tasks survived band [%.2f, %.2f]; "
            "returning all %d tasks unfiltered to avoid empty training set",
            low, high, len(rows),
        )
        return rows

    return filtered


__all__ = [
    "build_and_register",
    "build_dataset_rows",
    "prefilter_tasks",
    "register_dataset",
]

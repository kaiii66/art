#!/usr/bin/env python3
"""
compare_base_vs_sft.py — Quick base-vs-SFT comparison for tau2-bench.

What this is for:
    "Did SFT actually move the needle?" Run this BEFORE committing to a
    multi-hour RL run. It evaluates the base model and the SFT-trained
    LoRA on the held-out tau2-bench validation set and prints a
    side-by-side score table.

What it does NOT do:
    - No Weave Leaderboard publish (this is the whole point of this
      script vs `create_leaderboard_shaped_reward.py`).
    - No W&B run / no sentinel artifact -- nothing pollutes your
      project's leaderboard or "first base eval" gating.
    - Weave traces from individual rollouts may still appear under your
      project's Traces tab (that's just how `@weave.op` works); they're
      not surfaced as leaderboard rows and can be ignored.

Typical usage (matches the user's "medium" choice -- 1 trial/task,
both models in parallel):

    uv run python compare_base_vs_sft.py \\
        --config train_config.yaml \\
        --sft-uri-file pipeline_runs/04301749/.sft_lora_artifact_uri \\
        --num-trials 1 --concurrency 10

Or pass the URI directly:

    uv run python compare_base_vs_sft.py --sft-uri \\
        wandb-artifact:///kwt/tau2-ART-autoresearch-telecom-0430/tau2-sft-Qwen3-30B-A3B-Instruct-2507-05022004:v0
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

import weave
import art
from art.serverless.backend import ServerlessBackend

from tau2_art_helpers import Tau2BaseModelWrapper


def _parse_wandb_artifact_uri(uri: str) -> tuple[str, str, str, str]:
    """wandb-artifact:///<entity>/<project>/<name>:<alias> -> (e, p, n, a)."""
    body = uri.strip().removeprefix("wandb-artifact:///")
    path, _, alias = body.rpartition(":")
    if not alias:
        raise ValueError(
            f"URI {uri!r} has no :alias suffix; expected wandb-artifact:///e/p/n:vX"
        )
    parts = path.split("/", 2)
    if len(parts) != 3:
        raise ValueError(
            f"URI {uri!r} should have exactly entity/project/name (got {parts!r})"
        )
    entity, project, name = parts
    return entity, project, name, alias


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open() as f:
        return yaml.safe_load(f)


def _resolve_sft_uri(args: argparse.Namespace) -> str:
    if args.sft_uri:
        return args.sft_uri.strip()
    if args.sft_uri_file:
        p = Path(args.sft_uri_file)
        if not p.exists():
            raise FileNotFoundError(f"SFT URI file not found: {p}")
        return p.read_text().strip()
    raise SystemExit("Must pass --sft-uri or --sft-uri-file")


async def _run_one(
    *,
    sem: asyncio.Semaphore,
    label: str,
    wrapper: Tau2BaseModelWrapper,
    task_id: str,
    domain: str,
    trial: int,
    total: int,
    counter: list[int],
    t0: float,
) -> dict:
    """Run one (model, task, trial) rollout. Returns a result row."""
    async with sem:
        try:
            out = await wrapper.predict(task_id=task_id, domain=domain)
            success = float(out.get("success", 0.0))
            reward = float(out.get("reward", 0.0))
            err = None
        except Exception as e:
            success = 0.0
            reward = 0.0
            err = f"{type(e).__name__}: {e}"
        counter[0] += 1
        elapsed = time.time() - t0
        rate = counter[0] / max(elapsed, 1e-6)
        eta_s = (total - counter[0]) / max(rate, 1e-6)
        print(
            f"  [{counter[0]:3d}/{total}] {label:>4s} "
            f"task={task_id} trial={trial} success={success:.0f} "
            f"reward={reward:.3f} elapsed={elapsed:.0f}s eta={eta_s:.0f}s"
            + (f"  ERR={err}" if err else ""),
            flush=True,
        )
        return {
            "label": label,
            "task_id": task_id,
            "trial": trial,
            "success": success,
            "reward": reward,
            "error": err,
        }


def _summarize(rows: list[dict], label: str) -> dict:
    own = [r for r in rows if r["label"] == label]
    n_total = len(own)
    n_err = sum(1 for r in own if r["error"])
    successes = [r["success"] for r in own if not r["error"]]
    rewards = [r["reward"] for r in own if not r["error"]]
    return {
        "label": label,
        "n_total": n_total,
        "n_err": n_err,
        "n_ok": len(successes),
        "success_mean": statistics.mean(successes) if successes else 0.0,
        "reward_mean": statistics.mean(rewards) if rewards else 0.0,
        "success_std": statistics.stdev(successes) if len(successes) > 1 else 0.0,
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
    }


def _print_table(base: dict, sft: dict) -> None:
    print()
    print("=" * 78)
    print("RESULTS  (base = vanilla model, sft = LoRA-trained model)")
    print("=" * 78)
    print(
        f"{'metric':<22s}  {'base':>14s}  {'sft':>14s}  {'delta':>14s}"
    )
    print("-" * 78)
    print(
        f"{'success.mean':<22s}  {base['success_mean']:>14.4f}  {sft['success_mean']:>14.4f}  "
        f"{sft['success_mean'] - base['success_mean']:>+14.4f}"
    )
    print(
        f"{'task_reward.mean':<22s}  {base['reward_mean']:>14.4f}  {sft['reward_mean']:>14.4f}  "
        f"{sft['reward_mean'] - base['reward_mean']:>+14.4f}"
    )
    print(
        f"{'rollouts ok / total':<22s}  "
        f"{base['n_ok']:>5d} / {base['n_total']:<6d}  "
        f"{sft['n_ok']:>5d} / {sft['n_total']:<6d}"
    )
    if base["n_err"] or sft["n_err"]:
        print(
            f"{'rollouts errored':<22s}  {base['n_err']:>14d}  {sft['n_err']:>14d}"
        )
    print("=" * 78)
    delta = sft["success_mean"] - base["success_mean"]
    if delta > 0.02:
        print("VERDICT: SFT improved over base (success delta > +2pp). Proceed to RL.")
    elif delta < -0.02:
        print(
            "VERDICT: SFT REGRESSED vs base (success delta < -2pp). Investigate "
            "before kicking off RL: data quality, learning rate, # epochs, etc."
        )
    else:
        print(
            "VERDICT: SFT roughly matches base (|success delta| <= 2pp). With "
            "num_trials=1 the noise floor is wide; consider re-running with "
            "--num-trials 3 before deciding."
        )
    print("=" * 78)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick base-vs-SFT comparison on tau2-bench (no leaderboard publish)."
    )
    parser.add_argument("--config", default="train_config.yaml",
                        help="Path to train_config.yaml (drives base_model, agent_llm, user_llm, etc.)")
    parser.add_argument("--sft-uri", default=None,
                        help="wandb-artifact:/// URI for the SFT LoRA. Mutually exclusive with --sft-uri-file.")
    parser.add_argument("--sft-uri-file", default=None,
                        help="Path to a file containing the SFT URI on a single line "
                             "(e.g. pipeline_runs/<snapshot>/.sft_lora_artifact_uri).")
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Rollouts per (model, task). Default 1 (quick). Bump to 3 to match leaderboard variance.")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max in-flight rollouts (per process). Default 10.")
    parser.add_argument("--limit-tasks", type=int, default=None,
                        help="Cap the number of validation tasks (debug / smoke test).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override leaderboard.max_steps (default: read from config).")
    parser.add_argument("--inference-base-url", default=None,
                        help="Override the inference endpoint for the trained model "
                             "(default: https://api.inference.wandb.ai/v1).")
    args = parser.parse_args()

    config = _load_config(args.config)
    project = config["project"]
    domain = config["domain"]
    base_model = config["base_model"]
    base_short = _short(base_model)
    agent_llm = config.get("agent_llm", f"wandb/{base_model}")
    user_llm = config["user_llm"]
    eval_weave = config.get("validation_weave_dataset") or f"tau2-{domain}-validation-scenarios"

    lb_config = config.get("leaderboard", {})
    max_steps = args.max_steps or lb_config.get("max_steps", config.get("max_orchestrator_steps", 30))
    user_llm_args = lb_config.get("user_llm_args", config.get("user_llm_args", {"temperature": 1.0}))
    agent_llm_args = lb_config.get("agent_llm_args", {})

    shaped_kwargs = dict(
        use_shaped_reward=config.get("shaped_reward", False),
        shaped_reward_weights=config.get("shaped_reward_weights", {}),
    )

    sft_uri = _resolve_sft_uri(args)
    sft_entity, sft_project, sft_name, sft_alias = _parse_wandb_artifact_uri(sft_uri)
    print("\n=== compare_base_vs_sft ===")
    print(f"  config            : {args.config}")
    print(f"  project           : {project}  (config)")
    print(f"  base_model        : {base_model}")
    print(f"  agent_llm (base)  : {agent_llm}")
    print(f"  user_llm          : {user_llm}")
    print(f"  domain            : {domain}")
    print(f"  validation set    : {eval_weave}")
    print(f"  sft URI           : {sft_uri}")
    print(f"      entity={sft_entity}  project={sft_project}  name={sft_name}  alias={sft_alias}")
    print(f"  num_trials        : {args.num_trials}")
    print(f"  max_steps         : {max_steps}")
    print(f"  concurrency       : {args.concurrency}")
    if sft_project != project:
        print(
            f"  WARNING: SFT artifact lives in W&B project {sft_project!r} but "
            f"config.project is {project!r}. Eval will still work, but cross-check "
            f"this is intentional."
        )

    weave.init(project)

    print(f"\nLoading validation dataset: {eval_weave}")
    try:
        ds = weave.ref(eval_weave).get()
    except Exception as e:
        raise RuntimeError(f"Could not load Weave dataset {eval_weave}: {e}") from e
    rows = list(ds.rows)
    print(f"  loaded {len(rows)} validation rows")
    if args.limit_tasks is not None:
        rows = rows[: args.limit_tasks]
        print(f"  limited to first {len(rows)} tasks (--limit-tasks)")
    task_ids = [r["task_id"] for r in rows]

    print("\nBuilding base wrapper (no LoRA)...")
    base_wrapper = Tau2BaseModelWrapper(
        name=f"base-{base_short}",
        model=None,
        model_name=base_model,
        domain=domain,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        agent_llm_args=agent_llm_args,
        max_steps=max_steps,
        agent_llm=agent_llm,
        **shaped_kwargs,
    )

    print(f"Building SFT wrapper (pinned to alias :{sft_alias})...")
    backend = ServerlessBackend()
    trained_model = art.TrainableModel(
        name=sft_name,
        project=sft_project,
        base_model=base_model,
    )
    await trained_model.register(backend)
    inf_url = args.inference_base_url or lb_config.get(
        "inference_base_url", "https://api.inference.wandb.ai/v1"
    )
    if inf_url:
        prev = trained_model.inference_base_url
        trained_model.inference_base_url = inf_url
        if prev != inf_url:
            print(f"  inference_base_url override: {prev} -> {inf_url}")
    # Force entity to match the URI we were given. .register() may have
    # populated trained_model.entity from wandb's default_entity, which
    # might differ from the entity that uploaded the SFT artifact (e.g.
    # kwt). The pinned-alias path in tau2_rollout reads model.entity to
    # build wandb-artifact:///<entity>/<project>/<name>:<alias>, so this
    # MUST match where the artifact actually lives.
    if getattr(trained_model, "entity", None) != sft_entity:
        prev_e = getattr(trained_model, "entity", None)
        trained_model.entity = sft_entity
        print(f"  entity override: {prev_e} -> {sft_entity}")

    sft_wrapper = Tau2BaseModelWrapper(
        name=f"sft-alias-{sft_alias}",
        model=trained_model,
        model_name=f"sft-alias-{sft_alias}",
        domain=domain,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        agent_llm_args=agent_llm_args,
        max_steps=max_steps,
        pinned_alias=sft_alias,
        **shaped_kwargs,
    )

    total = len(task_ids) * args.num_trials * 2
    print(f"\nDispatching {total} rollouts ({len(task_ids)} tasks * {args.num_trials} trials * 2 models) "
          f"with concurrency={args.concurrency}...")

    sem = asyncio.Semaphore(args.concurrency)
    counter = [0]
    t0 = time.time()
    coros = []
    for trial in range(args.num_trials):
        for tid in task_ids:
            for label, wrapper in (("base", base_wrapper), ("sft", sft_wrapper)):
                coros.append(_run_one(
                    sem=sem, label=label, wrapper=wrapper,
                    task_id=tid, domain=domain, trial=trial,
                    total=total, counter=counter, t0=t0,
                ))
    rows_out = await asyncio.gather(*coros)

    elapsed = time.time() - t0
    print(f"\nAll rollouts done in {elapsed:.0f}s.")

    base_summary = _summarize(rows_out, "base")
    sft_summary = _summarize(rows_out, "sft")
    _print_table(base_summary, sft_summary)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

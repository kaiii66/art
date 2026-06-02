"""End-to-end local-RL pipeline orchestrator for tau2-bench.

Stages (in order):
  0. setup       Auto-generate the val split (idempotent — skips in <1s if
                 val already exists in split_tasks.json; no --force so the
                 committed split is always authoritative).
  1. pull_sft    Download the SFT LoRA from W&B into .art/ and write
                 .sft_endpoint_step + .last_trained_model next to the
                 snapshot config.
  2. rl          Local GRPO training via LocalBackend (with KL anchor).
                 Writes .best_rl_step on val improvement.
  3. upload_rl   Upload the best RL checkpoint back to the same W&B
                 collection so create_leaderboard_shaped_reward.py works
                 unchanged (resolves via W&B Inference).
  4. leaderboard Run create_leaderboard_shaped_reward.py --models all
                 (auto-discovers .sft_endpoint_step + .best_rl_step from
                 the snapshot dir).

Usage:
    uv run python local/run_pipeline_local.py
    uv run python local/run_pipeline_local.py --skip upload_rl leaderboard
    uv run python local/run_pipeline_local.py --dry-run
    uv run python local/run_pipeline_local.py --project-suffix 05121500
    uv run python local/run_pipeline_local.py --resume pipeline_runs/05121000
    # Re-run leaderboard alone:
    uv run python local/run_pipeline_local.py \\
        --resume pipeline_runs/05121000 --skip pull_sft rl upload_rl
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent

SRC_LOCAL_CONFIG = REPO_ROOT / "train_config_local.yaml"
RUNS_DIR = REPO_ROOT / "pipeline_runs"

ALL_STAGES = ["pull_sft", "rl", "upload_rl", "leaderboard"]

# Launch each stage subprocess with whatever `python` is on $PATH.
#
# In the on-prem image, the venv (`/workspace/.venv`) is on $PATH so `python`
# resolves to the venv interpreter directly.  We deliberately avoid `uv run`
# because uv re-installs the local project on each invocation, and that step
# resets dependency versions to match uv.lock — clobbering the
# huggingface-hub<1.0 / gql>=4 pins baked into the image and breaking
# transformers/weave imports.
#
# For local dev, run via `uv run python local/run_pipeline_local.py …` so the
# parent process sets up the venv; child processes inherit PATH+VIRTUAL_ENV.
UV_RUN: list[str] = []

try:
    from ruamel.yaml import YAML  # type: ignore[import-not-found]

    _RUAMEL = YAML()
    _RUAMEL.preserve_quotes = True
    _RUAMEL.indent(mapping=2, sequence=4, offset=2)
    _HAS_RUAMEL = True
except ImportError:
    import yaml as _pyyaml

    _HAS_RUAMEL = False


def yaml_load(path: Path):
    if _HAS_RUAMEL:
        with path.open() as f:
            return _RUAMEL.load(f)
    with path.open() as f:
        return _pyyaml.safe_load(f)


def yaml_dump(data, path: Path) -> None:
    if _HAS_RUAMEL:
        with path.open("w") as f:
            _RUAMEL.dump(data, f)
        return
    with path.open("w") as f:
        _pyyaml.safe_dump(data, f, sort_keys=False)


def make_snapshot(suffix: str, resume_dir: Path | None) -> tuple[Path, str]:
    """Create (or reuse) a pipeline snapshot dir.

    Returns (snapshot_dir, project_name).
    """
    if resume_dir is not None:
        snapshot = resume_dir.resolve()
        if not snapshot.exists():
            raise FileNotFoundError(f"--resume dir does not exist: {snapshot}")
        cfg = yaml_load(snapshot / "train_config_local.yaml")
        project = cfg["project"]
        print(f"[snapshot] resuming: {snapshot}")
        print(f"[snapshot] project  : {project}")
        return snapshot, project

    snapshot = RUNS_DIR / suffix
    snapshot.mkdir(parents=True, exist_ok=True)

    if not SRC_LOCAL_CONFIG.exists():
        raise FileNotFoundError(
            f"Source config not found: {SRC_LOCAL_CONFIG}. "
            "Expected art/train_config_local.yaml."
        )

    src_cfg = yaml_load(SRC_LOCAL_CONFIG)
    project = src_cfg["project"]

    # Give this run an isolated model_name so its .art/ checkpoint directory
    # and W&B RL artifact collection never collide with other runs.
    # The base name from train_config_local.yaml is preserved as a prefix;
    # the snapshot suffix (MMDDHHMM) makes each run unique.
    # sft_source.name is intentionally left pointing to the original SFT
    # collection — pull_sft_lora.py always seeds from the same SFT baseline.
    base_model_name = src_cfg["model_name"]
    run_model_name = f"{base_model_name}-rl-{suffix}"
    src_cfg["model_name"] = run_model_name

    yaml_dump(src_cfg, snapshot / "train_config_local.yaml")

    manifest = {
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "suffix": suffix,
        "project": project,
        "base_model_name": base_model_name,
        "model_name": run_model_name,
        "pipeline": "local-rl",
        "source_local_config": str(SRC_LOCAL_CONFIG),
        "ruamel_used": _HAS_RUAMEL,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[snapshot] dir        : {snapshot}")
    print(f"[snapshot] project    : {project}")
    print(f"[snapshot] model_name : {run_model_name}  (isolated per run)")
    if not _HAS_RUAMEL:
        print(
            "[snapshot] WARNING: ruamel.yaml not installed; install with "
            "`uv add ruamel.yaml` to preserve YAML comments in snapshots."
        )
    return snapshot, project


def run_stage(name: str, cmd: list[str], log_path: Path, *, dry_run: bool) -> None:
    print()
    print(f"=== stage: {name} ===")
    print(f"    cmd : {' '.join(cmd)}")
    print(f"    log : {log_path}")
    if dry_run:
        print("    (dry-run; not executing)")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(
            f"# stage: {name}\n# cmd: {' '.join(cmd)}\n"
            f"# started: {datetime.now().isoformat()}\n\n"
        )
        log_file.flush()
        # start_new_session: put the child in its own process group/session
        # so the rl stage can `killpg(getpgrp(), SIGKILL)` on its own group
        # at the end to take down vLLM EngineCore + multiprocessing helpers
        # without affecting this orchestrator.
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            start_new_session=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
        log_file.write(
            f"\n# exit_code: {rc}\n# ended: {datetime.now().isoformat()}\n"
        )

    if rc != 0:
        # The rl stage uses killpg(SIGKILL) to take down orphan vLLM children
        # at the end (the only way to unblock the parent's pipe-read).  This
        # makes it exit with rc=-9 even on success, so we look for a sentinel.
        sentinel = log_path.parent / f".{name.replace('-', '_')}_complete"
        if sentinel.exists():
            print(f"  [run_stage] ignoring rc={rc}; sentinel {sentinel.name} present")
        else:
            raise RuntimeError(
                f"Stage {name!r} failed with exit code {rc}; see {log_path}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=ALL_STAGES,
        help=f"Stages to skip (any of: {', '.join(ALL_STAGES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing subprocesses",
    )
    parser.add_argument(
        "--project-suffix",
        default=None,
        help="Override the MMDDHHMM suffix (default: now in America/Los_Angeles)",
    )
    parser.add_argument(
        "--no-publish-leaderboard",
        action="store_true",
        help="Don't pass --publish-leaderboard to the leaderboard stage",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Reuse an existing pipeline_runs/<dir> instead of creating a new snapshot",
    )
    parser.add_argument(
        "--upload-all-steps",
        action="store_true",
        help="Pass --upload-all to upload_rl_lora.py (uploads every checkpoint, not just best)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Pass --num-tasks to RL stage (useful for smoke testing)",
    )
    args = parser.parse_args()

    suffix = args.project_suffix or datetime.now(
        ZoneInfo("America/Los_Angeles")
    ).strftime("%m%d%H%M")
    snapshot, project = make_snapshot(suffix, args.resume)
    local_cfg = snapshot / "train_config_local.yaml"

    skip = set(args.skip)

    # Leaderboard models — only RL is available here (no serverless SFT stage)
    # but we still pass "all" which auto-discovers .sft_endpoint_step (the
    # seeded SFT checkpoint) and .best_rl_step.
    if "rl" in skip and "pull_sft" in skip:
        lb_models = ["base"]
    elif "rl" in skip:
        lb_models = ["base", "sft"]
    else:
        lb_models = ["all"]

    leaderboard_cmd = [
        *UV_RUN, "python", "create_leaderboard_shaped_reward.py",
        "--config", str(local_cfg),
        "--models", *lb_models,
    ]
    if not args.no_publish_leaderboard:
        leaderboard_cmd.append("--publish-leaderboard")

    upload_rl_cmd = [
        *UV_RUN, "python", "local/upload_rl_lora.py",
        "--config", str(local_cfg),
        "--snapshot-dir", str(snapshot),
    ]
    if args.upload_all_steps:
        upload_rl_cmd.append("--upload-all")

    rl_cmd = [
        *UV_RUN, "python", "local/train_tau2_local.py",
        "--config", str(local_cfg),
    ]
    if args.num_tasks is not None:
        rl_cmd += ["--num-tasks", str(args.num_tasks)]

    stage_cmds: dict[str, list[str]] = {
        "pull_sft": [
            *UV_RUN, "python", "local/pull_sft_lora.py",
            "--config", str(local_cfg),
            "--snapshot-dir", str(snapshot),
        ],
        "rl": rl_cmd,
        "upload_rl": upload_rl_cmd,
        "leaderboard": leaderboard_cmd,
    }

    run_cfg = yaml_load(local_cfg)
    run_model_name = run_cfg.get("model_name", "?")
    _domain = run_cfg.get("domain", "telecom")

    # Step 0 — auto-generate val split (idempotent: skips in <1s if val exists).
    # No --force: the committed split is authoritative; only regenerate manually.
    gen_cmd = [*UV_RUN, "python", "scripts/generate_val_split.py", "--domain", _domain]
    run_stage(
        "setup: generate val split (idempotent)",
        gen_cmd,
        snapshot / "00-generate-val-split.log",
        dry_run=args.dry_run,
    )

    print()
    print("=== local RL pipeline plan ===")
    print(f"    snapshot         : {snapshot}")
    print(f"    project          : {project}")
    print(f"    model_name       : {run_model_name}")
    print(f"    local_config     : {local_cfg}")
    print(f"    domain           : {_domain}")
    print(f"    stages enabled   : {[s for s in ALL_STAGES if s not in skip]}")
    print(f"    stages skipped   : {sorted(skip)}")
    print(f"    publish_lb       : {not args.no_publish_leaderboard}")

    for idx, stage in enumerate(ALL_STAGES, start=1):
        if stage in skip:
            print(f"\n=== stage: {stage} (skipped) ===")
            continue
        log_path = snapshot / f"{idx:02d}-{stage}.log"
        run_stage(stage, stage_cmds[stage], log_path, dry_run=args.dry_run)

    print()
    print("=== done ===")
    print(f"    snapshot : {snapshot}")
    print(f"    project  : {project}")
    if (snapshot / ".last_trained_model").exists():
        print(f"    model    : {(snapshot / '.last_trained_model').read_text().strip()}")
    if (snapshot / ".sft_endpoint_step").exists():
        print(f"    sft step : {(snapshot / '.sft_endpoint_step').read_text().strip()}")
    if (snapshot / ".best_rl_step").exists():
        print(f"    best rl  : step {(snapshot / '.best_rl_step').read_text().strip()}")
    print(
        f"\n    re-run leaderboard alone:\n"
        f"      uv run python local/run_pipeline_local.py "
        f"--resume {snapshot} --skip pull_sft rl upload_rl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

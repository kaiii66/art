"""End-to-end pipeline orchestrator for the tau2-bench ART workflow.

Runs in order:
  1. snapshot      copy train_config.yaml + train_distill_config.yaml into
                   pipeline_runs/<MMDDHHMM>/ and rewrite the `project` field
                   in both copies. Source-of-truth YAMLs in the repo root are
                   never mutated.
  2. upload        upload_dataset_to_wandb.py against the snapshot config
  3. sft           train_tau2_distill.py against the snapshot config
                   -> writes pipeline_runs/<tag>/.last_trained_model
  4. eval_sft      create_leaderboard_shaped_reward.py --models all
                   --publish-leaderboard (publishes the leaderboard once for
                   the new project, evaluates baseline + SFT)
  5. patch         read pipeline_runs/<tag>/.last_trained_model and write it
                   to the snapshot train_config.yaml's `continue_from_model`
                   field. This is what wires SFT -> RL: train_tau2.py reads
                   continue_from_model, instantiates the same ART collection,
                   and `model.get_step()` returns the SFT's last step so RL
                   appends step N+1, N+2, ... in the same collection.
  6. rl            train_tau2.py against the snapshot config
  7. eval_rl       create_leaderboard_shaped_reward.py --models trained
                   (the leaderboard is already published; this just adds the
                   RL row via Weave's accumulation)

Usage:
  uv run python run_pipeline.py
  uv run python run_pipeline.py --skip upload sft
  uv run python run_pipeline.py --dry-run
  uv run python run_pipeline.py --project-suffix 04202030
  uv run python run_pipeline.py --no-publish-leaderboard
  uv run python run_pipeline.py --resume pipeline_runs/04201530
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent
SRC_TRAIN_CONFIG = REPO_ROOT / "train_config.yaml"
SRC_DISTILL_CONFIG = REPO_ROOT / "train_distill_config.yaml"
RUNS_DIR = REPO_ROOT / "pipeline_runs"

ALL_STAGES = ["upload", "sft", "eval_sft", "rl", "eval_rl"]

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


def derive_project_name(current: str, suffix: str) -> str:
    """Strip a trailing -NNNN(NNNN) date stamp, append -<suffix>."""
    stripped = re.sub(r"-\d{4,8}$", "", current)
    return f"{stripped}-{suffix}"


def make_snapshot(suffix: str, resume_dir: Path | None) -> tuple[Path, str]:
    """Create (or reuse) snapshot dir; return (snapshot_dir, project_name)."""
    if resume_dir is not None:
        snapshot = resume_dir.resolve()
        if not snapshot.exists():
            raise FileNotFoundError(f"--resume dir does not exist: {snapshot}")
        train_cfg = yaml_load(snapshot / "train_config.yaml")
        project = train_cfg["project"]
        print(f"[snapshot] resuming existing snapshot: {snapshot}")
        print(f"[snapshot] project (from snapshot)   : {project}")
        return snapshot, project

    snapshot = RUNS_DIR / suffix
    snapshot.mkdir(parents=True, exist_ok=True)

    src_train = yaml_load(SRC_TRAIN_CONFIG)
    src_distill = yaml_load(SRC_DISTILL_CONFIG)

    new_project = derive_project_name(src_train["project"], suffix)

    src_train["project"] = new_project
    src_distill["project"] = new_project

    yaml_dump(src_train, snapshot / "train_config.yaml")
    yaml_dump(src_distill, snapshot / "train_distill_config.yaml")

    manifest = {
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "suffix": suffix,
        "project": new_project,
        "source_train_config": str(SRC_TRAIN_CONFIG),
        "source_distill_config": str(SRC_DISTILL_CONFIG),
        "ruamel_used": _HAS_RUAMEL,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[snapshot] dir     : {snapshot}")
    print(f"[snapshot] project : {new_project}")
    if not _HAS_RUAMEL:
        print(
            "[snapshot] WARNING: ruamel.yaml not installed; comments are stripped "
            "from snapshot copies (originals at repo root are untouched). "
            "`uv add ruamel.yaml` to preserve them."
        )
    return snapshot, new_project


def run_stage(
    name: str,
    cmd: list[str],
    log_path: Path,
    *,
    dry_run: bool,
) -> None:
    print()
    print(f"=== stage: {name} ===")
    print(f"    cmd : {' '.join(cmd)}")
    print(f"    log : {log_path}")
    if dry_run:
        print("    (dry-run; not executing)")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"# stage: {name}\n# cmd: {' '.join(cmd)}\n# started: {datetime.now().isoformat()}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
        log_file.write(f"\n# exit_code: {rc}\n# ended: {datetime.now().isoformat()}\n")

    if rc != 0:
        raise RuntimeError(f"stage {name!r} failed with exit code {rc}; see {log_path}")


def patch_continue_from_model(snapshot: Path) -> str:
    last_model_file = snapshot / ".last_trained_model"
    if not last_model_file.exists():
        raise FileNotFoundError(
            f"{last_model_file} not found — SFT stage did not write its model "
            f"name. Cannot wire continue_from_model for RL stage."
        )
    sft_name = last_model_file.read_text().strip()
    if not sft_name:
        raise ValueError(f"{last_model_file} is empty")

    train_cfg_path = snapshot / "train_config.yaml"
    cfg = yaml_load(train_cfg_path)
    cfg["continue_from_model"] = sft_name
    yaml_dump(cfg, train_cfg_path)

    print(f"[patch] continue_from_model -> {sft_name}")
    print(f"[patch] wrote {train_cfg_path}")
    return sft_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        help="Don't pass --publish-leaderboard to the eval_sft stage",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Reuse an existing pipeline_runs/<dir> instead of creating a new snapshot",
    )
    args = parser.parse_args()

    suffix = args.project_suffix or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")
    snapshot, project = make_snapshot(suffix, args.resume)
    train_cfg = snapshot / "train_config.yaml"
    distill_cfg = snapshot / "train_distill_config.yaml"

    skip = set(args.skip)
    print()
    print("=== pipeline plan ===")
    print(f"    snapshot         : {snapshot}")
    print(f"    project          : {project}")
    print(f"    train_config     : {train_cfg}")
    print(f"    distill_config   : {distill_cfg}")
    print(f"    stages enabled   : {[s for s in ALL_STAGES if s not in skip]}")
    print(f"    stages skipped   : {sorted(skip)}")
    print(f"    publish_lb (eval_sft): {not args.no_publish_leaderboard}")

    eval_sft_cmd = [
        "uv", "run", "python", "create_leaderboard_shaped_reward.py",
        "--config", str(train_cfg),
        "--models", "all",
    ]
    if not args.no_publish_leaderboard:
        eval_sft_cmd.append("--publish-leaderboard")

    stage_cmds: dict[str, list[str]] = {
        "upload": ["uv", "run", "python", "upload_dataset_to_wandb.py", "--config", str(train_cfg)],
        "sft": ["uv", "run", "python", "train_tau2_distill.py", "--config", str(distill_cfg)],
        "eval_sft": eval_sft_cmd,
        "rl": ["uv", "run", "python", "train_tau2.py", "--config", str(train_cfg)],
        "eval_rl": [
            "uv", "run", "python", "create_leaderboard_shaped_reward.py",
            "--config", str(train_cfg),
            "--models", "trained",
        ],
    }

    for idx, stage in enumerate(ALL_STAGES, start=2):
        if stage == "rl" and "rl" not in skip:
            print()
            print("=== stage: patch (between eval_sft and rl) ===")
            if args.dry_run:
                print("    (dry-run; would read .last_trained_model and patch continue_from_model)")
            else:
                patch_continue_from_model(snapshot)

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
        print(f"    sft model: {(snapshot / '.last_trained_model').read_text().strip()}")
    print(f"    re-run eval_rl alone: uv run python run_pipeline.py --resume {snapshot} --skip upload sft eval_sft rl")
    return 0


if __name__ == "__main__":
    sys.exit(main())

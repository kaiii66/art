"""
Upload tau2 training, validation, and base scenarios to W&B artifacts and Weave.

By default loads full train/test/base splits from the domain data (tasks.json +
split_tasks.json). Use --num-base-tasks N to limit base to first N tasks (same
order as tau2 run --num-tasks N). Domain (and project) come from config.

Idempotency:
  This stage is the FIRST one in run_pipeline.py, so it gets called once per
  autoresearch iteration. We do NOT want to re-upload the dataset every time
  (it costs network + creates a noisy `:vN` ladder of artifact versions and
  duplicates Weave datasets across iterations). The default behavior is:
    1. Probe W&B for the three dataset artifacts and Weave for the three
       Weave datasets configured in the YAML.
    2. If ALL six already exist, log a single "skip" line and exit before
       creating a wandb.run. Subsequent stages (sft, rl, leaderboard) will
       still resolve them via `wandb.run.use_artifact(...)` /
       `weave.ref(...).get()`.
    3. If ANY are missing (or --force is passed), do the full upload.

Usage:
    python upload_dataset_to_wandb.py
    python upload_dataset_to_wandb.py --config train_config.yaml
    python upload_dataset_to_wandb.py --domain telecom
    python upload_dataset_to_wandb.py --domain telecom --num-base-tasks 3
    python upload_dataset_to_wandb.py --domain telecom --info-only
    python upload_dataset_to_wandb.py --force          # re-upload even if exists
    python upload_dataset_to_wandb.py --no-skip-if-exists
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import wandb
import weave

from tau2.run import get_tasks, load_task_splits

load_dotenv()


def load_config(config_path: str = "train_config.yaml") -> dict:
    import yaml
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_file) as f:
        return yaml.safe_load(f)


def _resolve_artifact_names(config: dict, domain: str) -> dict:
    """Return the ARTIFACT collection names (no `:version`) keyed by split."""
    return {
        "train": config.get(
            "training_dataset_artifact", f"tau2-{domain}-training-scenarios"
        ).split(":")[0],
        "validation": config.get(
            "validation_dataset_artifact", f"tau2-{domain}-validation-scenarios"
        ).split(":")[0],
        "base": config.get("base_weave_dataset", f"tau2-{domain}-base-scenarios"),
    }


def _resolve_weave_names(config: dict, domain: str) -> dict:
    """Return the WEAVE dataset names keyed by split."""
    return {
        "train": config.get(
            "training_weave_dataset", f"tau2-{domain}-training-scenarios"
        ),
        "validation": config.get(
            "validation_weave_dataset", f"tau2-{domain}-validation-scenarios"
        ),
        "base": config.get("base_weave_dataset", f"tau2-{domain}-base-scenarios"),
    }


def _all_artifacts_present(project: str, artifact_names: dict) -> tuple[bool, list[str]]:
    """Probe W&B Artifact registry for `<project>/<name>:latest` for every name.

    Returns (all_present, missing_names). Uses wandb.Api() so we don't need an
    active wandb.run (which is exactly the point — we want to decide BEFORE
    starting one).
    """
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"  [probe] wandb.Api() failed ({type(e).__name__}: {e}); will re-upload")
        return False, list(artifact_names.values())
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    if not entity:
        print("  [probe] no WANDB_ENTITY and no api.default_entity; will re-upload")
        return False, list(artifact_names.values())
    missing = []
    for name in artifact_names.values():
        ref = f"{entity}/{project}/{name}:latest"
        try:
            api.artifact(ref)
        except Exception:
            missing.append(name)
    return (not missing), missing


def _all_weave_datasets_present(project: str, weave_names: dict) -> tuple[bool, list[str]]:
    """Probe Weave for each dataset by name. weave.init() must already be done."""
    missing = []
    for name in weave_names.values():
        try:
            weave.ref(name).get()
        except Exception:
            missing.append(name)
    return (not missing), missing


def main(
    config_path: str = "train_config.yaml",
    domain_override: str | None = None,
    info_only: bool = False,
    num_base_tasks: int | None = None,
    skip_if_exists: bool = True,
    force: bool = False,
):
    config = load_config(config_path)
    domain = domain_override if domain_override is not None else config["domain"]
    project = config["project"]
    # Group is written into the snapshot config by run_pipeline.make_snapshot,
    # e.g. `group: pipeline-04241730`. When this script is invoked outside the
    # pipeline (ad-hoc), no group is set and we default to "manual".
    group = config.get("group") or "manual"
    suffix = group.removeprefix("pipeline-") if group.startswith("pipeline-") else group

    if info_only:
        splits = load_task_splits(domain)
        if splits is None:
            print(f"Domain '{domain}' has no task splits.")
            return
        print(f"Domain: {domain}")
        print("Split sizes:")
        for name, task_ids in sorted(splits.items()):
            print(f"  {name}: {len(task_ids)} tasks")
        total = len(get_tasks(task_set_name=domain, task_split_name=None))
        print(f"Total task set size: {total} tasks")
        return

    # Idempotency probe: bail out before opening a wandb.run if everything is
    # already there. This is the common case in autoresearch (iter 2, 3, ...).
    artifact_names = _resolve_artifact_names(config, domain)
    weave_names = _resolve_weave_names(config, domain)

    if skip_if_exists and not force:
        print(f"\n[upload] Probing project='{project}' for existing dataset assets...")
        artifacts_ok, art_missing = _all_artifacts_present(project, artifact_names)
        # Weave probe needs weave.init() but is read-only; safe to do here
        # without a wandb.run.
        weave.init(project)
        weave_ok, weave_missing = _all_weave_datasets_present(project, weave_names)
        if artifacts_ok and weave_ok:
            print(
                f"[upload] All 3 W&B artifacts and 3 Weave datasets already exist in "
                f"project '{project}'. Skipping upload (no wandb.run created).\n"
                f"         artifacts: {sorted(artifact_names.values())}\n"
                f"         weave    : {sorted(weave_names.values())}\n"
                f"         Pass --force to re-upload."
            )
            return
        else:
            print(
                f"[upload] Missing assets detected; proceeding with full upload.\n"
                f"         missing artifacts: {art_missing}\n"
                f"         missing weave    : {weave_missing}"
            )

    # Full splits from domain data (train/test/base); base can be limited by --num-base-tasks
    training_tasks = get_tasks(
        task_set_name=domain,
        task_split_name="train",
        num_tasks=None,
    )
    validation_tasks = get_tasks(
        task_set_name=domain,
        task_split_name="test",
        num_tasks=None,
    )
    base_tasks = get_tasks(
        task_set_name=domain,
        task_split_name="base",
        num_tasks=num_base_tasks,
    )

    training_data = [{"task_id": t.id, "domain": domain} for t in training_tasks]
    validation_data = [{"task_id": t.id, "domain": domain} for t in validation_tasks]
    base_data = [{"task_id": t.id, "domain": domain} for t in base_tasks]

    print(
        f"Domain: {domain} | Training: {len(training_tasks)} (train) | "
        f"Validation: {len(validation_tasks)} (test) | Base: {len(base_tasks)}"
    )

    # Initialize W&B run — pinned to the iteration group so it nests under the
    # same expandable bundle as sft, rl, leaderboard for this iteration.
    run = wandb.init(
        project=project,
        group=group,
        name=f"upload-{suffix}",
        job_type="dataset-setup",
        tags=["tau2", "training", "validation", "dataset", "artifact", "weave"],
        config=config,
    )

    weave.init(project)

    run.summary["training_scenarios_count"] = len(training_tasks)
    run.summary["validation_scenarios_count"] = len(validation_tasks)
    run.summary["base_scenarios_count"] = len(base_tasks)
    run.summary["training_split"] = "train"
    run.summary["validation_split"] = "test"
    run.summary["domain"] = domain

    # ── Training ──
    training_file = "training_scenarios.json"
    with open(training_file, "w") as f:
        json.dump(training_data, f, indent=2)

    training_art_name = artifact_names["train"]
    training_artifact = wandb.Artifact(
        name=training_art_name,
        type="dataset",
        description=f"Training scenarios for tau2-bench {domain} ({len(training_tasks)} tasks from split train)",
        metadata={
            "split": "train",
            "num_scenarios": len(training_tasks),
            "domain": domain,
        },
    )
    training_artifact.add_file(training_file)
    run.log_artifact(training_artifact)

    training_weave_dataset = weave.Dataset(
        name=weave_names["train"],
        rows=training_data,
    )
    weave.publish(training_weave_dataset)

    # ── Validation ──
    validation_file = "validation_scenarios.json"
    with open(validation_file, "w") as f:
        json.dump(validation_data, f, indent=2)

    val_art_name = artifact_names["validation"]
    validation_artifact = wandb.Artifact(
        name=val_art_name,
        type="dataset",
        description=f"Validation scenarios for tau2-bench {domain} ({len(validation_tasks)} tasks from split test)",
        metadata={
            "split": "test",
            "num_scenarios": len(validation_tasks),
            "domain": domain,
        },
    )
    validation_artifact.add_file(validation_file)
    run.log_artifact(validation_artifact)

    validation_weave_dataset = weave.Dataset(
        name=weave_names["validation"],
        rows=validation_data,
    )
    weave.publish(validation_weave_dataset)

    # ── Base (same order as tau2 run --num-tasks N) ──
    base_file = "base_scenarios.json"
    with open(base_file, "w") as f:
        json.dump(base_data, f, indent=2)

    base_art_name = artifact_names["base"]
    base_artifact = wandb.Artifact(
        name=base_art_name,
        type="dataset",
        description=f"Base scenarios for tau2-bench {domain} ({len(base_tasks)} tasks; same order as tau2 run --num-tasks N)",
        metadata={
            "split": "base",
            "num_scenarios": len(base_tasks),
            "domain": domain,
        },
    )
    base_artifact.add_file(base_file)
    run.log_artifact(base_artifact)

    base_weave_dataset = weave.Dataset(
        name=weave_names["base"],
        rows=base_data,
    )
    weave.publish(base_weave_dataset)

    print(
        f"Uploaded training ({len(training_tasks)}) + validation ({len(validation_tasks)}) + base ({len(base_tasks)}) "
        f"to W&B and Weave"
    )
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload tau2 training and validation sets to W&B and Weave")
    parser.add_argument("--config", default="train_config.yaml", help="Path to YAML config")
    parser.add_argument("--domain", type=str, default=None, help="Override domain (e.g. telecom) for upload")
    parser.add_argument(
        "--num-base-tasks",
        type=int,
        default=None,
        metavar="N",
        help="Upload first N tasks for base dataset (same order as tau2 run --num-tasks N). Default: full base split.",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Print dataset/split sizes for the domain and exit (no upload)",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-if-exists",
        dest="skip_if_exists",
        action="store_true",
        default=True,
        help="(default) Skip upload if all 3 W&B dataset artifacts AND all 3 Weave datasets already exist in the project.",
    )
    skip_group.add_argument(
        "--no-skip-if-exists",
        dest="skip_if_exists",
        action="store_false",
        help="Always upload, even if assets already exist (creates a new wandb.run and bumps artifact versions).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Equivalent to --no-skip-if-exists; always re-upload (creates :vN+1 artifact versions).",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        domain_override=args.domain,
        info_only=args.info_only,
        num_base_tasks=args.num_base_tasks,
        skip_if_exists=args.skip_if_exists,
        force=args.force,
    )

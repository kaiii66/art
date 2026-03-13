"""
Upload tau2 training, validation, and base scenarios to W&B artifacts and Weave.

By default loads full train/test/base splits from the domain data (tasks.json +
split_tasks.json). Use --num-base-tasks N to limit base to first N tasks (same
order as tau2 run --num-tasks N). Domain (and project) come from config.

Usage:
    python upload_dataset_to_wandb.py
    python upload_dataset_to_wandb.py --config train_config.yaml
    python upload_dataset_to_wandb.py --domain telecom
    python upload_dataset_to_wandb.py --domain telecom --num-base-tasks 3
    python upload_dataset_to_wandb.py --domain telecom --info-only
"""
import argparse
import json
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


def main(
    config_path: str = "train_config.yaml",
    domain_override: str | None = None,
    info_only: bool = False,
    num_base_tasks: int | None = None,
):
    config = load_config(config_path)
    domain = domain_override if domain_override is not None else config["domain"]
    project = config["project"]

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

    # Initialize W&B run
    run = wandb.init(
        project=project,
        job_type="dataset-setup",
        tags=["tau2", "training", "validation", "dataset", "artifact", "weave"],
        name="tau2-upload-datasets",
        config=config,
    )

    # Initialize Weave
    weave.init(project)

    # Log dataset statistics
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

    training_art_name = config.get("training_dataset_artifact", f"tau2-{domain}-training-scenarios").split(":")[0]
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

    training_weave_name = config.get("training_weave_dataset", f"tau2-{domain}-training-scenarios")
    training_weave_dataset = weave.Dataset(
        name=training_weave_name,
        rows=training_data,
    )
    weave.publish(training_weave_dataset)

    # ── Validation ──
    validation_file = "validation_scenarios.json"
    with open(validation_file, "w") as f:
        json.dump(validation_data, f, indent=2)

    val_art_name = config.get("validation_dataset_artifact", f"tau2-{domain}-validation-scenarios").split(":")[0]
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

    val_weave_name = config.get("validation_weave_dataset", f"tau2-{domain}-validation-scenarios")
    validation_weave_dataset = weave.Dataset(
        name=val_weave_name,
        rows=validation_data,
    )
    weave.publish(validation_weave_dataset)

    # ── Base (same order as tau2 run --num-tasks N) ──
    base_file = "base_scenarios.json"
    with open(base_file, "w") as f:
        json.dump(base_data, f, indent=2)

    base_art_name = config.get("base_weave_dataset", f"tau2-{domain}-base-scenarios")
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
        name=base_art_name,
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
    args = parser.parse_args()
    main(
        config_path=args.config,
        domain_override=args.domain,
        info_only=args.info_only,
        num_base_tasks=args.num_base_tasks,
    )

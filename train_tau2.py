"""
ART GRPO training script for tau2-bench.

For baseline/leaderboard evaluation, use create_leaderboard.py instead.

Usage:
  python train_tau2.py
"""
import argparse
import asyncio
import json
import random
import yaml
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import weave
import wandb
import art
from art.serverless.backend import ServerlessBackend
from art.utils import iterate_dataset

from tau2.run import get_tasks
from tau2_art_helpers import (
    Tau2TaskScenario,
    tau2_rollout,
)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# Load training tasks from artifact
# ─────────────────────────────────────────────────────────────────────

def load_training_tasks_from_artifact(config, num_tasks=None):
    """Load training tasks from the W&B dataset artifact, in row order.

    Returns list of Task if config has training_dataset_artifact and download
    succeeds, else None (caller should fall back to get_tasks with task split).
    If num_tasks is set, returns only the first num_tasks (same as baseline loader).
    """
    artifact_name = config.get("training_dataset_artifact")
    if not artifact_name:
        return None
    run = wandb.run
    if run is None:
        return None
    try:
        artifact = run.use_artifact(artifact_name)
        download_dir = artifact.download()
        path = Path(download_dir) / "training_scenarios.json"
        if not path.exists():
            print(f"Artifact dir missing training_scenarios.json. Using task split.")
            return None
        with open(path) as f:
            rows = json.load(f)
    except Exception as e:
        print(f"Could not load training artifact {artifact_name}: {e}. Using task split.")
        return None
    if not rows:
        return []
    task_ids = [r["task_id"] for r in rows]
    domain = config.get("domain") or (rows[0].get("domain") if rows else None)
    if not domain:
        raise ValueError("config must set domain or rows must include domain")
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name="train",
        task_ids=task_ids,
    )
    id_to_task = {t.id: t for t in tasks}
    result = [id_to_task[tid] for tid in task_ids]
    if num_tasks is not None:
        result = result[:num_tasks]
    return result


# ─────────────────────────────────────────────────────────────────────
# Load validation tasks (Weave or W&B artifact); raise if configured but load fails
# ─────────────────────────────────────────────────────────────────────


def load_validation_tasks(config):
    """Load validation tasks from Weave or W&B artifact.

    Uses validation_weave_dataset if set, else validation_dataset_artifact.
    Returns empty list if neither is configured. Raises if a source is configured
    but download/load fails.
    """
    weave_name = config.get("validation_weave_dataset")
    artifact_name = config.get("validation_dataset_artifact")
    if not weave_name and not artifact_name:
        return []

    domain = config.get("domain")
    val_split = config.get("validation_task_split", "test")

    if weave_name:
        try:
            weave.init(config["project"])
            dataset = weave.ref(weave_name).get()
            rows = dataset.rows
        except Exception as e:
            raise RuntimeError(
                f"Could not load validation Weave dataset {weave_name}: {e}. "
                "Check validation_weave_dataset and run upload_dataset_to_wandb.py."
            ) from e
        if not rows:
            return []
        task_ids = [r["task_id"] for r in rows]
        if not domain:
            domain = rows[0].get("domain") if rows else None
        if not domain:
            raise ValueError("config must set domain or validation rows must include domain")
        tasks = get_tasks(
            task_set_name=domain,
            task_split_name=val_split,
            task_ids=task_ids,
        )
        id_to_task = {t.id: t for t in tasks}
        return [id_to_task[tid] for tid in task_ids]

    # validation_dataset_artifact
    run = wandb.run
    if run is None:
        raise RuntimeError(
            "validation_dataset_artifact is set but wandb.run is None. "
            "Ensure wandb.init() is called before load_validation_tasks."
        )
    try:
        artifact = run.use_artifact(artifact_name)
        download_dir = artifact.download()
        path = Path(download_dir) / "validation_scenarios.json"
        if not path.exists():
            raise FileNotFoundError(f"Artifact dir missing validation_scenarios.json: {download_dir}")
        with open(path) as f:
            rows = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load validation artifact {artifact_name}: {e}. "
            "Check validation_dataset_artifact and run upload_dataset_to_wandb.py."
        ) from e
    if not rows:
        return []
    task_ids = [r["task_id"] for r in rows]
    if not domain:
        domain = rows[0].get("domain") if rows else None
    if not domain:
        raise ValueError("config must set domain or artifact rows must include domain")
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=val_split,
        task_ids=task_ids,
    )
    id_to_task = {t.id: t for t in tasks}
    return [id_to_task[tid] for tid in task_ids]



# ─────────────────────────────────────────────────────────────────────
# Training mode
# ─────────────────────────────────────────────────────────────────────

async def run_training(model, backend, training_tasks, config, max_train_steps=None, validation_tasks=None):
    """GRPO training loop with optional validation every N steps and checkpoint metadata artifact (art-demo pattern)."""
    if validation_tasks is None:
        validation_tasks = []
    domain = config["domain"]
    user_llm = config["user_llm"]
    user_llm_args = config.get("user_llm_args", {"temperature": 1.0, "max_tokens": 16384})
    # NOTE: training rollouts deliberately do NOT pass `agent_llm` to tau2_rollout.
    # When agent_llm is None, tau2_rollout uses ARTAgent against the trainable
    # LoRA (what we want to optimize). Passing config['agent_llm'] would switch
    # to the static LLMAgent path and rollouts would no longer sample from the
    # policy being trained. Only `agent_llm_args` (temperature, max_tokens) is
    # plumbed through so the inference call has sane decoding settings.
    agent_llm_args = config.get("agent_llm_args", {"temperature": 1.0, "max_tokens": 16384})
    max_steps = config.get("max_orchestrator_steps", 30)
    groups_per_step = config["groups_per_step"]
    rollouts_per_group = config["rollouts_per_group"]
    learning_rate = config["learning_rate"]
    num_epochs = config.get("num_epochs", 1)
    use_shaped = config.get("shaped_reward", False)
    shaped_weights = config.get("shaped_reward_weights")
    validation_interval = config.get("validation_step_interval", 5)

    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"Model            : {config['base_model']}")
    print(f"Domain           : {domain}")
    print(f"Training tasks   : {len(training_tasks)}")
    if validation_tasks:
        print(f"Validation tasks : {len(validation_tasks)}")
    print(f"groups_per_step  : {groups_per_step}")
    print(f"rollouts_per_group: {rollouts_per_group}")
    print(f"learning_rate    : {learning_rate}")
    print(f"shaped_reward    : {use_shaped}")
    if max_train_steps:
        print(f"max_train_steps  : {max_train_steps}")
    print(f"{'='*60}\n")

    training_scenarios = [
        Tau2TaskScenario(task_id=t.id, domain=domain) for t in training_tasks
    ]

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        initial_step=0,
    )


    steps_completed = 0
    for batch in training_iterator:
        if max_train_steps and steps_completed >= max_train_steps:
            print(f"\nReached max_train_steps={max_train_steps}, stopping.")
            break

        print(f"\n--- Step {batch.step} (epoch {batch.epoch}, epoch_step {batch.epoch_step}) ---")

        train_groups = []
        for scenario in batch.items:
            scenario.step = batch.step
            train_groups.append(
                art.TrajectoryGroup(
                    tau2_rollout(
                        model,
                        scenario,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        agent_llm_args=agent_llm_args,
                        max_steps=max_steps,
                        use_shaped_reward=use_shaped,
                        shaped_reward_weights=shaped_weights,
                    )
                    for _ in range(rollouts_per_group)
                )
            )

        finished_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="rollouts",
            max_exceptions=rollouts_per_group * len(batch.items),
        )

        # Log training metrics before GRPO step
        all_trajs = [t for g in finished_groups for t in g.trajectories]
        if all_trajs:
            rewards = [t.reward for t in all_trajs]
            avg_reward = sum(rewards) / len(rewards)
            avg_task_reward = sum(t.metrics.get("task_reward", 0.0) for t in all_trajs) / len(all_trajs)
            avg_success = sum(t.metrics.get("success", 0.0) for t in all_trajs) / len(all_trajs)
            total_tokens = sum(t.metadata.get("completion_tokens", 0) for t in all_trajs)
            reward_std = (sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5

            label = "shaped" if use_shaped else "binary"
            print(f"  train reward({label})={avg_reward:.3f} (std={reward_std:.3f})  "
                  f"task_reward={avg_task_reward:.3f}  success={avg_success:.1%}  tokens={total_tokens}")

            # Log rollout metrics to W&B immediately so they appear even if model.train() fails or times out
            train_metrics = {
                "train/reward": avg_reward,
                "train/task_reward": avg_task_reward,
                "train/success": avg_success,
                "train/reward_std": reward_std,
                "train/completion_tokens": total_tokens,
                "train/num_trajectories": len(all_trajs),
            }
            if use_shaped:
                for key in ("action_fraction", "nl_fraction", "communicate_fraction",
                            "termination_bonus", "efficiency"):
                    vals = [t.metrics.get(key, -1.0) for t in all_trajs]
                    active = [v for v in vals if v >= 0]
                    if active:
                        train_metrics[f"train/{key}"] = sum(active) / len(active)
            wandb.log(train_metrics)

        # GRPO training step
        max_retries = 3
        train_success = False
        for attempt in range(max_retries):
            try:
                async with asyncio.timeout(1800):
                    # Note: KL-against-reference (`beta`/`kl_penalty_coef`) is not
                    # exposed by the current ServerlessBackend.train() API, so we
                    # don't pass it. config['kl_beta'] is currently unused.
                    result = await backend.train(
                        model,
                        finished_groups,
                        learning_rate=learning_rate,
                    )
                    await model.log(
                        finished_groups,
                        metrics=result.metrics,
                        step=result.step,
                        split="train",
                    )
                    train_success = True
                    break
            except (asyncio.TimeoutError, Exception) as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 10
                    print(f"  Training error (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"  Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  Training FAILED after {max_retries} attempts. Skipping batch.")

        if not train_success:
            print(f" Skipping validation because training failed")
            continue

        steps_completed += 1

        # Validation (art-demo pattern: run every validation_step_interval when validation_tasks provided)
        if validation_tasks and batch.step % validation_interval == 0:
            print(f"\n  Running validation on {len(validation_tasks)} tasks...")
            val_groups = []
            for task in validation_tasks:
                scenario = Tau2TaskScenario(step=batch.step, task_id=task.id, domain=domain)
                val_groups.append(
                    art.TrajectoryGroup([
                        tau2_rollout(
                            model,
                            scenario,
                            user_llm=user_llm,
                            user_llm_args=user_llm_args,
                            agent_llm_args=agent_llm_args,
                            max_steps=max_steps,
                        )
                    ])
                )
            finished_val_groups = await art.gather_trajectory_groups(
                val_groups,
                pbar_desc="validation",
                max_exceptions=len(validation_tasks),
            )
            await model.log(finished_val_groups, split="val")
            val_trajs = [t for g in finished_val_groups for t in g.trajectories]
            if val_trajs:
                val_reward = sum(t.reward for t in val_trajs) / len(val_trajs)
                val_success = sum(t.metrics.get("success", 0.0) for t in val_trajs) / len(val_trajs)
                val_tokens = sum(t.metadata.get("completion_tokens", 0) for t in val_trajs)
                wandb.log({
                    "val/reward": val_reward,
                    "val/success": val_success,
                    "val/completion_tokens": val_tokens,
                    "val/num_tasks": len(val_trajs),
                })
                print(f"  val reward={val_reward:.3f}  success={val_success:.1%}")

        # Checkpoint metadata artifact (art-demo pattern; actual weights managed by ART backend)
        if config.get("save_checkpoint_artifact", True):
            current_step = await model.get_step()
            checkpoint_artifact = wandb.Artifact(
                name=f"{config['base_model'].replace('/', '-')}-checkpoint",
                type="model",
                description=f"Checkpoint at step {current_step}",
                metadata={
                    "step": current_step,
                    "epoch": batch.epoch,
                    "epoch_step": batch.epoch_step,
                    "learning_rate": learning_rate,
                    "base_model": config["base_model"],
                    "training_dataset_artifact": config.get("training_dataset_artifact"),
                    "validation_dataset_artifact": config.get("validation_dataset_artifact"),
                },
            )
            wandb.run.log_artifact(checkpoint_artifact)

    print(f"\nTraining complete. {steps_completed} steps finished.")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

async def main(args):
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.num_tasks is not None:
        config["_num_tasks"] = args.num_tasks
    if args.groups_per_step is not None:
        config["groups_per_step"] = args.groups_per_step
    if args.rollouts_per_group is not None:
        config["rollouts_per_group"] = args.rollouts_per_group

    random.seed(config.get("random_seed", 42))

    # ── Names (W&B run + ART model) ──
    # The W&B run name is always uniquely timestamped so each training run is
    # distinct in the W&B UI. The ART model.name (= W&B artifact collection)
    # is either:
    #   - `continue_from_model`, when set, so GRPO continues that collection's
    #     checkpoint history (e.g. RL on top of an SFT collection's step 6 →
    #     produces step 7, 8, … in the same collection, all servable).
    #   - `<model_name>-<timestamp>`, otherwise, for a fresh LoRA in a new
    #     auto-named collection.
    lr_str = f"{config['learning_rate']:.0e}".replace("-0", "-")
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    run_name = (
        f"train-{config['domain']}-g{config['groups_per_step']}"
        f"-r{config['rollouts_per_group']}-lr{lr_str}"
        f"-{now_pt.strftime('%Y%m%d-%H%M')}"
    )
    continue_from = config.get("continue_from_model")
    if continue_from:
        model_name = continue_from
    else:
        model_name = f"{config['model_name']}-{now_pt.strftime('%Y%m%d-%H%M')}"

    # ── W&B (main training run) ──
    wandb.init(
        project=config["project"],
        name=run_name,
        config=config,
        job_type=config.get("wandb_job_type", "train"),
    )

    # ── ART model ──
    model = art.TrainableModel(
        name=model_name,
        project=config["project"],
        base_model=config["base_model"],
    )
    backend = ServerlessBackend()
    await model.register(backend)

    starting_step = await model.get_step()
    if continue_from:
        print(
            f"Continuing collection '{model_name}' from step {starting_step}. "
            f"GRPO will append step {starting_step + 1}, {starting_step + 2}, …"
        )
        if starting_step == 0:
            print(
                "WARNING: continue_from_model is set but the collection's latest "
                "registered step is 0 (base init). RL will effectively train from "
                "the base LoRA. Verify the source collection actually has the "
                "expected SFT checkpoint registered with the ART backend."
            )
    else:
        print(
            f"Fresh collection '{model_name}'. GRPO starts from base LoRA "
            f"(step {starting_step})."
        )

    # So create_leaderboard.py can find this run's model without --trained-model-name
    (config_path.resolve().parent / ".last_trained_model").write_text(model_name)

    domain = config["domain"]
    task_split = config.get("task_split_name", "train")
    num_tasks = config.get("_num_tasks")

    training_tasks = load_training_tasks_from_artifact(config, num_tasks=num_tasks)
    if training_tasks is None:
        training_tasks = get_tasks(
            task_set_name=domain,
            task_split_name=task_split,
            num_tasks=num_tasks,
        )
        print(f"Loaded {len(training_tasks)} training tasks from {domain}/{task_split}")
    else:
        print(f"Loaded {len(training_tasks)} training tasks from W&B artifact ({config['training_dataset_artifact']})")
    validation_tasks = load_validation_tasks(config)
    if validation_tasks:
        print(f"Loaded {len(validation_tasks)} validation tasks")
    await run_training(
        model,
        backend,
        training_tasks,
        config,
        max_train_steps=args.max_train_steps,
        validation_tasks=validation_tasks,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ART GRPO training for tau2-bench")
    parser.add_argument("--config", default="train_config.yaml", help="Path to YAML config")
    parser.add_argument("--num-tasks", type=int, default=None, help="Limit number of training tasks")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Stop after N training steps")
    parser.add_argument("--groups-per-step", type=int, default=None, help="Override groups_per_step")
    parser.add_argument("--rollouts-per-group", type=int, default=None, help="Override rollouts_per_group")
    args = parser.parse_args()

    asyncio.run(main(args))

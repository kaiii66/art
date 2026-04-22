"""
Shaped-reward leaderboard for tau2-bench: compare base (Qwen) and trained (GRPO) models
on the held-out validation set with TWO columns:

  - success.mean       (binary, headline)    "did the model solve the task?"
  - task_reward.mean   (shaped, diagnostic)  continuous training-objective signal

No pass^k. The shaped column reports a different unit (continuous, possibly >1 before
step penalty) than the binary leaderboard in create_leaderboard.py, so this script
publishes its own Weave Evaluation / Leaderboard objects (suffixed with "-shaped")
to keep the two histories cleanly separated.

Important caveats:
  - task_reward.mean here is the CONTINUOUS shaped score, NOT comparable to the
    binary task_reward in create_leaderboard.py or to published tau2-bench numbers.
  - Use success.mean for objective comparison; task_reward.mean is for fine-grained
    training-objective signal (especially useful early in training when binary
    success is near zero).
  - At num_trials=1, single-sample variance on ~40 validation tasks is roughly
    +/- 7-8pp on the binary column.

Usage:
    python create_leaderboard_shaped_reward.py --models all --publish-leaderboard
    python create_leaderboard_shaped_reward.py --models trained --trained-model-name <name>

Model options: base, trained, all
"""
import argparse
import asyncio
import yaml
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
import wandb
import art
from art.serverless.backend import ServerlessBackend

from tau2_art_helpers import (
    Tau2BaseModelWrapper,
    score_task_reward,
    score_success,
)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


async def main(
    config_path: str = "train_config.yaml",
    models_to_eval: list = None,
    trained_model_name: str = None,
    trained_model_step: int = None,
    trained_model_alias: str = None,
    publish_leaderboard: bool = False,
):
    if models_to_eval is None:
        models_to_eval = ["all"]
    eval_all = "all" in models_to_eval
    should_eval_base = eval_all or "base" in models_to_eval
    should_eval_trained = eval_all or "trained" in models_to_eval

    config = load_config(config_path)
    project = config["project"]
    domain = config["domain"]
    base_model = config["base_model"]
    agent_llm = config.get("agent_llm", f"wandb/{base_model}")
    user_llm = config["user_llm"]
    eval_weave = config.get("validation_weave_dataset") or f"tau2-{domain}-validation-scenarios"
    trained_name = trained_model_name or config.get("leaderboard_trained_model_name")
    if not trained_name:
        last_model_file = Path(config_path).resolve().parent / ".last_trained_model"
        if last_model_file.exists():
            trained_name = last_model_file.read_text().strip()
    # Pin the trained-model evaluation to a specific LoRA checkpoint. Resolution
    # order (highest precedence first):
    #   1. CLI --trained-model-alias / --trained-model-step
    #   2. config.leaderboard_trained_model_alias / leaderboard_trained_model_step
    #   3. .best_rl_step file next to the snapshot config (auto-written by
    #      train_tau2.py at the step with the highest val/reward). This makes
    #      `eval_rl` in run_pipeline.py automatically score the best RL step
    #      instead of whichever step is currently :latest.
    pinned_alias = trained_model_alias or config.get("leaderboard_trained_model_alias")
    pinned_step = trained_model_step
    if pinned_step is None:
        pinned_step = config.get("leaderboard_trained_model_step")
    if pinned_step is None and pinned_alias is None:
        best_step_file = Path(config_path).resolve().parent / ".best_rl_step"
        if best_step_file.exists():
            try:
                pinned_step = int(best_step_file.read_text().strip())
                print(
                    f"  [eval_rl] auto-pinned to best RL step {pinned_step} "
                    f"from {best_step_file.name}"
                )
            except (ValueError, OSError) as e:
                print(f"  [eval_rl] could not read {best_step_file}: {e}")
    lb_config = config.get("leaderboard", {})
    num_trials = lb_config.get("num_trials", 1)
    max_steps = lb_config.get("max_steps", config.get("max_orchestrator_steps", 30))
    user_llm_args = lb_config.get("user_llm_args", config.get("user_llm_args", {"temperature": 1.0}))
    agent_llm_args = lb_config.get("agent_llm_args", {})

    # Shaped-reward kwargs forwarded to Tau2BaseModelWrapper -> tau2_rollout.
    # When use_shaped_reward=True, traj.reward becomes the continuous shaped score
    # while traj.metrics["success"] stays binary (preserved inside tau2_rollout).
    shaped_kwargs = dict(
        use_shaped_reward=config.get("shaped_reward", False),
        shaped_reward_weights=config.get("shaped_reward_weights", {}),
    )

    wc = weave.init(project)

    print("\nLoading validation dataset...")
    try:
        original = weave.ref(eval_weave).get()
    except Exception as e:
        raise RuntimeError(f"Could not load Weave dataset {eval_weave}: {e}") from e
    print(f"Loaded {len(original.rows)} rows from {eval_weave}")

    leaderboard_dataset_name = f"tau2-{domain}-validation-scenarios-leaderboard-shaped"
    # Always rebuild leaderboard dataset from current validation data so we never
    # use a stale cached copy (e.g. if the config was switched to a new domain).
    dataset = weave.Dataset(name=leaderboard_dataset_name, rows=original.rows)
    weave.publish(dataset)
    print("Published leaderboard dataset from current validation data")

    scorers = [score_success, score_task_reward]
    eval_name = f"tau2-{domain}-evaluation-leaderboard-shaped"
    shared_evaluation = weave.Evaluation(
        name=eval_name,
        dataset=dataset,
        scorers=scorers,
        trials=num_trials,
    )
    weave.publish(shared_evaluation)
    print(f"Using evaluation with {num_trials} trial(s) per task (shaped reward enabled: {shaped_kwargs['use_shaped_reward']})")

    run = wandb.init(
        project=project,
        name="tau2-leaderboard-evaluation-shaped",
        config=config,
        job_type="leaderboard",
    )

    models = []
    model_names = []
    display_names = []

    if should_eval_base:
        print(f"\nLoading base model: {base_model} (agent_llm={agent_llm})")
        base_wrapper = Tau2BaseModelWrapper(
            name="qwen3-30b-baseline",
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
        models.append(base_wrapper)
        model_names.append("base")
        display_names.append(f"{base_model} (base)")

    if should_eval_trained and trained_name:
        try:
            print(f"\nLoading trained model: {trained_name}...")
            backend = ServerlessBackend()
            trained_model = art.TrainableModel(
                name=trained_name,
                project=project,
                base_model=base_model,
            )
            await trained_model.register(backend)
            latest_step = await trained_model.get_step()
            if pinned_alias is not None:
                eval_step_label = f"pinned alias :{pinned_alias}"
                print(
                    f"Pinning evaluation to artifact alias :{pinned_alias} "
                    f"(collection latest is step {latest_step})"
                )
            elif pinned_step is not None:
                eval_step_label = f"pinned step {pinned_step}"
                print(
                    f"Pinning evaluation to checkpoint :step{pinned_step} "
                    f"(collection latest is step {latest_step})"
                )
            else:
                eval_step_label = f"latest step {latest_step}"
            trained_display = f"{base_model} (GRPO @ {eval_step_label})"
            trained_wrapper = Tau2BaseModelWrapper(
                model=trained_model,
                model_name=trained_display,
                domain=domain,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                agent_llm_args=agent_llm_args,
                max_steps=max_steps,
                pinned_step=pinned_step,
                pinned_alias=pinned_alias,
                **shaped_kwargs,
            )
            models.append(trained_wrapper)
            model_names.append("trained")
            display_names.append(trained_display)
        except Exception as e:
            print(f"Could not load trained model {trained_name}: {e}")
    elif should_eval_trained and not trained_name:
        print("\nSkipping trained model (set leaderboard_trained_model_name in config, pass --trained-model-name, or run train first to create .last_trained_model)")

    if not models:
        print("\nNo models to evaluate.")
        run.finish()
        return

    for idx, (model, name, display_name) in enumerate(zip(models, model_names, display_names), 1):
        print(f"\nEvaluating {idx}/{len(models)}: {display_name}")
        await shared_evaluation.evaluate(model, __weave={"display_name": display_name})
        print(f"Completed: {display_name}")

    try:
        eval_ref_uri = get_ref(shared_evaluation).uri()
        lb_columns = [
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref_uri,
                scorer_name="score_success",
                summary_metric_path="success.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref_uri,
                scorer_name="score_task_reward",
                summary_metric_path="task_reward.mean",
            ),
        ]
        leaderboard_spec = leaderboard.Leaderboard(
            name=f"tau2-{domain}-leaderboard-shaped-v1",
            description=f"tau2-bench {domain} held-out validation: binary success (headline) + shaped task_reward (diagnostic).",
            columns=lb_columns,
        )
        if publish_leaderboard:
            ref = weave.publish(leaderboard_spec)
            print(f"\nLeaderboard published: {ref}")
        else:
            print("\nLeaderboard spec created but not published (results will appear in existing leaderboard).")
    except Exception as e:
        print(f"Failed to create leaderboard: {e}")
        import traceback
        traceback.print_exc()

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tau2-bench shaped-reward leaderboard: compare base and trained models with shaped + binary metrics")
    parser.add_argument("--config", default="train_config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["base", "trained", "all"],
        default=["all"],
        help="Models to evaluate (default: all)",
    )
    parser.add_argument("--trained-model-name", type=str, default=None, help="Trained model name (overrides config)")
    parser.add_argument(
        "--trained-model-step",
        type=int,
        default=None,
        help=(
            "Pin trained-model evaluation to LoRA checkpoint alias :step{N} "
            "instead of :latest. Default: auto-read .best_rl_step next to the "
            "config (written by train_tau2.py at the best val/reward step), "
            "else fall back to :latest."
        ),
    )
    parser.add_argument(
        "--trained-model-alias",
        type=str,
        default=None,
        help=(
            "Pin trained-model evaluation directly to a W&B artifact alias "
            "(e.g. 'v1', 'latest'). Takes precedence over --trained-model-step "
            "and over .best_rl_step auto-discovery."
        ),
    )
    parser.add_argument(
        "--publish-leaderboard",
        action="store_true",
        help="Publish/overwrite leaderboard definition (use only for first time or to update structure)",
    )
    args = parser.parse_args()
    asyncio.run(main(
        config_path=args.config,
        models_to_eval=args.models,
        trained_model_name=args.trained_model_name,
        trained_model_step=args.trained_model_step,
        trained_model_alias=args.trained_model_alias,
        publish_leaderboard=args.publish_leaderboard,
    ))

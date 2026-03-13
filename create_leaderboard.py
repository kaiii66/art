"""
Leaderboard script for tau2-bench: compare base (Qwen) and trained (GRPO) models.

Uses Weave Evaluation on the baseline set and builds a leaderboard with task_reward and success.

Usage:
    python create_leaderboard.py --models all --publish-leaderboard
    python create_leaderboard.py --models trained --trained-model-name <name>

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
    PassAtKScorer,
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
    base_weave = config.get("base_weave_dataset") or f"tau2-{domain}-base-scenarios"
    trained_name = trained_model_name or config.get("leaderboard_trained_model_name")
    if not trained_name:
        last_model_file = Path(config_path).resolve().parent / ".last_trained_model"
        if last_model_file.exists():
            trained_name = last_model_file.read_text().strip()
    lb_config = config.get("leaderboard", {})
    num_trials = lb_config.get("num_trials", 1)
    max_steps = lb_config.get("max_steps", config.get("max_orchestrator_steps", 30))
    user_llm_args = lb_config.get("user_llm_args", config.get("user_llm_args", {"temperature": 1.0}))
    agent_llm_args = lb_config.get("agent_llm_args", {})

    wc = weave.init(project)

    # Load baseline dataset (same set as train_tau2.py --mode baseline)
    print("\nLoading baseline dataset...")
    try:
        original = weave.ref(base_weave).get()
    except Exception as e:
        raise RuntimeError(f"Could not load Weave dataset {base_weave}: {e}") from e
    print(f"Loaded {len(original.rows)} rows from {base_weave}")

    leaderboard_dataset_name = f"tau2-{domain}-base-scenarios-leaderboard"
    # Always build leaderboard dataset from current baseline data so we never use a stale
    # cached dataset (e.g. airline rows when config was switched to telecom).
    dataset = weave.Dataset(name=leaderboard_dataset_name, rows=original.rows)
    weave.publish(dataset)
    print("Published leaderboard dataset from current baseline data")

    # Scorers and shared evaluation
    pass_at_k_scorer = PassAtKScorer(num_trials=num_trials)
    scorers = [score_task_reward, score_success, pass_at_k_scorer]
    eval_name = f"tau2-{domain}-evaluation-leaderboard"
    shared_evaluation = weave.Evaluation(
        name=eval_name,
        dataset=dataset,
        scorers=scorers,
        trials=num_trials,
    )
    weave.publish(shared_evaluation)
    print(f"Using evaluation with {num_trials} trial(s) per task")

    run = wandb.init(
        project=project,
        name="tau2-leaderboard-evaluation",
        config=config,
        job_type="leaderboard",
    )

    models = []
    model_names = []
    display_names = []

    # Base model (via tau2 LLMAgent)
    if should_eval_base:
        print(f"\nLoading base model: {base_model} (agent_llm={agent_llm})")
        base_wrapper = Tau2BaseModelWrapper(
            model=None,
            model_name=base_model,
            domain=domain,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            agent_llm_args=agent_llm_args,
            max_steps=max_steps,
            agent_llm=agent_llm,
        )
        models.append(base_wrapper)
        model_names.append("base")
        display_names.append(f"{base_model} (base)")

    # Trained (ART TrainableModel)
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
            step = await trained_model.get_step()
            trained_wrapper = Tau2BaseModelWrapper(
                model=trained_model,
                model_name=f"{base_model} (GRPO @ step {step})",
                domain=domain,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                agent_llm_args=agent_llm_args,
                max_steps=max_steps,
            )
            models.append(trained_wrapper)
            model_names.append("trained")
            display_names.append(f"{base_model} (GRPO @ step {step})")
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

    # Leaderboard
    try:
        eval_ref_uri = get_ref(shared_evaluation).uri()
        lb_columns = [
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref_uri,
                scorer_name="score_task_reward",
                summary_metric_path="task_reward.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref_uri,
                scorer_name="score_success",
                summary_metric_path="success.mean",
            ),
        ]
        for k in range(1, num_trials + 1):
            lb_columns.append(
                leaderboard.LeaderboardColumn(
                    evaluation_object_ref=eval_ref_uri,
                    scorer_name="PassAtKScorer",
                    summary_metric_path=f"pass^{k}.mean",
                ),
            )
        leaderboard_spec = leaderboard.Leaderboard(
            name=f"tau2-{domain}-leaderboard-v4",
            description=f"tau2-bench {domain}: task_reward, success, and pass^k ({num_trials} trials).",
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
    parser = argparse.ArgumentParser(description="tau2-bench leaderboard: compare base and trained models")
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
        "--publish-leaderboard",
        action="store_true",
        help="Publish/overwrite leaderboard definition (use only for first time or to update structure)",
    )
    args = parser.parse_args()
    asyncio.run(main(
        config_path=args.config,
        models_to_eval=args.models,
        trained_model_name=args.trained_model_name,
        publish_leaderboard=args.publish_leaderboard,
    ))

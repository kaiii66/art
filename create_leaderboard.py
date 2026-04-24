"""
Leaderboard script for tau2-bench: compare base (Qwen) and trained (GRPO) models.

Uses Weave Evaluation on the validation set and builds a leaderboard with task_reward and success.

This is the BINARY-reward sibling of create_leaderboard_shaped_reward.py. Both
leaderboards live in the same W&B project but publish to different Weave
Evaluation/Leaderboard objects (suffixed with `-validation` here vs `-shaped`
there) so their histories don't get tangled.

Idempotency / autoresearch design (mirrors create_leaderboard_shaped_reward.py):
  - weave.publish(Dataset|Evaluation|Leaderboard) is content-hash deduped, so
    republishing the same scaffolding on every invocation is a no-op. No CLI
    flag needed.
  - The base row is gated by a W&B sentinel artifact
    (`tau2-leaderboard-binary-base-evaluated:latest`); after the first
    successful base eval it's auto-skipped. Use --reevaluate-base to force.
  - Each weave.Model wrapper gets a stable, descriptive `name=` field tying
    its leaderboard row to the iteration suffix and the LoRA checkpoint step.

Usage:
    python create_leaderboard.py
    python create_leaderboard.py --models all
    python create_leaderboard.py --models trained --trained-model-name <name>
    python create_leaderboard.py --reevaluate-base

Model options: base, trained, all
"""
import argparse
import asyncio
import os
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


BASE_SENTINEL_ARTIFACT = "tau2-leaderboard-binary-base-evaluated"


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _base_already_evaluated(project: str) -> bool:
    try:
        api = wandb.Api()
    except Exception as e:
        print(f"  [base-sentinel] wandb.Api() failed ({type(e).__name__}: {e}); will evaluate base")
        return False
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    if not entity:
        print("  [base-sentinel] no entity resolvable; will evaluate base")
        return False
    ref = f"{entity}/{project}/{BASE_SENTINEL_ARTIFACT}:latest"
    try:
        api.artifact(ref)
        return True
    except Exception:
        return False


def _log_base_sentinel(run, base_short: str, base_model: str) -> None:
    try:
        art_obj = wandb.Artifact(
            name=BASE_SENTINEL_ARTIFACT,
            type="leaderboard-marker",
            description=(
                "Sentinel: the base model has been evaluated and added to the "
                "binary-reward Weave leaderboard. Subsequent autoresearch "
                "iterations skip the base row to avoid duplicates."
            ),
            metadata={
                "base_model": base_model,
                "base_model_short": base_short,
                "wandb_run_path": run.path if run is not None else None,
            },
        )
        run.log_artifact(art_obj)
        print(f"  [base-sentinel] logged {BASE_SENTINEL_ARTIFACT} so future iterations skip the base row")
    except Exception as e:
        print(f"  [base-sentinel] failed to log sentinel ({type(e).__name__}: {e}); base may be re-evaluated next iter")


async def main(
    config_path: str = "train_config.yaml",
    models_to_eval: list = None,
    trained_model_name: str = None,
    trained_model_step: int = None,
    trained_model_alias: str = None,
    reevaluate_base: bool = False,
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
    base_short = _short(base_model)
    agent_llm = config.get("agent_llm", f"wandb/{base_model}")
    user_llm = config["user_llm"]
    eval_weave = config.get("validation_weave_dataset") or f"tau2-{domain}-validation-scenarios"

    # Iteration group: written by run_pipeline.make_snapshot. Falls back to
    # "manual" for ad-hoc invocations.
    group = config.get("group") or "manual"
    suffix = group.removeprefix("pipeline-") if group.startswith("pipeline-") else group

    trained_name = trained_model_name or config.get("leaderboard_trained_model_name")
    if not trained_name:
        last_model_file = Path(config_path).resolve().parent / ".last_trained_model"
        if last_model_file.exists():
            trained_name = last_model_file.read_text().strip()
    pinned_step = trained_model_step
    if pinned_step is None:
        pinned_step = config.get("leaderboard_trained_model_step")
    pinned_alias = trained_model_alias
    if pinned_alias is None:
        pinned_alias = config.get("leaderboard_trained_model_alias")
    lb_config = config.get("leaderboard", {})
    num_trials = lb_config.get("num_trials", 1)
    max_steps = lb_config.get("max_steps", config.get("max_orchestrator_steps", 30))
    user_llm_args = lb_config.get("user_llm_args", config.get("user_llm_args", {"temperature": 1.0}))
    agent_llm_args = lb_config.get("agent_llm_args", {})

    weave.init(project)

    print("\nLoading validation dataset...")
    try:
        original = weave.ref(eval_weave).get()
    except Exception as e:
        raise RuntimeError(f"Could not load Weave dataset {eval_weave}: {e}") from e
    print(f"Loaded {len(original.rows)} rows from {eval_weave}")

    leaderboard_dataset_name = f"tau2-{domain}-validation-scenarios-leaderboard"
    dataset = weave.Dataset(name=leaderboard_dataset_name, rows=original.rows)
    weave.publish(dataset)

    pass_at_k_scorer = PassAtKScorer(num_trials=num_trials)
    scorers = [score_task_reward, score_success, pass_at_k_scorer]
    eval_name = f"tau2-{domain}-evaluation-leaderboard-validation"
    shared_evaluation = weave.Evaluation(
        name=eval_name,
        dataset=dataset,
        scorers=scorers,
        trials=num_trials,
    )
    weave.publish(shared_evaluation)
    print(f"Using evaluation '{eval_name}' (trials={num_trials})")

    run = wandb.init(
        project=project,
        group=group,
        name=f"leaderboard-binary-{suffix}",
        config=config,
        job_type="leaderboard",
    )

    # Auto-skip base after first iteration (sentinel artifact gate).
    base_eval_skipped = False
    if should_eval_base:
        if reevaluate_base:
            print("\n[base] --reevaluate-base set; will re-evaluate base row")
        elif _base_already_evaluated(project):
            print(
                "\n[base] sentinel artifact found; base model already in leaderboard "
                "from a previous iteration. Skipping base row this iteration "
                "(use --reevaluate-base to force)."
            )
            should_eval_base = False
            base_eval_skipped = True
        else:
            print("\n[base] no sentinel found; this is the first iteration — evaluating base row")

    models = []
    model_names = []
    display_names = []

    if should_eval_base:
        print(f"\nLoading base model: {base_model} (agent_llm={agent_llm})")
        base_row_name = f"base-{base_short}"
        base_wrapper = Tau2BaseModelWrapper(
            name=base_row_name,
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
        display_names.append(base_row_name)

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
                trained_row_name = f"trained-{suffix}-alias-{pinned_alias}"
                print(
                    f"Pinning evaluation to artifact alias :{pinned_alias} "
                    f"(collection latest is step {latest_step})"
                )
            elif pinned_step is not None:
                trained_row_name = f"trained-{suffix}-step{pinned_step}"
                print(
                    f"Pinning evaluation to checkpoint :step{pinned_step} "
                    f"(collection latest is step {latest_step})"
                )
            else:
                trained_row_name = f"trained-{suffix}-step{latest_step}-latest"
                print(f"Defaulting to :latest (step {latest_step})")
            trained_wrapper = Tau2BaseModelWrapper(
                name=trained_row_name,
                model=trained_model,
                model_name=trained_row_name,
                domain=domain,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                agent_llm_args=agent_llm_args,
                max_steps=max_steps,
                pinned_step=pinned_step,
                pinned_alias=pinned_alias,
            )
            models.append(trained_wrapper)
            model_names.append("trained")
            display_names.append(trained_row_name)
        except Exception as e:
            print(f"Could not load trained model {trained_name}: {e}")
    elif should_eval_trained and not trained_name:
        print("\nSkipping trained model (set leaderboard_trained_model_name in config, pass --trained-model-name, or run train first to create .last_trained_model)")

    if not models:
        print("\nNo models to evaluate.")
        if base_eval_skipped:
            print("(base sentinel exists; trained row needs a trained model name.)")
        run.finish()
        return

    base_completed = False
    for idx, (model, name, display_name) in enumerate(zip(models, model_names, display_names), 1):
        print(f"\nEvaluating {idx}/{len(models)}: {display_name}")
        await shared_evaluation.evaluate(model, __weave={"display_name": display_name})
        print(f"Completed: {display_name}")
        if name == "base":
            base_completed = True

    if base_completed:
        _log_base_sentinel(run, base_short, base_model)

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
            name=f"tau2-{domain}-leaderboard-validation",
            description=f"tau2-bench {domain}: task_reward, success, and pass^k ({num_trials} trials).",
            columns=lb_columns,
        )
        ref = weave.publish(leaderboard_spec)
        print(f"\nLeaderboard ref (idempotent): {ref}")
    except Exception as e:
        print(f"Failed to publish leaderboard spec: {e}")
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
        "--trained-model-step",
        type=int,
        default=None,
        help=(
            "Pin trained-model evaluation to LoRA checkpoint alias :step{N} "
            "instead of :latest. Use 0 to evaluate the freshly forked SFT init "
            "before any RL updates."
        ),
    )
    parser.add_argument(
        "--trained-model-alias",
        type=str,
        default=None,
        help=(
            "Pin trained-model evaluation directly to a W&B artifact alias "
            "(e.g. 'v1', 'latest'). Takes precedence over --trained-model-step. "
            "Use this when the desired artifact exists but lacks a step{N} alias."
        ),
    )
    parser.add_argument(
        "--reevaluate-base",
        action="store_true",
        help=(
            "Force re-evaluation of the base model row, even if a previous "
            "iteration already logged it."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(
        config_path=args.config,
        models_to_eval=args.models,
        trained_model_name=args.trained_model_name,
        trained_model_step=args.trained_model_step,
        trained_model_alias=args.trained_model_alias,
        reevaluate_base=args.reevaluate_base,
    ))

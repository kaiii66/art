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
    python create_leaderboard_shaped_reward.py --models rl --trained-model-name <name>
    python create_leaderboard_shaped_reward.py --models base sft rl

Model options: base, sft, rl, all
  - base : raw `base_model` via tau2 LLMAgent
  - sft  : trained collection pinned to .sft_endpoint_step (final SFT checkpoint)
  - rl   : trained collection pinned to .best_rl_step  (best val/reward RL step)
  - all  : base + sft + rl
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
    score_task_reward,
    score_success,
    score_error,
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
    # Backwards-compat: --models trained is deprecated; map to "rl".
    if "trained" in models_to_eval:
        models_to_eval = [m if m != "trained" else "rl" for m in models_to_eval]
        print("WARNING: --models trained is deprecated; use --models rl")
    eval_all = "all" in models_to_eval
    should_eval_base = eval_all or "base" in models_to_eval
    should_eval_sft = eval_all or "sft" in models_to_eval
    should_eval_rl = eval_all or "rl" in models_to_eval

    config = load_config(config_path)
    project = config["project"]
    domain = config["domain"]
    base_model = config["base_model"]
    # `config.get(key, default)` returns the existing value (even None), so an
    # explicit `agent_llm: null` in yaml (the training-time setting) skips the
    # default — and the base eval row would then crash on `model.inference_api_key`.
    # Treat None the same as missing here, falling back to W&B Inference for the
    # bare base model.
    agent_llm = config.get("agent_llm") or f"wandb/{base_model}"
    user_llm = config["user_llm"]
    # Prefer the dedicated leaderboard dataset (test split, clean holdout).
    # Fall back to validation_weave_dataset for configs that pre-date the split
    # (e.g. old pipeline_runs snapshots that don't have leaderboard_weave_dataset).
    eval_weave = (
        config.get("leaderboard_weave_dataset")
        or config.get("validation_weave_dataset")
        or f"tau2-{domain}-validation-scenarios"
    )
    trained_name = trained_model_name or config.get("leaderboard_trained_model_name")
    if not trained_name:
        last_model_file = Path(config_path).resolve().parent / ".last_trained_model"
        if last_model_file.exists():
            trained_name = last_model_file.read_text().strip()

    # Pin point for the "sft" row.
    #
    # Resolution order (highest precedence first):
    #   1. .best_sft_step  -- written by train_tau2_distill.run_distillation_sft
    #      every time val/reward improves. This is the checkpoint the
    #      leaderboard should evaluate; pinning to the LAST step (#2) on a
    #      noisy 4-task validation routinely picks an over-trained tail and
    #      makes SFT look 20-30 pp worse than its peak (see
    #      kwt/tau2-ART-autoresearch-telecom/sft-04271957-step28 ≈ 50% vs
    #      kwt/tau2-ART-distill-05151739 final-step ≈ 9%).
    #   2. .sft_endpoint_step -- the final SFT step. Used by older snapshots
    #      that pre-date best-step tracking. Also the fallback when an SFT run
    #      somehow finished without writing .best_sft_step.
    # If neither sidecar is present the sft row is skipped (e.g. ad-hoc /
    # non-pipeline runs, or RL-only resumes).
    sft_pinned_step = None
    snapshot_dir = Path(config_path).resolve().parent
    for fname in (".best_sft_step", ".sft_endpoint_step"):
        sft_step_file = snapshot_dir / fname
        if not sft_step_file.exists():
            continue
        try:
            sft_pinned_step = int(sft_step_file.read_text().strip())
            print(f"  [leaderboard] sft pin = step {sft_pinned_step} (from {fname})")
            break
        except (ValueError, OSError) as e:
            print(f"  [leaderboard] could not read {sft_step_file}: {e}")

    # Pin point for the "rl" row. Resolution order (highest precedence first):
    #   1. CLI --trained-model-alias / --trained-model-step
    #   2. config.leaderboard_trained_model_alias / leaderboard_trained_model_step
    #   3. .best_rl_step file next to the snapshot config (auto-written by
    #      train_tau2.py at the step with the highest val/reward). Falls back
    #      to :latest if none of the above resolve.
    rl_pinned_alias = trained_model_alias or config.get("leaderboard_trained_model_alias")
    rl_pinned_step = trained_model_step
    if rl_pinned_step is None:
        rl_pinned_step = config.get("leaderboard_trained_model_step")
    if rl_pinned_step is None and rl_pinned_alias is None:
        best_step_file = Path(config_path).resolve().parent / ".best_rl_step"
        if best_step_file.exists():
            try:
                rl_pinned_step = int(best_step_file.read_text().strip())
                print(
                    f"  [leaderboard] rl pin = step {rl_pinned_step} "
                    f"(from {best_step_file.name})"
                )
            except (ValueError, OSError) as e:
                print(f"  [leaderboard] could not read {best_step_file}: {e}")

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

    scorers = [score_success, score_task_reward, score_error]
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

    # Both "sft" and "rl" rows evaluate the SAME trained collection at
    # different LoRA checkpoint steps, so we register the TrainableModel once
    # and create one wrapper per requested role with its own pinned_step.
    if (should_eval_sft or should_eval_rl) and trained_name:
        try:
            print(f"\nLoading trained model: {trained_name}...")
            backend = ServerlessBackend()
            trained_model = art.TrainableModel(
                name=trained_name,
                project=project,
                base_model=base_model,
            )
            await trained_model.register(backend)

            # Route trained-model inference through the public W&B Inference
            # endpoint instead of the ART training backend. The training
            # backend (api.training.wandb.ai) is the registration default and
            # has been returning Cloudflare 524s under load; the public
            # inference endpoint serves the same `wandb-artifact:///...:stepN`
            # references and is the production-scaled path. Override is
            # configurable via leaderboard.inference_base_url.
            inference_base_override = lb_config.get(
                "inference_base_url", "https://api.inference.wandb.ai/v1"
            )
            if inference_base_override:
                prev_url = trained_model.inference_base_url
                trained_model.inference_base_url = inference_base_override
                print(
                    f"Overriding inference_base_url: {prev_url} -> "
                    f"{trained_model.inference_base_url}"
                )

            latest_step = await trained_model.get_step()
            print(f"Collection latest step: {latest_step}")

            if should_eval_sft:
                if sft_pinned_step is None:
                    print(
                        "Skipping sft row (no .sft_endpoint_step found). "
                        "This is normal for ad-hoc runs without the pipeline."
                    )
                else:
                    # Per-run isolation gives `trained_name` a `-rl-<suffix>` tag
                    # but SFT checkpoints live in the original (unsuffixed) W&B
                    # collection pointed to by `sft_source.name`.  If config sets
                    # sft_source, use it; otherwise fall back to trained_name
                    # (the legacy single-collection layout).
                    sft_src = config.get("sft_source", {})
                    sft_collection_name = sft_src.get("name") or trained_name
                    sft_collection_project = sft_src.get("project") or project
                    sft_collection_entity = sft_src.get("entity") or config.get("wandb_entity") or os.getenv("WANDB_ENTITY", "kwt")
                    if sft_collection_name != trained_name or sft_collection_project != project:
                        print(
                            f"Resolving sft row to original collection: "
                            f"{sft_collection_entity}/{sft_collection_project}/{sft_collection_name}"
                        )
                        sft_model = art.TrainableModel(
                            name=sft_collection_name,
                            project=sft_collection_project,
                            base_model=base_model,
                        )
                        await sft_model.register(backend)
                        sft_model.inference_base_url = trained_model.inference_base_url
                    else:
                        sft_model = trained_model
                    sft_label = f"SFT @ step {sft_pinned_step}"
                    sft_display = f"{base_model} ({sft_label})"
                    print(f"Pinning sft row to checkpoint :step{sft_pinned_step}")
                    sft_wrapper = Tau2BaseModelWrapper(
                        model=sft_model,
                        model_name=sft_display,
                        domain=domain,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        agent_llm_args=agent_llm_args,
                        max_steps=max_steps,
                        pinned_step=sft_pinned_step,
                        **shaped_kwargs,
                    )
                    models.append(sft_wrapper)
                    model_names.append("sft")
                    display_names.append(sft_display)

            if should_eval_rl:
                if rl_pinned_alias is not None:
                    rl_label = f"GRPO @ alias :{rl_pinned_alias}"
                    print(
                        f"Pinning rl row to artifact alias :{rl_pinned_alias} "
                        f"(collection latest is step {latest_step})"
                    )
                elif rl_pinned_step is not None:
                    rl_label = f"GRPO @ step {rl_pinned_step}"
                    if rl_pinned_step != latest_step:
                        rl_label += f" (best, latest={latest_step})"
                    print(
                        f"Pinning rl row to checkpoint :step{rl_pinned_step} "
                        f"(collection latest is step {latest_step})"
                    )
                else:
                    rl_label = f"GRPO @ step {latest_step}"
                    print(f"No rl pin found; rl row defaults to :latest (step {latest_step})")
                rl_display = f"{base_model} ({rl_label})"
                rl_wrapper = Tau2BaseModelWrapper(
                    model=trained_model,
                    model_name=rl_display,
                    domain=domain,
                    user_llm=user_llm,
                    user_llm_args=user_llm_args,
                    agent_llm_args=agent_llm_args,
                    max_steps=max_steps,
                    pinned_step=rl_pinned_step,
                    pinned_alias=rl_pinned_alias,
                    **shaped_kwargs,
                )
                models.append(rl_wrapper)
                model_names.append("rl")
                display_names.append(rl_display)
        except Exception as e:
            print(f"Could not load trained model {trained_name}: {e}")
    elif (should_eval_sft or should_eval_rl) and not trained_name:
        print(
            "\nSkipping sft / rl rows (no trained model name available). "
            "Set leaderboard_trained_model_name in config, pass --trained-model-name, "
            "or run train first to create .last_trained_model."
        )

    # Always include a frontier-baseline GPT-4.1-mini agent row when an
    # OpenAI key is in the environment. This gives every leaderboard a
    # constant "are we beating a strong off-the-shelf model?" reference
    # without requiring a separate eval pass. The wrapper uses tau2's
    # LLMAgent (no ART backend needed); litellm picks up OPENAI_API_KEY
    # from env. Skipped silently when the key is absent so airgapped runs
    # don't fail.
    if os.getenv("OPENAI_API_KEY"):
        # Pin to a dated snapshot so historical leaderboards stay
        # reproducible if OpenAI rolls a new minor version.
        gpt41_model_id = "openai/gpt-4.1-mini-2025-04-14"
        gpt41_display = f"{gpt41_model_id} (frontier-baseline)"
        gpt41_agent_args = {**agent_llm_args}
        gpt41_agent_args.setdefault("temperature", 0.0)
        gpt41_wrapper = Tau2BaseModelWrapper(
            name="gpt-4.1-mini",
            model=None,
            model_name=gpt41_display,
            domain=domain,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            agent_llm_args=gpt41_agent_args,
            max_steps=max_steps,
            agent_llm=gpt41_model_id,
            **shaped_kwargs,
        )
        models.append(gpt41_wrapper)
        model_names.append("gpt-4.1-mini")
        display_names.append(gpt41_display)
        print(f"\nAdding GPT-4.1-mini agent row (OPENAI_API_KEY detected): {gpt41_display}")
    else:
        print(
            "\nSkipping GPT-4.1-mini agent row (OPENAI_API_KEY not set). "
            "Add it to art/.env to enable the frontier-baseline row."
        )

    if os.getenv("GEMINI_API_KEY"):
        gemini_model_id = "gemini/gemini-3.5-flash"
        gemini_display = f"{gemini_model_id} (frontier-baseline)"
        gemini_agent_args = {**agent_llm_args}
        gemini_agent_args.setdefault("temperature", 0.0)
        gemini_wrapper = Tau2BaseModelWrapper(
            name="gemini-3.5-flash",
            model=None,
            model_name=gemini_display,
            domain=domain,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            agent_llm_args=gemini_agent_args,
            max_steps=max_steps,
            agent_llm=gemini_model_id,
            **shaped_kwargs,
        )
        models.append(gemini_wrapper)
        model_names.append("gemini-3.5-flash")
        display_names.append(gemini_display)
        print(f"\nAdding Gemini 3.5 Flash agent row (GEMINI_API_KEY detected): {gemini_display}")
    else:
        print(
            "\nSkipping Gemini 3.5 Flash row (GEMINI_API_KEY not set). "
            "Add it to art/.env to enable the Gemini frontier-baseline row."
        )

    if not models:
        print("\nNo models to evaluate.")
        run.finish()
        return

    completed_rows: list[str] = []
    failed_rows: list[tuple[str, str]] = []
    for idx, (model, name, display_name) in enumerate(zip(models, model_names, display_names), 1):
        print(f"\nEvaluating {idx}/{len(models)}: {display_name}")
        try:
            await shared_evaluation.evaluate(model, __weave={"display_name": display_name})
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"FAILED: {display_name} -> {err}")
            if "api.training.wandb.ai" in err or "524" in err:
                print(
                    "  Hint: this is a Cloudflare 524 from the ART training backend "
                    "(api.training.wandb.ai), not a bug in this script. The trained "
                    "LoRA checkpoint fetch timed out upstream. Check W&B status, then "
                    "re-run leaderboard alone with --resume <snapshot> --skip upload sft rl."
                )
            failed_rows.append((display_name, err))
            continue
        print(f"Completed: {display_name}")
        completed_rows.append(display_name)

    print(f"\nRow status: {len(completed_rows)} completed, {len(failed_rows)} failed")
    for d in completed_rows:
        print(f"  ok    : {d}")
    for d, err in failed_rows:
        print(f"  failed: {d} ({err})")

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
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_ref_uri,
                scorer_name="score_error",
                summary_metric_path="dropped.mean",
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
        choices=["base", "sft", "rl", "trained", "all"],
        default=["all"],
        help=(
            "Models to evaluate (default: all). 'sft' prefers .best_sft_step "
            "and falls back to .sft_endpoint_step. 'rl' uses .best_rl_step (or "
            ":latest fallback). 'trained' is a deprecated alias for 'rl'. The "
            "frontier-baseline 'gpt-4.1-mini' row is added automatically when "
            "OPENAI_API_KEY is set."
        ),
    )
    parser.add_argument("--trained-model-name", type=str, default=None, help="Trained model name (overrides config)")
    parser.add_argument(
        "--trained-model-step",
        type=int,
        default=None,
        help=(
            "Pin the rl row to LoRA checkpoint alias :step{N} instead of "
            ":latest. Default: auto-read .best_rl_step next to the config "
            "(written by train_tau2.py at the best val/reward step), else "
            "fall back to :latest. Does not affect the sft row."
        ),
    )
    parser.add_argument(
        "--trained-model-alias",
        type=str,
        default=None,
        help=(
            "Pin the rl row directly to a W&B artifact alias (e.g. 'v1', "
            "'latest'). Takes precedence over --trained-model-step and over "
            ".best_rl_step auto-discovery. Does not affect the sft row."
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

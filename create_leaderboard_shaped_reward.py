"""
Shaped-reward leaderboard for tau2-bench: compare base (Qwen) and trained (GRPO) models
on the held-out validation set with TWO columns:

  - success.mean       (binary, headline)    "did the model solve the task?"
  - task_reward.mean   (shaped, diagnostic)  continuous training-objective signal

No pass^k. The shaped column reports a different unit (continuous, possibly >1 before
step penalty) than the binary leaderboard in create_leaderboard.py, so this script
publishes its own Weave Evaluation / Leaderboard objects (suffixed with "-shaped")
to keep the two histories cleanly separated.

Idempotency / autoresearch design:
  This script is the LAST stage in run_pipeline.py and is invoked once per
  autoresearch iteration. We want a SINGLE leaderboard that grows with one
  bundle of new rows per iteration:

    iter 1 :  3 rows  (base-<model>, sft-<tag1>-stepN, rl-<tag1>-stepM)
    iter 2 :  2 rows  (sft-<tag2>-stepN, rl-<tag2>-stepM)
    iter 3 :  2 rows  (sft-<tag3>-stepN, rl-<tag3>-stepM)
    ...

  Mechanics:
   1. weave.publish(Dataset|Evaluation|Leaderboard) is content-hash deduped, so
      republishing the same scaffolding on every invocation is a no-op (just
      retrieves the existing object). No CLI flag needed.
   2. The base model row is identical across iterations, so we use a W&B
      sentinel artifact (`tau2-leaderboard-shaped-base-evaluated:latest`) to
      record that base has already been evaluated. If the sentinel exists,
      we skip the base row this iteration (still emits sft + rl).
   3. Each Tau2BaseModelWrapper gets a stable, descriptive `name` field so its
      row in the Weave leaderboard is trivially traceable back to a W&B run:
        - base : "base-<base_model_short>"           (e.g. base-Qwen3-30B-A3B-Instruct-2507)
        - sft  : "sft-<iter_suffix>-step<N>"          (e.g. sft-04241342-step6)
        - rl   : "rl-<iter_suffix>-step<N>" / "-alias-<X>" / "-latest"

Usage:
    python create_leaderboard_shaped_reward.py
    python create_leaderboard_shaped_reward.py --models all
    python create_leaderboard_shaped_reward.py --models rl --trained-model-name <name>
    python create_leaderboard_shaped_reward.py --models base sft rl
    python create_leaderboard_shaped_reward.py --reevaluate-base    # force base re-eval

Model options: base, sft, rl, all
  - base : raw `base_model` via tau2 LLMAgent (auto-skipped if sentinel exists)
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
)


BASE_SENTINEL_ARTIFACT = "tau2-leaderboard-shaped-base-evaluated"


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _short(model_id: str) -> str:
    """Filesystem/URL-safe short tag for a HF model id (`Qwen/Qwen3-...` -> `Qwen3-...`)."""
    return model_id.split("/")[-1]


def _base_already_evaluated(project: str) -> bool:
    """Probe the project for the base-evaluated sentinel artifact.

    Returns True iff a previous iteration has logged
    `<entity>/<project>/tau2-leaderboard-shaped-base-evaluated:latest`. This is
    how we make the base row appear once and only once in the leaderboard,
    regardless of how many iterations run_pipeline.py is invoked.
    """
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
    """Persist a tiny marker artifact to indicate the base row is now in the eval."""
    try:
        art_obj = wandb.Artifact(
            name=BASE_SENTINEL_ARTIFACT,
            type="leaderboard-marker",
            description=(
                "Sentinel: the base model has been evaluated and added to the "
                "shaped-reward Weave leaderboard. Subsequent autoresearch "
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
    base_short = _short(base_model)
    agent_llm = config.get("agent_llm", f"wandb/{base_model}")
    user_llm = config["user_llm"]
    eval_weave = config.get("validation_weave_dataset") or f"tau2-{domain}-validation-scenarios"

    # Iteration group: written by run_pipeline.make_snapshot. Falls back to
    # "manual" for ad-hoc invocations outside the pipeline.
    group = config.get("group") or "manual"
    suffix = group.removeprefix("pipeline-") if group.startswith("pipeline-") else group

    trained_name = trained_model_name or config.get("leaderboard_trained_model_name")
    if not trained_name:
        last_model_file = Path(config_path).resolve().parent / ".last_trained_model"
        if last_model_file.exists():
            trained_name = last_model_file.read_text().strip()

    # ── On-prem mode detection ──
    # The on-prem backend (run_pipeline.py --backend onprem) writes
    # `.sft_lora_artifact_uri` and `.rl_lora_artifact_uri` next to the snapshot
    # config. These point at W&B Inference type=lora artifacts (PEFT adapters
    # over the base model). When present, we evaluate them via litellm's
    # `wandb/<artifact-uri>` path instead of going through the ART
    # TrainableModel infrastructure.
    snapshot_dir = Path(config_path).resolve().parent
    onprem_sft_uri = None
    onprem_rl_uri = None
    sft_uri_file = snapshot_dir / ".sft_lora_artifact_uri"
    rl_uri_file = snapshot_dir / ".rl_lora_artifact_uri"
    if sft_uri_file.exists():
        onprem_sft_uri = sft_uri_file.read_text().strip() or None
    if rl_uri_file.exists():
        onprem_rl_uri = rl_uri_file.read_text().strip() or None
    onprem_mode = bool(onprem_sft_uri or onprem_rl_uri)
    if onprem_mode:
        print(
            f"  [leaderboard] on-prem mode: "
            f"sft_uri={onprem_sft_uri or '(none)'} "
            f"rl_uri={onprem_rl_uri or '(none)'}"
        )

    # Pin point for the "sft" row. Resolution order (highest precedence first):
    #   1. .best_sft_step  -- written by train_tau2_distill.py when chunk
    #      validation on dev produces a usable signal. This is the
    #      authoritative leaderboard pin: the step that actually achieved the
    #      highest val/success on dev, regardless of where SFT subsequently
    #      stopped (with non-destructive early-stop, :latest may sit
    #      `sft_early_stop_patience` chunks past best_step).
    #   2. .sft_endpoint_step  -- back-compat fallback for old snapshots that
    #      pre-date .best_sft_step. Written by train_tau2.py from
    #      `await model.get_step()` at RL start, so it equals the SFT
    #      collection's :latest at the moment RL began.
    # If neither file exists the sft row is skipped (ad-hoc / non-pipeline runs).
    sft_pinned_step = None
    snapshot_dir_for_sft = Path(config_path).resolve().parent
    best_sft_file = snapshot_dir_for_sft / ".best_sft_step"
    endpoint_file = snapshot_dir_for_sft / ".sft_endpoint_step"
    sft_step_file: Path | None = None
    sft_step_source: str | None = None
    if best_sft_file.exists():
        sft_step_file = best_sft_file
        sft_step_source = "best (dev val)"
    elif endpoint_file.exists():
        sft_step_file = endpoint_file
        sft_step_source = "endpoint (collection :latest at RL start)"
    if sft_step_file is not None:
        try:
            sft_pinned_step = int(sft_step_file.read_text().strip())
            print(
                f"  [leaderboard] sft pin = step {sft_pinned_step} "
                f"(from {sft_step_file.name}, source={sft_step_source})"
            )
        except (ValueError, OSError) as e:
            print(f"  [leaderboard] could not read {sft_step_file}: {e}")

    # Pin point for the NEW "sft-endpoint" row: the SFT collection's :latest
    # at the moment RL began, i.e., where RL actually started training from.
    # When SFT early-stops, this is `best_sft_step + slack` (slack <= patience),
    # which differs from `.best_sft_step` (the headline sft row) and is the
    # correct baseline for measuring "delta improvement RL made". Skipped
    # downstream if equal to sft_pinned_step (no slack, would be a duplicate).
    sft_endpoint_step: int | None = None
    if endpoint_file.exists():
        try:
            sft_endpoint_step = int(endpoint_file.read_text().strip())
        except (ValueError, OSError) as e:
            print(f"  [leaderboard] could not read {endpoint_file}: {e}")

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

    weave.init(project)

    print("\nLoading validation dataset...")
    try:
        original = weave.ref(eval_weave).get()
    except Exception as e:
        raise RuntimeError(f"Could not load Weave dataset {eval_weave}: {e}") from e
    print(f"Loaded {len(original.rows)} rows from {eval_weave}")

    # The four Weave scaffold objects (Dataset, Evaluation, Leaderboard cols,
    # Leaderboard) are content-hash deduped by weave.publish, so it's safe
    # (and idempotent) to (re)publish them every iteration. The first
    # iteration creates them; every subsequent iteration just reuses the
    # existing version.
    leaderboard_dataset_name = f"tau2-{domain}-validation-scenarios-leaderboard-shaped"
    dataset = weave.Dataset(name=leaderboard_dataset_name, rows=original.rows)
    weave.publish(dataset)

    scorers = [score_success, score_task_reward]
    eval_name = f"tau2-{domain}-evaluation-leaderboard-shaped"
    shared_evaluation = weave.Evaluation(
        name=eval_name,
        dataset=dataset,
        scorers=scorers,
        trials=num_trials,
    )
    weave.publish(shared_evaluation)
    print(
        f"Using evaluation '{eval_name}' (trials={num_trials}, "
        f"shaped_reward={shaped_kwargs['use_shaped_reward']})"
    )

    run = wandb.init(
        project=project,
        group=group,
        name=f"leaderboard-{suffix}",
        config=config,
        job_type="leaderboard",
    )

    # Decide whether to actually evaluate base. We evaluate only if (a) the
    # caller asked for it AND (b) it hasn't been evaluated before in this
    # project (sentinel artifact missing) OR they asked for --reevaluate-base.
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
            **shaped_kwargs,
        )
        models.append(base_wrapper)
        model_names.append("base")
        display_names.append(base_row_name)

    # ── On-prem leaderboard rows (W&B Inference type=lora artifacts) ──
    # Each URI is a fully-resolved `wandb-artifact:///team/proj/name:version`
    # that W&B Inference can serve directly. We wire it through tau2's
    # LLMAgent path (agent_llm=...) by prefixing with `wandb/` so litellm
    # routes the call to api.inference.wandb.ai.
    if onprem_mode:
        if should_eval_sft and onprem_sft_uri:
            sft_row_name = f"sft-{suffix}-onprem"
            print(f"\n[onprem] sft row -> {onprem_sft_uri}")
            sft_wrapper = Tau2BaseModelWrapper(
                name=sft_row_name,
                model=None,
                model_name=onprem_sft_uri,
                domain=domain,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                agent_llm_args=agent_llm_args,
                max_steps=max_steps,
                agent_llm=f"wandb/{onprem_sft_uri}",
                **shaped_kwargs,
            )
            models.append(sft_wrapper)
            model_names.append("sft")
            display_names.append(sft_row_name)
        elif should_eval_sft:
            print("\n[onprem] sft row requested but .sft_lora_artifact_uri is missing; skipping.")

        if should_eval_rl and onprem_rl_uri:
            rl_row_name = f"rl-{suffix}-onprem"
            print(f"\n[onprem] rl row -> {onprem_rl_uri}")
            rl_wrapper = Tau2BaseModelWrapper(
                name=rl_row_name,
                model=None,
                model_name=onprem_rl_uri,
                domain=domain,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
                agent_llm_args=agent_llm_args,
                max_steps=max_steps,
                agent_llm=f"wandb/{onprem_rl_uri}",
                **shaped_kwargs,
            )
            models.append(rl_wrapper)
            model_names.append("rl")
            display_names.append(rl_row_name)
        elif should_eval_rl:
            print("\n[onprem] rl row requested but .rl_lora_artifact_uri is missing; skipping.")

    # ── Serverless leaderboard rows (ART TrainableModel) ──
    # Both "sft" and "rl" rows evaluate the SAME trained collection at
    # different LoRA checkpoint steps, so we register the TrainableModel once
    # and create one wrapper per requested role with its own pinned_step.
    if not onprem_mode and (should_eval_sft or should_eval_rl) and trained_name:
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
                    # Stable, descriptive name: ties this leaderboard row back
                    # to its W&B run group (`pipeline-<suffix>`) and to its
                    # exact LoRA checkpoint step. Identical naming convention
                    # used in `name=` on the Weave Model and `display_name=`
                    # on the eval call so the two surfaces agree.
                    sft_row_name = f"sft-{suffix}-step{sft_pinned_step}"
                    print(f"Pinning sft row to checkpoint :step{sft_pinned_step}")
                    sft_wrapper = Tau2BaseModelWrapper(
                        name=sft_row_name,
                        model=trained_model,
                        model_name=sft_row_name,
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
                    display_names.append(sft_row_name)

                # Additional row: where RL actually started training from.
                # When SFT early-stops, .sft_endpoint_step lags .best_sft_step
                # by up to `sft_early_stop_patience` chunks. Evaluating this
                # checkpoint gives the true baseline for `rl - sft-endpoint`,
                # i.e., the value RL added on top of its actual starting point.
                # Skipped when redundant (endpoint == best, no slack) or when
                # .sft_endpoint_step is unavailable (ad-hoc runs without RL).
                if (
                    sft_endpoint_step is not None
                    and sft_endpoint_step != sft_pinned_step
                ):
                    sft_endpoint_row_name = (
                        f"sft-endpoint-{suffix}-step{sft_endpoint_step}"
                    )
                    print(
                        f"Adding sft-endpoint row pinned to :step{sft_endpoint_step} "
                        f"(where RL started; baseline for rl delta)"
                    )
                    sft_endpoint_wrapper = Tau2BaseModelWrapper(
                        name=sft_endpoint_row_name,
                        model=trained_model,
                        model_name=sft_endpoint_row_name,
                        domain=domain,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                        agent_llm_args=agent_llm_args,
                        max_steps=max_steps,
                        pinned_step=sft_endpoint_step,
                        **shaped_kwargs,
                    )
                    models.append(sft_endpoint_wrapper)
                    model_names.append("sft-endpoint")
                    display_names.append(sft_endpoint_row_name)

            if should_eval_rl:
                if rl_pinned_alias is not None:
                    rl_row_name = f"rl-{suffix}-alias-{rl_pinned_alias}"
                    print(
                        f"Pinning rl row to artifact alias :{rl_pinned_alias} "
                        f"(collection latest is step {latest_step})"
                    )
                elif rl_pinned_step is not None:
                    rl_row_name = f"rl-{suffix}-step{rl_pinned_step}"
                    if rl_pinned_step != latest_step:
                        print(
                            f"Pinning rl row to checkpoint :step{rl_pinned_step} "
                            f"(best, collection latest is step {latest_step})"
                        )
                    else:
                        print(f"Pinning rl row to checkpoint :step{rl_pinned_step}")
                else:
                    rl_row_name = f"rl-{suffix}-step{latest_step}-latest"
                    print(f"No rl pin found; rl row defaults to :latest (step {latest_step})")
                rl_wrapper = Tau2BaseModelWrapper(
                    name=rl_row_name,
                    model=trained_model,
                    model_name=rl_row_name,
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
                display_names.append(rl_row_name)
        except Exception as e:
            print(f"Could not load trained model {trained_name}: {e}")
    elif not onprem_mode and (should_eval_sft or should_eval_rl) and not trained_name:
        print(
            "\nSkipping sft / rl rows (no trained model name available). "
            "Set leaderboard_trained_model_name in config, pass --trained-model-name, "
            "or run train first to create .last_trained_model. "
            "(For on-prem mode, ensure .sft_lora_artifact_uri / .rl_lora_artifact_uri exist.)"
        )

    if not models:
        print("\nNo models to evaluate (base may have been auto-skipped).")
        if base_eval_skipped:
            print("(base sentinel exists; sft/rl rows still need a trained model.)")
        run.finish()
        return

    completed_rows: list[str] = []
    failed_rows: list[tuple[str, str]] = []
    base_completed = False
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
        if name == "base":
            base_completed = True

    # Drop the sentinel ONLY after a successful base eval, so a base failure
    # doesn't lock future iterations out of retrying it.
    if base_completed:
        _log_base_sentinel(run, base_short, base_model)

    print(f"\nRow status: {len(completed_rows)} completed, {len(failed_rows)} failed")
    for d in completed_rows:
        print(f"  ok    : {d}")
    for d, err in failed_rows:
        print(f"  failed: {d} ({err})")

    # Always (re)publish the leaderboard. Weave content-hash dedup means this
    # is a no-op when the spec hasn't changed; it ensures the leaderboard
    # exists on first iteration without needing a special CLI flag.
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
        ref = weave.publish(leaderboard_spec)
        print(f"\nLeaderboard ref (idempotent): {ref}")
    except Exception as e:
        print(f"Failed to publish leaderboard spec: {e}")
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
            "Models to evaluate (default: all). 'sft' uses .sft_endpoint_step, "
            "'rl' uses .best_rl_step (or :latest fallback). 'trained' is a "
            "deprecated alias for 'rl'. The 'base' row is auto-skipped after "
            "the first successful evaluation in this project (use --reevaluate-base "
            "to force)."
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
        "--reevaluate-base",
        action="store_true",
        help=(
            "Force re-evaluation of the base model row, even if a previous "
            "iteration already logged it. Useful when changing user_llm or "
            "agent_llm_args in a way that should reflect on the base baseline."
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

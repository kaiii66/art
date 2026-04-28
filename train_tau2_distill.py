"""
ART SFT-distillation training script for tau2-bench.

Pipeline:
  Phase A. Run teacher LLM (MiniMax-M2.5) as the tau2 agent on training tasks
           and capture each rollout as an SFT-ready Trajectory.
  Phase B. Optionally filter for successful teacher rollouts, then chunked
           supervised fine-tuning of the student (Qwen3-30B-A3B-Instruct-2507)
           via art's train_sft. Between SFT chunks, run student validation
           rollouts and log success/reward to W&B (same metric names as
           train_tau2.py).

Usage:
  python train_tau2_distill.py --config train_distill_config.yaml
"""
import argparse
import asyncio
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
from art.utils.sft import create_sft_dataset_iterator

from tau2.run import get_tasks
from tau2_art_helpers import (
    Tau2TaskScenario,
    tau2_rollout,
    tau2_teacher_rollout,
)
from train_tau2 import (
    load_training_tasks_from_artifact,
    load_validation_tasks,
)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# Phase A: teacher rollouts
# ─────────────────────────────────────────────────────────────────────

async def _run_teacher_pass(tasks, rollouts_per_task, config, label):
    """Run `rollouts_per_task` teacher rollouts on each task in `tasks`.

    Returns the list of Trajectories that completed (regardless of success).
    Failed/exceptioned rollouts are dropped with a warning.
    """
    domain = config["domain"]
    teacher_llm = config["teacher_llm"]
    teacher_llm_args = config.get("teacher_llm_args", {"temperature": 0.7})
    user_llm = config["user_llm"]
    user_llm_args = config.get("user_llm_args", {"temperature": 1.0})
    max_steps = config.get("max_orchestrator_steps", 30)
    concurrency = config.get("teacher_concurrency", 4)

    sem = asyncio.Semaphore(concurrency)

    async def _one(task, trial_idx):
        scenario = Tau2TaskScenario(step=0, task_id=task.id, domain=domain)
        async with sem:
            try:
                return await tau2_teacher_rollout(
                    task_scenario=scenario,
                    teacher_llm=teacher_llm,
                    teacher_llm_args=teacher_llm_args,
                    user_llm=user_llm,
                    user_llm_args=user_llm_args,
                    max_steps=max_steps,
                )
            except Exception as e:
                print(f"  [{label}] task={task.id} trial={trial_idx} FAILED: {e}")
                return None

    coros = [
        _one(task, i)
        for task in tasks
        for i in range(rollouts_per_task)
    ]

    print(
        f"  [{label}] {len(coros)} rollouts ({len(tasks)} tasks × {rollouts_per_task}) "
        f"at concurrency={concurrency}..."
    )

    results = await asyncio.gather(*coros)
    return [t for t in results if t is not None]


async def generate_teacher_trajectories(training_tasks, config):
    """Run the teacher LLM on training tasks and return Trajectories.

    Two strategies controlled by `prefilter_tasks`:
      - prefilter_tasks=False (legacy): run `teacher_rollouts_per_task` on EVERY
        task regardless of whether the teacher can solve it. Wastes compute on
        tasks the teacher consistently fails.
      - prefilter_tasks=True (recommended): first run 1 probe rollout per task
        to identify which tasks the teacher can solve, then run
        `teacher_rollouts_per_task - 1` ADDITIONAL rollouts on solvable tasks
        only. Probe successes are kept, so total rollouts on a solvable task
        equals `teacher_rollouts_per_task`.
    """
    rollouts_per_task = config.get("teacher_rollouts_per_task", 1)
    prefilter = config.get("prefilter_tasks", False)

    if not prefilter:
        print(f"\nPhase A: generating teacher rollouts on all {len(training_tasks)} tasks "
              f"(prefilter_tasks=False)")
        all_trajs = await _run_teacher_pass(
            training_tasks, rollouts_per_task, config, label="teacher",
        )
    else:
        print(f"\nPhase A: prefilter probe — 1 rollout per task on "
              f"{len(training_tasks)} tasks to find solvable ones")
        probe_trajs = await _run_teacher_pass(
            training_tasks, rollouts_per_task=1, config=config, label="probe",
        )
        solvable_ids = {
            t.metadata["task_id"] for t in probe_trajs
            if t.metrics.get("success", 0.0) >= 1.0
        }
        solvable_tasks = [t for t in training_tasks if t.id in solvable_ids]
        probe_success_rate = (
            len(solvable_tasks) / len(training_tasks) if training_tasks else 0.0
        )
        print(f"  probe: {len(solvable_tasks)} / {len(training_tasks)} tasks solvable "
              f"({probe_success_rate:.1%})")
        wandb.log({
            "teacher/probe_num_tasks": len(training_tasks),
            "teacher/probe_solvable_tasks": len(solvable_tasks),
            "teacher/probe_success_rate": probe_success_rate,
        })

        additional = max(0, rollouts_per_task - 1)
        if additional > 0 and solvable_tasks:
            print(f"\nPhase A: collecting {additional} additional rollouts × "
                  f"{len(solvable_tasks)} solvable tasks")
            additional_trajs = await _run_teacher_pass(
                solvable_tasks, additional, config, label="collect",
            )
            all_trajs = probe_trajs + additional_trajs
        else:
            all_trajs = probe_trajs

    if not all_trajs:
        return []

    success_rate = sum(t.metrics.get("success", 0.0) for t in all_trajs) / len(all_trajs)
    avg_reward = sum(t.reward for t in all_trajs) / len(all_trajs)
    successful_count = sum(1 for t in all_trajs if t.metrics.get("success", 0.0) >= 1.0)
    print(
        f"\n  teacher TOTAL: {len(all_trajs)} trajectories  "
        f"success={success_rate:.1%} ({successful_count}/{len(all_trajs)})  "
        f"avg_reward={avg_reward:.3f}"
    )
    wandb.log({
        "teacher/num_trajectories": len(all_trajs),
        "teacher/success_rate": success_rate,
        "teacher/avg_reward": avg_reward,
    })
    return all_trajs


# ─────────────────────────────────────────────────────────────────────
# Validation: run STUDENT model on validation tasks (mirrors train_tau2.py)
# ─────────────────────────────────────────────────────────────────────

async def run_student_validation(model, validation_tasks, config, step):
    """Run the student (ART model) on validation tasks; log to W&B.

    Returns the mean val/success across the validation tasks (or None when
    validation could not run / produced no trajectories). The caller in
    `run_distillation_sft` uses this to track the best chunk so the SFT
    collection can be trimmed to the best-validating checkpoint at the end.
    """
    if not validation_tasks:
        return None
    domain = config["domain"]
    user_llm = config["user_llm"]
    user_llm_args = config.get("user_llm_args", {"temperature": 1.0})
    max_steps = config.get("max_orchestrator_steps", 30)

    print(f"\n  [validation] running student on {len(validation_tasks)} tasks...")
    val_groups = []
    for task in validation_tasks:
        scenario = Tau2TaskScenario(step=step, task_id=task.id, domain=domain)
        val_groups.append(
            art.TrajectoryGroup([
                tau2_rollout(
                    model,
                    scenario,
                    user_llm=user_llm,
                    user_llm_args=user_llm_args,
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
    if not val_trajs:
        return None
    val_reward = sum(t.reward for t in val_trajs) / len(val_trajs)
    val_success = sum(t.metrics.get("success", 0.0) for t in val_trajs) / len(val_trajs)
    val_tokens = sum(t.metadata.get("completion_tokens", 0) for t in val_trajs)
    wandb.log({
        "val/reward": val_reward,
        "val/success": val_success,
        "val/completion_tokens": val_tokens,
        "val/num_tasks": len(val_trajs),
        "val/sft_step": step,
    })
    print(f"  [validation] reward={val_reward:.3f}  success={val_success:.1%}")
    return val_success


# ─────────────────────────────────────────────────────────────────────
# Phase B: chunked SFT with periodic validation
# ─────────────────────────────────────────────────────────────────────

async def run_distillation_sft(model, trajectories, validation_tasks, config):
    """Run chunked SFT distillation, then trim the collection to the best chunk.

    Returns the ART step number (in the model's W&B collection) of the best
    chunk by val/success, or None if no validation produced a usable signal.
    Caller (`main`) records this to `.best_sft_step` for diagnostics.

    After all chunks finish, the SFT collection is trimmed in-place so that
    only the best-validating checkpoint remains. This makes
    `model.get_step()` return `best_step` for downstream consumers
    (`train_tau2.py` for the next RL stage and `create_leaderboard*.py`),
    so RL trains from and the leaderboard evaluates the same checkpoint
    instead of whatever the final chunk happened to be.
    """
    epochs = config.get("sft_epochs", 2)
    batch_size = config.get("sft_batch_size", 2)
    peak_lr = float(config.get("sft_peak_lr", 2e-4))
    warmup_ratio = float(config.get("sft_warmup_ratio", 0.1))
    schedule_type = config.get("sft_schedule_type", "cosine")
    chunk_size = config.get("sft_chunk_size", 2)
    val_every = config.get("validation_every_n_chunks", 1)

    print(f"\n{'='*60}")
    print(f"Phase B: SFT distillation")
    print(f"Trajectories     : {len(trajectories)}")
    print(f"Epochs           : {epochs}")
    print(f"Batch size       : {batch_size}")
    print(f"Peak LR          : {peak_lr}")
    print(f"Schedule         : {schedule_type} (warmup_ratio={warmup_ratio})")
    print(f"Chunk size       : {chunk_size} batches per train_sft call")
    print(f"Validation cadence: every {val_every} chunk(s)")
    print(f"{'='*60}")

    # Run baseline validation BEFORE any SFT so we can see baseline-vs-trained
    # progress. Use step=-1 so it doesn't collide on the same x-axis point as
    # the first post-SFT validation (which logs at chunk.step=0).
    await run_student_validation(model, validation_tasks, config, step=-1)

    # Materialize the iterator so we know the total chunk count up front,
    # which lets us guarantee a final validation after the last chunk
    # regardless of validation_every_n_chunks.
    all_chunks = list(create_sft_dataset_iterator(
        trajectories=trajectories,
        chunk_size=chunk_size,
        epochs=epochs,
        batch_size=batch_size,
        peak_lr=peak_lr,
        schedule_type=schedule_type,
        warmup_ratio=warmup_ratio,
        shuffle=True,
        seed=config.get("random_seed", 42),
    ))
    total_chunks = len(all_chunks)
    print(f"Total chunks     : {total_chunks}")

    # Best-chunk tracking. Strict `>` matches train_tau2.py's .best_rl_step
    # convention: when multiple chunks tie on val/success, the EARLIEST
    # winning chunk is preferred (less training -> less overfitting risk
    # and a smaller divergence from the SFT seed for downstream RL).
    best_val_success = float("-inf")
    best_step: int | None = None

    chunks_done = 0
    last_validated_step = None
    for i, chunk in enumerate(all_chunks):
        is_last_chunk = i == total_chunks - 1
        print(
            f"\n--- SFT chunk {i+1}/{total_chunks}: step={chunk.step} "
            f"epoch={chunk.epoch} epoch_step={chunk.epoch_step} "
            f"trajs={len(chunk.trajectories)} ---"
        )
        try:
            await model.train_sft(chunk.trajectories, chunk.config)
        except Exception as e:
            print(f"  train_sft FAILED: {e}. Skipping chunk.")
            continue

        chunks_done += 1
        wandb.log({
            "sft/chunk_idx": chunks_done,
            "sft/step": chunk.step,
            "sft/epoch": chunk.epoch,
            "sft/epoch_step": chunk.epoch_step,
        })

        # Validate on the configured cadence OR always after the final chunk
        # so we never miss the most important data point: the final model.
        should_validate = (chunks_done % val_every == 0) or is_last_chunk
        if should_validate and chunk.step != last_validated_step:
            val_success = await run_student_validation(
                model, validation_tasks, config, step=chunk.step,
            )
            last_validated_step = chunk.step
            if val_success is not None and val_success > best_val_success:
                # Use ART's own step counter (not chunk.step, which is the
                # internal sft batch counter) since the leaderboard pin and
                # train_tau2.py's continue-from logic both key off
                # model.get_step().
                try:
                    candidate_step = await model.get_step()
                except Exception as e:
                    print(f"  [best-chunk] could not read model.get_step(): {e}")
                    continue
                best_val_success = val_success
                best_step = candidate_step
                print(
                    f"  [best-chunk] new best val/success={best_val_success:.3f} "
                    f"@ ART step {best_step}"
                )
                if wandb.run is not None:
                    wandb.run.summary["sft/best_step"] = best_step
                    wandb.run.summary["sft/best_val_success"] = best_val_success
                wandb.log({
                    "sft/best_step": best_step,
                    "sft/best_val_success": best_val_success,
                })

    print(f"\nSFT distillation complete. {chunks_done}/{total_chunks} chunks finished.")

    # Trim the W&B collection so that only `best_step` remains. This makes
    # the downstream `model.get_step()` calls in train_tau2.py and the
    # leaderboard's `:step{N}` pin both resolve to the best checkpoint.
    if best_step is None:
        print(
            "  [best-chunk] no validation produced a usable val/success; "
            "skipping collection trim. Downstream RL will start from the "
            "final SFT chunk (legacy behavior)."
        )
        return None

    try:
        current_latest = await model.get_step()
    except Exception as e:
        print(f"  [best-chunk] could not read model.get_step() before trim: {e}")
        return best_step

    if current_latest == best_step:
        print(
            f"  [best-chunk] best step ({best_step}) is already the collection's "
            f"latest; no trim needed."
        )
        return best_step

    backend = model.backend()
    delete_fn = getattr(backend, "_delete_checkpoint_files", None)
    if delete_fn is None:
        print(
            "  [best-chunk] backend does not expose _delete_checkpoint_files; "
            f"cannot trim. RL will use latest step={current_latest}, leaderboard "
            f"will pin to step={current_latest}, but best step was {best_step}."
        )
        return best_step

    print(
        f"  [best-chunk] trimming SFT collection: keeping only :step{best_step} "
        f"(was :step{current_latest})"
    )
    try:
        await delete_fn(model, [best_step])
    except Exception as e:
        print(
            f"  [best-chunk] trim FAILED ({type(e).__name__}: {e}). RL will use "
            f"latest step={current_latest}; leaderboard will follow. Best step "
            f"was {best_step} -- see .best_sft_step for the intended pin."
        )
        return best_step

    try:
        new_latest = await model.get_step()
    except Exception as e:
        print(f"  [best-chunk] could not verify post-trim model.get_step(): {e}")
        return best_step
    if new_latest != best_step:
        print(
            f"  [best-chunk] WARNING trim ran but new latest is {new_latest}, "
            f"not best_step={best_step}. Downstream RL/leaderboard will use "
            f"{new_latest}."
        )
    else:
        print(
            f"  [best-chunk] trimmed collection: latest is now :step{best_step} "
            f"(val/success={best_val_success:.3f})"
        )
    return best_step


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
    if args.num_validation_tasks is not None:
        config["num_validation_tasks"] = args.num_validation_tasks
    if args.sft_epochs is not None:
        config["sft_epochs"] = args.sft_epochs
    if args.teacher_rollouts_per_task is not None:
        config["teacher_rollouts_per_task"] = args.teacher_rollouts_per_task
    if args.sft_peak_lr is not None:
        config["sft_peak_lr"] = args.sft_peak_lr
    if args.prefilter_tasks is not None:
        config["prefilter_tasks"] = args.prefilter_tasks

    random.seed(config.get("random_seed", 42))

    # ── W&B ──
    # The pipeline writes `group: pipeline-<MMDDHHMM>` into the snapshot
    # config so every wandb.run from this iteration (upload, sft, rl,
    # leaderboard) collapses under one expandable group in the W&B UI.
    # Ad-hoc invocations without the pipeline default to "manual".
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    group = config.get("group") or "manual"
    suffix = group.removeprefix("pipeline-") if group.startswith("pipeline-") else group
    run_name = (
        f"sft-{suffix}-r{config.get('teacher_rollouts_per_task', 1)}"
        f"-ep{config.get('sft_epochs', 2)}"
    )
    wandb.init(
        project=config["project"],
        group=group,
        name=run_name,
        config=config,
        job_type=config.get("wandb_job_type", "distill"),
    )

    # ── ART model ──
    # Tie the SFT collection name to the iteration suffix (same string used in
    # the snapshot dir name, W&B group, and leaderboard row labels) so it's
    # trivially greppable across all four surfaces.
    model_name = f"{config['model_name']}-{suffix}"
    model = art.TrainableModel(
        name=model_name,
        project=config["project"],
        base_model=config["base_model"],
    )
    backend = ServerlessBackend()
    await model.register(backend)

    # So create_leaderboard.py can pick this up automatically.
    (config_path.resolve().parent / ".last_trained_model").write_text(model_name)

    # ── Tasks ──
    # SFT always uses the full train split (`num_tasks=None`). Use --num-tasks
    # to bound the size on the CLI for ad-hoc smoke tests.
    domain = config["domain"]
    num_validation = config.get("num_validation_tasks")
    cli_num_tasks = getattr(args, "num_tasks", None)

    training_tasks = load_training_tasks_from_artifact(config, num_tasks=cli_num_tasks)
    if training_tasks is None:
        training_tasks = get_tasks(
            task_set_name=domain,
            task_split_name="train",
            num_tasks=cli_num_tasks,
        )
        print(f"Loaded {len(training_tasks)} training tasks from {domain}/train")
    else:
        print(
            f"Loaded {len(training_tasks)} training tasks from W&B artifact "
            f"({config['training_dataset_artifact']})"
        )

    validation_tasks = load_validation_tasks(config)
    if num_validation is not None:
        validation_tasks = validation_tasks[:num_validation]
    print(f"Loaded {len(validation_tasks)} validation tasks")

    # ── Phase A: teacher rollouts ──
    teacher_trajectories = await generate_teacher_trajectories(training_tasks, config)
    if not teacher_trajectories:
        print("No teacher trajectories produced. Aborting.")
        wandb.finish(exit_code=1)
        return

    if config.get("filter_successful_only", True):
        kept = [t for t in teacher_trajectories if t.metrics.get("success", 0.0) >= 1.0]
        dropped = len(teacher_trajectories) - len(kept)
        print(f"  filter_successful_only: kept {len(kept)} / {len(teacher_trajectories)} "
              f"(dropped {dropped} failed)")
        wandb.log({
            "teacher/kept_trajectories": len(kept),
            "teacher/dropped_trajectories": dropped,
        })
        if not kept:
            print("All teacher rollouts failed. Cannot run SFT. Aborting.")
            wandb.finish(exit_code=1)
            return
        teacher_trajectories = kept

    # ── Phase B: SFT ──
    best_step = await run_distillation_sft(
        model, teacher_trajectories, validation_tasks, config
    )

    # Diagnostics record. The leaderboard does not need to read this because
    # `.sft_endpoint_step` (written by train_tau2.py from model.get_step())
    # will already equal best_step after the post-SFT collection trim. Keeping
    # this around lets us detect mismatches when the trim silently no-oped
    # (e.g. private API removed in a future ART release).
    if best_step is not None:
        try:
            (config_path.resolve().parent / ".best_sft_step").write_text(str(best_step))
            print(f"  [best-chunk] wrote .best_sft_step = {best_step}")
        except Exception as e:
            print(f"  [best-chunk] failed to write .best_sft_step: {e}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ART SFT-distillation for tau2-bench")
    parser.add_argument("--config", default="train_distill_config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Cap training tasks for teacher rollouts (default: full train split)")
    parser.add_argument("--num-validation-tasks", type=int, default=None,
                        help="Override number of validation tasks")
    parser.add_argument("--sft-epochs", type=int, default=None,
                        help="Override SFT epochs")
    parser.add_argument("--teacher-rollouts-per-task", type=int, default=None,
                        help="Override teacher rollouts per task")
    parser.add_argument("--sft-peak-lr", type=float, default=None,
                        help="Override SFT peak learning rate")
    prefilter_group = parser.add_mutually_exclusive_group()
    prefilter_group.add_argument(
        "--prefilter-tasks", dest="prefilter_tasks", action="store_true",
        default=None,
        help="Probe tasks with 1 rollout first; concentrate remaining rollouts "
             "on solvable tasks only.",
    )
    prefilter_group.add_argument(
        "--no-prefilter-tasks", dest="prefilter_tasks", action="store_false",
        default=None,
        help="Disable probe; run all teacher_rollouts_per_task on every task.",
    )
    args = parser.parse_args()

    asyncio.run(main(args))

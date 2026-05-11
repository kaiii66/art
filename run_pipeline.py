"""End-to-end pipeline orchestrator for the tau2-bench training workflow.

Two backends are supported:

  --backend serverless  (default; legacy)
      Uses OpenPipe ART's ServerlessBackend. The original SFT and RL stages
      run in-process (train_tau2_distill.py, train_tau2.py) and the trained
      model lives in the ART collection registry. continue_from_model wires
      SFT -> RL via the existing ART checkpoint mechanism.

  --backend onprem
      Uses the on-prem 8xH100 K8s cluster (cwb607-ray) plus W&B Inference for
      model serving. SFT runs Axolotl in a K8s Job; RL runs rLLM (verl backend
      + colocated vLLM TP=8) in a K8s Job; both publish a `type=lora` artifact
      to W&B Inference. The pipeline reads each Job's resulting
      wandb-artifact:/// URI and passes it to the next stage.

Stage list (both backends share `snapshot`, `upload`, `leaderboard`):
  1. snapshot      copy train_config.yaml + train_distill_config.yaml into
                   pipeline_runs/<MMDDHHMM>/ and write a `group: pipeline-<MMDDHHMM>`
                   field into both copies. Source-of-truth YAMLs in the repo
                   root are never mutated. The W&B `project` field is NOT
                   rewritten so every iteration logs to the same stable project.
  2. upload        upload_dataset_to_wandb.py against the snapshot config
                   (idempotent: auto-skips if the dataset artifacts and Weave
                   datasets already exist; pass --force to re-upload).
  3. sft           serverless: train_tau2_distill.py (in-process)
                   onprem    : prepare_sft_data.py (Phase A locally) then
                               kubectl apply onprem/k8s/sft_job.yaml + wait,
                               write pipeline_runs/<tag>/.sft_lora_artifact_uri
                               with the resulting wandb-artifact:/// URI.
  4. patch         serverless: read .last_trained_model and write it to the
                               snapshot train_config.yaml's `continue_from_model`.
                   onprem    : verify .sft_lora_artifact_uri exists (passed
                               directly to the RL job env, no config patching).
  5. rl            serverless: train_tau2.py
                   onprem    : kubectl apply onprem/k8s/rl_job.yaml + wait,
                               write pipeline_runs/<tag>/.rl_lora_artifact_uri.
  6. leaderboard   create_leaderboard_shaped_reward.py --models all
                   (auto-discovers .sft_endpoint_step + .best_rl_step in
                   serverless mode, or .sft_lora_artifact_uri + .rl_lora_artifact_uri
                   in onprem mode).

Usage:
  uv run python run_pipeline.py                                # serverless (default)
  uv run python run_pipeline.py --backend onprem
  uv run python run_pipeline.py --backend onprem --image-tag $(git rev-parse --short HEAD)
  uv run python run_pipeline.py --skip upload sft
  uv run python run_pipeline.py --dry-run
  uv run python run_pipeline.py --project-suffix 04202030
  uv run python run_pipeline.py --resume pipeline_runs/04201530
"""
from __future__ import annotations

import argparse
import json
import os
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

ALL_STAGES = ["upload", "sft", "rl", "leaderboard"]

# On-prem backend constants (see onprem/ directory).
ONPREM_KUBECONFIG_DEFAULT = "/Users/ktan/.kube/config-cwb607-ray"
ONPREM_GHCR_USER_DEFAULT = os.environ.get("GHCR_USER", "")  # e.g. "kwt"
ONPREM_SFT_TEMPLATE = REPO_ROOT / "onprem" / "k8s" / "sft_job.yaml"
ONPREM_RL_TEMPLATE = REPO_ROOT / "onprem" / "k8s" / "rl_job.yaml"
ONPREM_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ONPREM_BASE_MODEL_SHORT = ONPREM_BASE_MODEL.split("/")[-1]
WANDB_PROJECT_BASE = "tau2-ART-autoresearch-telecom-"

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


def make_snapshot(
    suffix: str,
    resume_dir: Path | None,
    *,
    computed_project: str | None = None,
) -> tuple[Path, str, str]:
    """Create (or reuse) snapshot dir; return (snapshot_dir, project, group).

    When `computed_project` is provided it overrides the `project:` field in
    both source YAMLs before writing the snapshot copies, so neither
    train_config.yaml nor train_distill_config.yaml need to be edited between
    pipeline runs.  On --resume the project is always read from the existing
    snapshot (which already has the correct value baked in).
    """
    if resume_dir is not None:
        snapshot = resume_dir.resolve()
        if not snapshot.exists():
            raise FileNotFoundError(f"--resume dir does not exist: {snapshot}")
        train_cfg = yaml_load(snapshot / "train_config.yaml")
        project = train_cfg["project"]
        group = train_cfg.get("group") or f"pipeline-{snapshot.name}"
        if "group" not in train_cfg:
            # Backfill: older snapshots may not have a group field; persist one.
            train_cfg["group"] = group
            yaml_dump(train_cfg, snapshot / "train_config.yaml")
            distill = yaml_load(snapshot / "train_distill_config.yaml")
            distill["group"] = group
            yaml_dump(distill, snapshot / "train_distill_config.yaml")
        print(f"[snapshot] resuming existing snapshot: {snapshot}")
        print(f"[snapshot] project (from snapshot)   : {project}")
        print(f"[snapshot] group   (from snapshot)   : {group}")
        return snapshot, project, group

    snapshot = RUNS_DIR / suffix
    snapshot.mkdir(parents=True, exist_ok=True)

    src_train = yaml_load(SRC_TRAIN_CONFIG)
    src_distill = yaml_load(SRC_DISTILL_CONFIG)

    # Use the auto-derived project when provided; otherwise fall back to the
    # value baked into train_config.yaml.  Both snapshot copies are always
    # written with the same project so SFT/RL/leaderboard share artifacts.
    project = computed_project or src_train["project"]
    src_train["project"] = project
    src_distill["project"] = project

    # Per-iteration group: collapses every wandb.init() in this pipeline run
    # (upload, sft, rl, leaderboard) into one expandable bundle in the project.
    group = f"pipeline-{suffix}"
    src_train["group"] = group
    src_distill["group"] = group

    yaml_dump(src_train, snapshot / "train_config.yaml")
    yaml_dump(src_distill, snapshot / "train_distill_config.yaml")

    manifest = {
        "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "suffix": suffix,
        "project": project,
        "group": group,
        "source_train_config": str(SRC_TRAIN_CONFIG),
        "source_distill_config": str(SRC_DISTILL_CONFIG),
        "ruamel_used": _HAS_RUAMEL,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[snapshot] dir     : {snapshot}")
    print(f"[snapshot] project : {project}")
    print(f"[snapshot] group   : {group}")
    if not _HAS_RUAMEL:
        print(
            "[snapshot] WARNING: ruamel.yaml not installed; comments are stripped "
            "from snapshot copies (originals at repo root are untouched). "
            "`uv add ruamel.yaml` to preserve them."
        )
    return snapshot, project, group


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


# ─────────────────────────────────────────────────────────────────────
# On-prem (K8s) backend helpers
# ─────────────────────────────────────────────────────────────────────

def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "latest"


def _onprem_image(component: str, tag: str, ghcr_user: str) -> str:
    if not ghcr_user:
        raise RuntimeError(
            "GHCR_USER not set. Pass --ghcr-user or export GHCR_USER so we can "
            "build the image reference (ghcr.io/<user>/tau2-<component>:<tag>)."
        )
    return f"ghcr.io/{ghcr_user}/tau2-{component}:{tag}"


def _wandb_entity() -> str | None:
    """Best-effort: read WANDB_ENTITY from the env (set by .env)."""
    return os.environ.get("WANDB_ENTITY") or None


def _parse_wandb_artifact_uri(uri: str) -> tuple[str, str, str, str]:
    """Split `wandb-artifact:///<entity>/<project>/<name>:<alias>` -> 4-tuple.

    Used by the onprem leaderboard bridge to translate the URI we wrote to
    .sft_lora_artifact_uri / .rl_lora_artifact_uri (from the trainer pod's
    wandb_lora_upload.py output) into the (--{sft|rl}-trained-model-name,
    --{sft|rl}-model-alias) flag pair that create_leaderboard_shaped_reward.py
    expects.
    """
    body = uri.strip().removeprefix("wandb-artifact:///")
    path, _, alias = body.rpartition(":")
    if not alias:
        raise ValueError(
            f"URI {uri!r} has no :alias suffix; expected wandb-artifact:///e/p/n:vX"
        )
    parts = path.split("/", 2)
    if len(parts) != 3:
        raise ValueError(
            f"URI {uri!r} should have exactly entity/project/name (got {parts!r})"
        )
    entity, project, name = parts
    return entity, project, name, alias


def _build_leaderboard_cmd(
    *,
    train_cfg: Path,
    snapshot: Path,
    backend: str,
) -> list[str]:
    """Build the leaderboard subprocess argv, augmenting with onprem URI flags.

    Serverless: returns the legacy `--config X --models all` invocation. The
    leaderboard script auto-discovers the trained-model name + steps from
    .last_trained_model / .sft_endpoint_step / .best_rl_step.

    Onprem: also reads .sft_lora_artifact_uri (if present) and passes
    --sft-trained-model-name + --sft-model-alias so the script can register
    and pin the SFT row to the W&B artifact we uploaded from the cluster.
    Same for RL via .rl_lora_artifact_uri -> --trained-model-name +
    --trained-model-alias. Without these, onprem-uploaded LoRAs aren't
    discoverable (they don't carry :step{N} aliases the serverless path
    expects -- only :v0/:v1 W&B version aliases).
    """
    if backend != "onprem":
        return [
            "uv", "run", "python", "create_leaderboard_shaped_reward.py",
            "--config", str(train_cfg),
            "--models", "all",
        ]

    sft_uri_file = snapshot / ".sft_lora_artifact_uri"
    rl_uri_file = snapshot / ".rl_lora_artifact_uri"

    # Only evaluate rows that were actually trained this run.
    models = ["base"]
    if sft_uri_file.exists():
        models.append("sft")
    if rl_uri_file.exists():
        models.append("rl")

    cmd = [
        "uv", "run", "python", "create_leaderboard_shaped_reward.py",
        "--config", str(train_cfg),
        "--models", *models,
    ]

    if sft_uri_file.exists():
        try:
            _, _, sft_name, sft_alias = _parse_wandb_artifact_uri(sft_uri_file.read_text())
            cmd += ["--sft-trained-model-name", sft_name, "--sft-model-alias", sft_alias]
            print(f"    [leaderboard] sft pin: --sft-trained-model-name {sft_name} "
                  f"--sft-model-alias {sft_alias} (from {sft_uri_file.name})")
        except ValueError as e:
            print(f"    [leaderboard] WARNING: could not parse {sft_uri_file}: {e}")

    if rl_uri_file.exists():
        try:
            _, _, rl_name, rl_alias = _parse_wandb_artifact_uri(rl_uri_file.read_text())
            cmd += ["--trained-model-name", rl_name, "--trained-model-alias", rl_alias]
            print(f"    [leaderboard] rl pin: --trained-model-name {rl_name} "
                  f"--trained-model-alias {rl_alias} (from {rl_uri_file.name})")
        except ValueError as e:
            print(f"    [leaderboard] WARNING: could not parse {rl_uri_file}: {e}")

    return cmd


def _onprem_run_prepare_sft(snapshot: Path, dry_run: bool) -> str:
    """Run Phase A (teacher rollouts) locally and return the dataset URI.

    Short-circuits if the snapshot already has a `.sft_dataset_artifact_uri`
    marker file -- typically because a previous prepare-sft run already
    rolled out the teacher trajectories and uploaded them to W&B. Reusing
    the existing artifact saves ~10-20 minutes of teacher rollouts on every
    SFT-only retry. To force a fresh rollout, delete the marker file
    (`rm <snapshot>/.sft_dataset_artifact_uri`) before re-running.
    """
    marker = snapshot / ".sft_dataset_artifact_uri"
    if marker.exists() and not dry_run:
        existing_uri = marker.read_text().strip()
        print("\n=== stage: prepare-sft (skipped — reusing existing artifact) ===")
        print(f"    marker file  : {marker}")
        print(f"    artifact URI : {existing_uri}")
        print("    (delete the marker file to force a fresh teacher rollout)")
        return existing_uri

    distill_cfg = snapshot / "train_distill_config.yaml"
    cmd = [
        "uv", "run", "python", "-m", "onprem.scripts.prepare_sft_data",
        "--config", str(distill_cfg),
        "--out-dir", str(snapshot / "sft-data"),
    ]
    log_path = snapshot / "03a-prepare-sft.log"
    run_stage("prepare-sft", cmd, log_path, dry_run=dry_run)
    if dry_run:
        return "wandb-artifact:///DRY-RUN/DRY-RUN/DRY-RUN:latest"
    if not marker.exists():
        raise FileNotFoundError(
            f"{marker} not found after prepare-sft stage. "
            "Did onprem/scripts/prepare_sft_data.py finish successfully?"
        )
    return marker.read_text().strip()


def _onprem_submit_sft_job(
    *,
    snapshot: Path,
    suffix: str,
    project: str,
    group: str,
    image_tag: str,
    ghcr_user: str,
    dataset_uri: str,
    dry_run: bool,
) -> str:
    """Render + apply the SFT K8s Job and return the resulting LoRA URI."""
    from onprem.pipeline.k8s_runner import submit_and_wait, render_manifest

    job_name = f"tau2-sft-{suffix}"
    lora_artifact_name = f"tau2-sft-{ONPREM_BASE_MODEL_SHORT}-{suffix}"
    image = _onprem_image("sft", image_tag, ghcr_user)
    rendered = snapshot / "sft_job.rendered.yaml"

    substitutions = {
        "JOB_NAME":                 job_name,
        "IMAGE":                    image,
        "SFT_DATASET_ARTIFACT_URI": dataset_uri,
        "WANDB_PROJECT":            project,
        "WANDB_RUN_GROUP":          group,
        "WANDB_NAME":               f"sft-{suffix}",
        "PIPELINE_SUFFIX":          suffix,
        "LORA_ARTIFACT_NAME":       lora_artifact_name,
    }

    if dry_run:
        render_manifest(ONPREM_SFT_TEMPLATE, rendered, substitutions)
        print(f"    (dry-run; rendered manifest -> {rendered}; not applied)")
        entity = _wandb_entity() or "<entity>"
        return f"wandb-artifact:///{entity}/{project}/{lora_artifact_name}:latest"

    uri = submit_and_wait(
        template_path=ONPREM_SFT_TEMPLATE,
        rendered_path=rendered,
        substitutions=substitutions,
        job_name=job_name,
        suffix=suffix,
        kind="sft",
        snapshot_dir=snapshot,
    )
    return uri


def _onprem_submit_rl_job(
    *,
    snapshot: Path,
    suffix: str,
    project: str,
    group: str,
    image_tag: str,
    ghcr_user: str,
    starting_lora_uri: str,
    dry_run: bool,
) -> str:
    """Render + apply the RL K8s Job and return the resulting LoRA URI."""
    from onprem.pipeline.k8s_runner import submit_and_wait, render_manifest

    job_name = f"tau2-rl-{suffix}"
    lora_artifact_name = f"tau2-rl-{ONPREM_BASE_MODEL_SHORT}-{suffix}"
    image = _onprem_image("rllm", image_tag, ghcr_user)
    rendered = snapshot / "rl_job.rendered.yaml"

    substitutions = {
        "JOB_NAME":             job_name,
        "IMAGE":                image,
        "STARTING_LORA_URI":    starting_lora_uri,
        "WANDB_PROJECT":        project,
        "WANDB_RUN_GROUP":      group,
        "WANDB_NAME":           f"rl-{suffix}",
        "PIPELINE_SUFFIX":      suffix,
        "LORA_ARTIFACT_NAME":   lora_artifact_name,
    }

    if dry_run:
        render_manifest(ONPREM_RL_TEMPLATE, rendered, substitutions)
        print(f"    (dry-run; rendered manifest -> {rendered}; not applied)")
        entity = _wandb_entity() or "<entity>"
        return f"wandb-artifact:///{entity}/{project}/{lora_artifact_name}:latest"

    uri = submit_and_wait(
        template_path=ONPREM_RL_TEMPLATE,
        rendered_path=rendered,
        substitutions=substitutions,
        job_name=job_name,
        suffix=suffix,
        kind="rl",
        snapshot_dir=snapshot,
    )
    return uri


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backend",
        default="serverless",
        choices=["serverless", "onprem"],
        help="serverless = OpenPipe ART (legacy); onprem = K8s on cwb607-ray + W&B Inference LoRA.",
    )
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
        help="Print what would run without executing subprocesses or kubectl",
    )
    parser.add_argument(
        "--project-suffix",
        default=None,
        help="Override the MMDDHHMM suffix (default: now in America/Los_Angeles)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Reuse an existing pipeline_runs/<dir> instead of creating a new snapshot",
    )
    parser.add_argument(
        "--image-tag",
        default=None,
        help="GHCR image tag for on-prem Jobs (default: short git SHA, fallback 'latest').",
    )
    parser.add_argument(
        "--ghcr-user",
        default=None,
        help="GHCR username (e.g. your GitHub handle). Falls back to $GHCR_USER.",
    )
    parser.add_argument(
        "--kubeconfig",
        default=None,
        help=f"On-prem only: kubeconfig path (default: $KUBECONFIG or {ONPREM_KUBECONFIG_DEFAULT}).",
    )
    args = parser.parse_args()

    suffix = args.project_suffix or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")
    # Auto-derive project from MMDD so every pipeline run automatically lands
    # in the correct W&B project without requiring manual YAML edits.
    computed_project = WANDB_PROJECT_BASE + suffix[:4]
    snapshot, project, group = make_snapshot(suffix, args.resume, computed_project=computed_project)
    train_cfg = snapshot / "train_config.yaml"
    distill_cfg = snapshot / "train_distill_config.yaml"

    # On-prem-specific: resolve image tag, ghcr user, and kubeconfig once up front.
    image_tag = args.image_tag or _git_short_sha()
    ghcr_user = args.ghcr_user or ONPREM_GHCR_USER_DEFAULT
    if args.backend == "onprem":
        kubeconfig = args.kubeconfig or os.environ.get("KUBECONFIG") or ONPREM_KUBECONFIG_DEFAULT
        os.environ["KUBECONFIG"] = kubeconfig
        if not Path(kubeconfig).exists():
            print(
                f"WARNING: KUBECONFIG path {kubeconfig} not found. kubectl will fail. "
                "Set --kubeconfig or export KUBECONFIG."
            )

    skip = set(args.skip)
    print()
    print("=== pipeline plan ===")
    print(f"    backend          : {args.backend}")
    print(f"    snapshot         : {snapshot}")
    print(f"    project          : {project}")
    print(f"    group            : {group}")
    print(f"    train_config     : {train_cfg}")
    print(f"    distill_config   : {distill_cfg}")
    print(f"    stages enabled   : {[s for s in ALL_STAGES if s not in skip]}")
    print(f"    stages skipped   : {sorted(skip)}")
    if args.backend == "onprem":
        print(f"    image tag        : {image_tag}")
        print(f"    ghcr user        : {ghcr_user or '(unset; pass --ghcr-user or export GHCR_USER)'}")
        print(f"    kubeconfig       : {os.environ.get('KUBECONFIG', '(unset)')}")

    # ── stage: upload (shared) ──
    upload_cmd = ["uv", "run", "python", "upload_dataset_to_wandb.py", "--config", str(train_cfg)]
    if "upload" not in skip:
        run_stage("upload", upload_cmd, snapshot / "02-upload.log", dry_run=args.dry_run)
    else:
        print("\n=== stage: upload (skipped) ===")

    # ── stage: sft ──
    if "sft" not in skip:
        if args.backend == "serverless":
            sft_cmd = ["uv", "run", "python", "train_tau2_distill.py", "--config", str(distill_cfg)]
            run_stage("sft", sft_cmd, snapshot / "03-sft.log", dry_run=args.dry_run)
        else:
            print("\n=== stage: sft (onprem) ===")
            dataset_uri = _onprem_run_prepare_sft(snapshot, args.dry_run)
            sft_uri = _onprem_submit_sft_job(
                snapshot=snapshot, suffix=suffix, project=project, group=group,
                image_tag=image_tag, ghcr_user=ghcr_user,
                dataset_uri=dataset_uri, dry_run=args.dry_run,
            )
            (snapshot / ".sft_lora_artifact_uri").write_text(sft_uri)
            print(f"    sft LoRA URI -> {sft_uri}")
    else:
        print("\n=== stage: sft (skipped) ===")

    # ── stage: patch (implicit, runs right before rl) ──
    if "rl" not in skip:
        print("\n=== stage: patch (between sft and rl) ===")
        if args.dry_run:
            print(f"    (dry-run; backend={args.backend} would patch continue_from_model / verify SFT URI)")
        elif args.backend == "serverless":
            patch_continue_from_model(snapshot)
        else:
            sft_uri_file = snapshot / ".sft_lora_artifact_uri"
            if not sft_uri_file.exists():
                raise FileNotFoundError(
                    f"{sft_uri_file} not found — SFT stage did not write its LoRA URI. "
                    "Cannot wire STARTING_LORA_URI for the RL Job."
                )
            print(f"    [patch] STARTING_LORA_URI -> {sft_uri_file.read_text().strip()}")

    # ── stage: rl ──
    if "rl" not in skip:
        if args.backend == "serverless":
            rl_cmd = ["uv", "run", "python", "train_tau2.py", "--config", str(train_cfg)]
            run_stage("rl", rl_cmd, snapshot / "04-rl.log", dry_run=args.dry_run)
        else:
            print("\n=== stage: rl (onprem) ===")
            sft_uri = (snapshot / ".sft_lora_artifact_uri").read_text().strip() \
                if (snapshot / ".sft_lora_artifact_uri").exists() \
                else f"wandb-artifact:///{_wandb_entity() or '<entity>'}/{project}/tau2-sft-{ONPREM_BASE_MODEL_SHORT}-{suffix}:latest"
            rl_uri = _onprem_submit_rl_job(
                snapshot=snapshot, suffix=suffix, project=project, group=group,
                image_tag=image_tag, ghcr_user=ghcr_user,
                starting_lora_uri=sft_uri, dry_run=args.dry_run,
            )
            (snapshot / ".rl_lora_artifact_uri").write_text(rl_uri)
            print(f"    rl LoRA URI -> {rl_uri}")
    else:
        print("\n=== stage: rl (skipped) ===")

    # ── stage: leaderboard ──
    if "leaderboard" not in skip:
        # Build the leaderboard command here (after RL) so that .rl_lora_artifact_uri
        # is already written by the RL stage and the RL model is included in the
        # leaderboard. Building it at startup (before RL) silently drops the RL row.
        leaderboard_cmd = _build_leaderboard_cmd(
            train_cfg=train_cfg,
            snapshot=snapshot,
            backend=args.backend,
        )
        run_stage("leaderboard", leaderboard_cmd, snapshot / "05-leaderboard.log", dry_run=args.dry_run)
    else:
        print("\n=== stage: leaderboard (skipped) ===")

    print()
    print("=== done ===")
    print(f"    snapshot : {snapshot}")
    print(f"    project  : {project}")
    print(f"    group    : {group}")
    if (snapshot / ".last_trained_model").exists():
        print(f"    sft model        : {(snapshot / '.last_trained_model').read_text().strip()}")
    if (snapshot / ".sft_lora_artifact_uri").exists():
        print(f"    sft LoRA URI     : {(snapshot / '.sft_lora_artifact_uri').read_text().strip()}")
    if (snapshot / ".rl_lora_artifact_uri").exists():
        print(f"    rl LoRA URI      : {(snapshot / '.rl_lora_artifact_uri').read_text().strip()}")
    if (snapshot / ".sft_endpoint_step").exists():
        print(f"    sft step         : {(snapshot / '.sft_endpoint_step').read_text().strip()}")
    if (snapshot / ".best_rl_step").exists():
        print(f"    best rl          : step {(snapshot / '.best_rl_step').read_text().strip()}")
    print(f"    re-run leaderboard alone: uv run python run_pipeline.py --backend {args.backend} --resume {snapshot} --skip upload sft rl")
    return 0


if __name__ == "__main__":
    sys.exit(main())

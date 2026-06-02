"""Single-entry orchestrator for the tau2-bench end-to-end pipeline.

Wraps the existing 5-step README flow into one command:

    1. Generate an MMDDHHMM suffix in US/Pacific (or reuse via --suffix).
    2. Run SFT via run_pipeline.py --skip rl leaderboard (in-process,
       ServerlessBackend) — produces a W&B LoRA artifact and writes
       pipeline_runs/<SUFFIX>/.last_trained_model.
    3. Patch train_config_local.yaml in-place so sft_source.{entity,project,
       name} + top-level project point at the SFT artifact just created.
       (Also writes a snapshot copy at pipeline_runs/<SUFFIX>/.)
    4. docker build + docker push the on-prem RL image (tagged
       ghcr.io/$GHCR_USER/tau2-art:<short-sha>-<SUFFIX> so kubelet always
       re-pulls).
    5. Hand off to local/k8s_submit.py to render + kubectl apply the Job;
       inside the pod, run_pipeline_local.py does pull_sft → rl → upload_rl
       → leaderboard.

Usage
-----
    # Full pipeline (≈3 h, lands a Weave leaderboard with base / SFT / RL):
    uv run python run_full_pipeline.py

    # Smoke test (4 tasks, skips upload_rl + leaderboard, ≈30 min):
    uv run python run_full_pipeline.py --smoke --tail

    # Re-use an existing SFT snapshot (e.g., debugging the RL side):
    uv run python run_full_pipeline.py --suffix 05151200 --skip-sft

    # Re-submit a Job with the same image tag (skip rebuild + push):
    uv run python run_full_pipeline.py --suffix 05151200 --skip-sft --skip-build

The wrapper never modifies run_pipeline.py, local/k8s_submit.py, or any
template — it composes them as subprocesses, so each remains independently
runnable per the README's "Advanced" / Steps 1–5.
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

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG = REPO_ROOT / "train_config_local.yaml"
RUNS_DIR = REPO_ROOT / "pipeline_runs"

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


def _stream_subprocess(name: str, cmd: list[str], log_path: Path | None) -> int:
    """Run cmd; tee stdout/stderr to console (and optional log file).

    Returns the subprocess exit code.  Caller decides whether to raise.
    """
    print()
    print(f"=== {name} ===")
    print(f"    cmd : {' '.join(cmd)}")
    if log_path is not None:
        print(f"    log : {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = log_path.open("w") if log_path is not None else None
    if log_file is not None:
        log_file.write(
            f"# stage: {name}\n# cmd: {' '.join(cmd)}\n"
            f"# started: {datetime.now().isoformat()}\n\n"
        )
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
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        rc = proc.wait()
    finally:
        if log_file is not None:
            log_file.write(
                f"\n# exit_code: {rc if 'rc' in locals() else 'unknown'}\n"
                f"# ended: {datetime.now().isoformat()}\n"
            )
            log_file.close()
    return rc


def _preflight(slurm: bool = False, slurm_namespace: str = "tenant-slurm",
               slurm_login_pod: str = "slurm-login-0",
               slurm_login_container: str = "sshd") -> None:
    """Fail-fast on missing env / binaries / cluster access."""
    missing_env = [k for k in ("WANDB_API_KEY", "HF_TOKEN", "GHCR_USER") if not os.getenv(k)]
    if missing_env:
        raise SystemExit(
            f"missing env vars: {', '.join(missing_env)}.  "
            f"Populate them in {REPO_ROOT}/.env (see README 'Prerequisites')."
        )
    for binary in ("docker", "kubectl"):
        if shutil.which(binary) is None:
            raise SystemExit(f"{binary!r} not found on PATH.  See README Prerequisites.")

    if slurm:
        # Verify the Slurm login pod is reachable and cluster has idle nodes.
        sinfo = subprocess.run(
            ["kubectl", "exec", "-n", slurm_namespace, slurm_login_pod,
             "-c", slurm_login_container, "--", "sinfo", "--noheader"],
            capture_output=True, text=True,
        )
        if sinfo.returncode != 0:
            raise SystemExit(
                f"Cannot reach Slurm login pod {slurm_namespace}/{slurm_login_pod} "
                f"(container: {slurm_login_container}).  "
                f"Check KUBECONFIG and that the pod is Running.\n{sinfo.stderr.strip()}"
            )
        idle_lines = [l for l in sinfo.stdout.splitlines() if "idle" in l]
        if not idle_lines:
            print(
                f"[preflight] WARNING: no idle Slurm nodes found — job will queue.\n"
                f"{sinfo.stdout.strip()}"
            )
        else:
            print(f"[preflight] Slurm: {len(idle_lines)} partition(s) have idle nodes ✓")
    else:
        ns_check = subprocess.run(
            ["kubectl", "get", "ns", "tau2"],
            capture_output=True, text=True,
        )
        if ns_check.returncode != 0:
            raise SystemExit(
                "kubectl cannot see namespace 'tau2' — bootstrap it per the README "
                "'Prerequisites' section (PVCs + secrets)."
            )


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "nogit"


def _patch_sft_source(
    snapshot: Path,
    project: str,
    sft_name: str,
    sft_step_override: str | int | None,
) -> None:
    """Overwrite train_config_local.yaml's sft_source + top-level project.

    The Dockerfile's `COPY . .` then bakes this into the image so the pod
    reads the right SFT collection without manual editing.  Also copies the
    patched file to the snapshot dir for traceability.

    `sft_step_override` controls the step pin baked into the docker config:

    - None (default): auto-discover from `<snapshot>/.best_sft_step`. If the
      sidecar is missing (e.g. SFT was skipped or pre-best-step-tracking),
      fall back to "latest" with a warning.
    - "best" / "latest": passed through verbatim — the in-image
      pull_sft_lora.py will resolve via the sidecar / W&B :latest alias.
    - <int> / numeric string: hard-pin to that exact step (useful for
      ablations or reproducing a specific historical SFT).
    """
    if not _HAS_RUAMEL:
        print(
            "[patch] WARNING: ruamel.yaml not installed — comments in "
            f"{LOCAL_CONFIG.name} will be stripped by the patch.  "
            "Install it with `uv add ruamel.yaml` (or `pip install ruamel.yaml`) "
            "to preserve them."
        )
    cfg = yaml_load(LOCAL_CONFIG)
    cfg["project"] = project
    # Rebase model_name to the just-created SFT collection so the per-run
    # RL artifact (which run_pipeline_local.py builds as
    # f"{model_name}-rl-{SUFFIX}") doesn't carry a stale YYYYMMDD-HHMM from
    # whatever was checked into the repo.  E.g.,
    #   sft_name = tau2-distill-...-20260515-1000
    # → RL collection = tau2-distill-...-20260515-1000-rl-05151000
    cfg["model_name"] = sft_name
    sft = cfg.setdefault("sft_source", {})
    sft["entity"] = os.getenv("WANDB_ENTITY", "kwt")
    sft["project"] = project
    sft["name"] = sft_name

    # Resolve the step pin to bake into the image. Auto-discovery reads
    # .best_sft_step that train_tau2_distill.py wrote at the best val/reward
    # chunk; this is what makes the on-cluster RL stage continue from the
    # best SFT checkpoint rather than the noisy final one.
    best_step_file = snapshot / ".best_sft_step"
    if sft_step_override is None:
        if best_step_file.exists():
            try:
                resolved_step: int | str = int(best_step_file.read_text().strip())
                print(
                    f"[patch] {LOCAL_CONFIG.name} -> sft_source.step="
                    f"{resolved_step} (auto-discovered from .best_sft_step)"
                )
            except (ValueError, OSError) as e:
                resolved_step = "latest"
                print(
                    f"[patch] WARNING: could not parse {best_step_file}: {e}; "
                    f"falling back to sft_source.step='latest'"
                )
        else:
            resolved_step = "latest"
            print(
                f"[patch] WARNING: no .best_sft_step in {snapshot}; "
                f"falling back to sft_source.step='latest'. Re-run SFT to "
                "get best-step tracking, or pass --sft-step <int|best|latest>."
            )
    else:
        # Honour the explicit override. Convert numeric strings to int so the
        # YAML lands as `step: 28` (not `step: '28'`) which keeps it visually
        # consistent with the auto-discovery path.
        if isinstance(sft_step_override, str) and sft_step_override.isdigit():
            resolved_step = int(sft_step_override)
        else:
            resolved_step = sft_step_override
        print(
            f"[patch] {LOCAL_CONFIG.name} -> sft_source.step={resolved_step!r} "
            f"(--sft-step override)"
        )
    sft["step"] = resolved_step
    yaml_dump(cfg, LOCAL_CONFIG)

    # Snapshot copy for the audit trail.
    snapshot_copy = snapshot / "train_config_local.yaml"
    shutil.copy2(LOCAL_CONFIG, snapshot_copy)

    print(f"[patch] {LOCAL_CONFIG.name} -> project={project}")
    print(f"[patch] {LOCAL_CONFIG.name} -> model_name={sft_name}")
    print(f"[patch] {LOCAL_CONFIG.name} -> sft_source.name={sft_name}")
    print(f"[patch] snapshot copy: {snapshot_copy}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suffix", default=None,
        help="MMDDHHMM run suffix (default: now() in America/Los_Angeles). "
             "Shared by SFT and RL so cross-stage state is correlated.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-test the RL side: --num-tasks 4, skip upload_rl + leaderboard.",
    )
    parser.add_argument(
        "--skip-sft", action="store_true",
        help="Skip the SFT stage.  Requires --suffix pointing at an existing "
             "pipeline_runs/<SUFFIX>/ with .last_trained_model already present.",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip docker build + push.  Reuses the image tag stored in "
             "pipeline_runs/<SUFFIX>/k8s_manifest.json from a prior submit.",
    )
    parser.add_argument(
        "--tail", action="store_true",
        help="After kubectl apply, stream `kubectl logs -f` until the Job ends.",
    )
    parser.add_argument(
        "--slurm", action="store_true",
        help=(
            "Submit the RL job via Slurm (sbatch through the SUNK login pod) "
            "instead of a vanilla Kubernetes Job.  Use on clusters running SUNK "
            "(Slurm on Kubernetes) where kubectl apply Jobs cannot land on GPU nodes."
        ),
    )
    parser.add_argument(
        "--slurm-namespace", default="tenant-slurm", metavar="NS",
        help="K8s namespace of the Slurm login pod (default: tenant-slurm)",
    )
    parser.add_argument(
        "--slurm-login-pod", default="slurm-login-0", metavar="POD",
        help="Name of the Slurm login pod (default: slurm-login-0)",
    )
    parser.add_argument(
        "--slurm-login-container", default="sshd", metavar="CTR",
        help="Container inside the login pod to exec into (default: sshd)",
    )
    parser.add_argument(
        "--nfs-base", default="/mnt/data/kai", metavar="PATH",
        help=(
            "Base path on the NFS share shared between login and compute nodes. "
            "Artifacts, logs and enroot credentials are written here. "
            "(default: /mnt/data/kai)"
        ),
    )
    parser.add_argument(
        "--slurm-partition", default="h100", metavar="PARTITION",
        help="Slurm partition to target (default: h100)",
    )
    parser.add_argument(
        "--slurm-time-limit", default="08:00:00", metavar="HH:MM:SS",
        help="Slurm wall-clock time limit (default: 08:00:00)",
    )
    parser.add_argument(
        "--sft-step",
        default=None,
        help=(
            "Override the SFT checkpoint step baked into the docker image. "
            "Defaults to auto-discovery (reads pipeline_runs/<SUFFIX>/.best_sft_step "
            "and pins sft_source.step to that integer). Pass an integer to "
            "hard-pin (e.g. --sft-step 28), or 'best' / 'latest' to delegate "
            "resolution to the in-image pull_sft_lora.py."
        ),
    )
    args = parser.parse_args()

    _preflight(
        slurm=args.slurm,
        slurm_namespace=args.slurm_namespace,
        slurm_login_pod=args.slurm_login_pod,
        slurm_login_container=args.slurm_login_container,
    )

    suffix = args.suffix or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")
    snapshot = RUNS_DIR / suffix
    snapshot.mkdir(parents=True, exist_ok=True)

    print()
    print("=== run_full_pipeline ===")
    print(f"    suffix     : {suffix}")
    print(f"    snapshot   : {snapshot}")
    print(f"    smoke      : {args.smoke}")
    print(f"    skip_sft   : {args.skip_sft}")
    print(f"    skip_build : {args.skip_build}")
    print(f"    tail       : {args.tail}")
    print(f"    backend    : {'slurm' if args.slurm else 'k8s'}")

    # ── 1. SFT ────────────────────────────────────────────────────────────
    if not args.skip_sft:
        sft_cmd = [
            "uv", "run", "python", "run_pipeline.py",
            "--project-suffix", suffix,
            "--skip", "rl", "leaderboard",
        ]
        rc = _stream_subprocess("stage: sft", sft_cmd, log_path=None)
        if rc != 0:
            raise SystemExit(f"SFT stage failed with exit code {rc}; aborting.")
    else:
        print(f"\n=== stage: sft (skipped via --skip-sft) ===")

    # ── 2. Read handoff sidecars ──────────────────────────────────────────
    last_model_file = snapshot / ".last_trained_model"
    manifest_file = snapshot / "manifest.json"
    if not last_model_file.exists():
        raise SystemExit(
            f"expected {last_model_file} but file is missing.  "
            f"SFT did not complete, or --skip-sft was used with a stale --suffix."
        )
    sft_name = last_model_file.read_text().strip()
    if not sft_name:
        raise SystemExit(f"{last_model_file} is empty.")
    if not manifest_file.exists():
        raise SystemExit(f"expected {manifest_file} but file is missing.")
    manifest = json.loads(manifest_file.read_text())
    project = manifest.get("project")
    if not project:
        raise SystemExit(f"{manifest_file} has no 'project' field.")
    print()
    print(f"[handoff] project  = {project}")
    print(f"[handoff] sft_name = {sft_name}")

    # ── 3. Patch train_config_local.yaml in place ─────────────────────────
    _patch_sft_source(snapshot, project, sft_name, args.sft_step)

    # ── 4. docker build + push ────────────────────────────────────────────
    gh_user = os.getenv("GHCR_USER")
    image_tag = f"ghcr.io/{gh_user}/tau2-art:{_git_short_sha()}-{suffix}"

    if not args.skip_build:
        build_cmd = [
            "docker", "build", "--progress=plain",
            "-f", "onprem/Dockerfile.art-rl",
            "-t", image_tag, ".",
        ]
        build_log = snapshot / "docker-build.log"
        rc = _stream_subprocess("stage: docker build", build_cmd, log_path=build_log)
        if rc != 0:
            raise SystemExit(f"docker build failed (rc={rc}); see {build_log}.")

        push_cmd = ["docker", "push", image_tag]
        rc = _stream_subprocess("stage: docker push", push_cmd, log_path=None)
        if rc != 0:
            raise SystemExit(f"docker push failed (rc={rc}).")
    else:
        manifest_path = snapshot / "k8s_manifest.json"
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text())
            image_tag = prior.get("image_tag", image_tag)
            print(f"\n=== stage: docker build/push (skipped via --skip-build) ===")
            print(f"    reusing image_tag from {manifest_path.name}: {image_tag}")
        else:
            print(f"\n=== stage: docker build/push (skipped via --skip-build) ===")
            print(f"    no prior k8s_manifest.json; using computed tag: {image_tag}")

    # ── 5. submit ─────────────────────────────────────────────────────────
    if args.slurm:
        submit_cmd = [
            "uv", "run", "python", "local/slurm_submit.py",
            "--image-tag", image_tag,
            "--suffix", suffix,
            "--slurm-namespace", args.slurm_namespace,
            "--slurm-login-pod", args.slurm_login_pod,
            "--slurm-login-container", args.slurm_login_container,
            "--nfs-base", args.nfs_base,
            "--partition", args.slurm_partition,
            "--time-limit", args.slurm_time_limit,
        ]
        if args.smoke:
            submit_cmd += ["--skip-stages", "upload_rl leaderboard", "--num-tasks", "4"]
        if args.tail:
            submit_cmd += ["--tail"]
        rc = _stream_subprocess("stage: slurm submit", submit_cmd, log_path=None)
        if rc != 0:
            raise SystemExit(f"slurm_submit failed (rc={rc}).")
    else:
        submit_cmd = [
            "uv", "run", "python", "local/k8s_submit.py",
            "--image-tag", image_tag,
            "--suffix", suffix,
        ]
        if args.smoke:
            submit_cmd += ["--skip-stages", "upload_rl leaderboard", "--num-tasks", "4"]
        if args.tail:
            submit_cmd += ["--tail"]
        rc = _stream_subprocess("stage: k8s submit", submit_cmd, log_path=None)
        if rc != 0:
            raise SystemExit(f"k8s_submit failed (rc={rc}).")

    # ── 6. Summary ────────────────────────────────────────────────────────
    print()
    print("=== run_full_pipeline complete ===")
    print(f"    suffix              : {suffix}")
    print(f"    snapshot            : {snapshot}")
    print(f"    image               : {image_tag}")
    print(f"    job                 : tau2-art-rl-{suffix}")
    print(f"    project             : {project}")
    if args.slurm:
        print(f"    tail logs           : kubectl exec -n {args.slurm_namespace} "
              f"{args.slurm_login_pod} -c {args.slurm_login_container} -- "
              f"tail -f {args.nfs_base}/logs/tau2-{suffix}-<JOBID>.log")
    else:
        print(f"    tail logs           : kubectl logs -f job/tau2-art-rl-{suffix} -n tau2")
    print(f"    weave leaderboard   : "
          f"https://wandb.ai/{os.getenv('WANDB_ENTITY', 'kwt')}/{project}"
          f"/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())

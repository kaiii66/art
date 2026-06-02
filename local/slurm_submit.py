"""Submit the tau2-art RL pipeline as a Slurm batch job via the SUNK login pod.

Mirrors the interface of local/k8s_submit.py so run_full_pipeline.py can swap
between backends with a single --slurm flag.

Workflow
--------
1. Ensures NFS working directories exist (kubectl exec mkdir).
2. Writes enroot credentials to the NFS share so compute nodes can pull the
   GHCR image (enroot reads $HOME/.config/enroot/.credentials at job time).
3. Renders an sbatch script onto the NFS share with all env vars and pyxis
   container-mount directives.
4. Submits via ``kubectl exec -n <ns> <login-pod> -c <container> -- sbatch``.
5. Optionally tails the job log file through the same kubectl exec path.

Usage
-----
    # Dry-run (print rendered sbatch, skip submission):
    uv run python local/slurm_submit.py \\
        --image-tag ghcr.io/kaiii66/tau2-art:abc1234-05151200 \\
        --suffix 05151200 --dry-run

    # Submit and tail:
    uv run python local/slurm_submit.py \\
        --image-tag ghcr.io/kaiii66/tau2-art:abc1234-05151200 \\
        --suffix 05151200 --tail

    # Override Slurm cluster topology:
    uv run python local/slurm_submit.py \\
        --image-tag ... --suffix ... \\
        --slurm-namespace tenant-slurm \\
        --slurm-login-pod slurm-login-0 \\
        --nfs-base /mnt/data/kai \\
        --partition h100 \\
        --time-limit 10:00:00
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
LOCAL_CONFIG = REPO_ROOT / "train_config_local.yaml"
RUNS_DIR = REPO_ROOT / "pipeline_runs"


def _load_config(path: Path) -> dict:
    if _yaml is None:
        raise RuntimeError("PyYAML is not installed; run: uv add pyyaml")
    with path.open() as f:
        return _yaml.safe_load(f)


def _kubectl_exec(pod: str, ns: str, container: str, cmd: str) -> subprocess.CompletedProcess:
    """Run a shell command inside the Slurm login pod."""
    return subprocess.run(
        ["kubectl", "exec", "-n", ns, pod, "-c", container, "--", "bash", "-c", cmd],
        capture_output=True,
        text=True,
    )


def _get_ghcr_token() -> tuple[str, str] | None:
    """Return (username, token) from ~/.docker/config.json, or None."""
    docker_cfg = Path.home() / ".docker" / "config.json"
    if not docker_cfg.exists():
        return None
    try:
        d = json.loads(docker_cfg.read_text())
        auth_b64 = d.get("auths", {}).get("ghcr.io", {}).get("auth", "")
        if not auth_b64:
            return None
        user, _, token = base64.b64decode(auth_b64).decode().partition(":")
        return user.strip(), token.strip()
    except Exception:
        return None


def _setup_nfs(pod: str, ns: str, container: str, nfs_base: str, creds: tuple[str, str] | None) -> None:
    """Create working dirs and write enroot credentials on the NFS share."""
    dirs = f"{nfs_base}/tau2-artifacts {nfs_base}/tau2-data {nfs_base}/logs {nfs_base}/.config/enroot"
    r = _kubectl_exec(pod, ns, container, f"mkdir -p {dirs}")
    if r.returncode != 0:
        raise RuntimeError(f"Failed to create NFS dirs: {r.stderr.strip()}")
    print(f"[slurm_submit] NFS dirs ensured under {nfs_base}")

    if creds:
        user, token = creds
        creds_cmd = (
            f"echo 'machine ghcr.io login {user} password {token}' "
            f"> {nfs_base}/.config/enroot/.credentials && "
            f"chmod 600 {nfs_base}/.config/enroot/.credentials"
        )
        r = _kubectl_exec(pod, ns, container, creds_cmd)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to write enroot credentials: {r.stderr.strip()}")
        print(f"[slurm_submit] enroot credentials written to {nfs_base}/.config/enroot/.credentials")
    else:
        print("[slurm_submit] WARNING: no GHCR credentials found in ~/.docker/config.json; "
              "image pull will fail if the image is private")


def _render_sbatch(
    *,
    image_tag: str,
    suffix: str,
    wandb_project: str,
    wandb_api_key: str,
    hf_token: str,
    openai_api_key: str,
    gemini_api_key: str,
    nfs_base: str,
    partition: str,
    time_limit: str,
    skip_stages: str,
    num_tasks: str,
) -> str:
    # Build optional frontier-baseline lines so they're absent (not empty)
    # when keys are missing — avoids passing empty strings into the container.
    openai_export = f"export OPENAI_API_KEY='{openai_api_key}'" if openai_api_key else "# OPENAI_API_KEY not set — gpt-4.1-mini leaderboard row will be skipped"
    gemini_export = f"export GEMINI_API_KEY='{gemini_api_key}'" if gemini_api_key else "# GEMINI_API_KEY not set — gemini leaderboard row will be skipped"
    frontier_env = ",OPENAI_API_KEY,GEMINI_API_KEY" if (openai_api_key or gemini_api_key) else ""

    return f"""#!/bin/bash
#SBATCH --job-name=tau2-art-rl-{suffix}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time={time_limit}
#SBATCH --output={nfs_base}/logs/tau2-{suffix}-%j.log
#SBATCH --error={nfs_base}/logs/tau2-{suffix}-%j.log

# Point enroot to NFS-backed credentials (compute nodes read from $HOME)
export HOME={nfs_base}

export WANDB_API_KEY='{wandb_api_key}'
export HF_TOKEN='{hf_token}'
export WANDB_PROJECT='{wandb_project}'
export WANDB_RUN_GROUP='pipeline-{suffix}'
export WANDB_NAME='art-rl-{suffix}'
export PIPELINE_SUFFIX='{suffix}'
export SKIP_STAGES='{skip_stages}'
export NUM_TASKS='{num_tasks}'
export TAU2_DATA_DIR='/workspace/data'
export HF_HOME='/artifacts/hf-cache'
export HUGGINGFACE_HUB_CACHE='/artifacts/hf-cache/hub'
export HF_HUB_ENABLE_HF_TRANSFER='1'
export TOKENIZERS_PARALLELISM='false'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export NCCL_P2P_DISABLE='0'
export NCCL_DEBUG='WARN'
# Frontier-baseline leaderboard rows (optional — rows are skipped if keys absent)
{openai_export}
{gemini_export}

srun \\
  --container-image={image_tag} \\
  --container-mounts={nfs_base}/tau2-artifacts:/artifacts,{nfs_base}/tau2-data:/data,{nfs_base}/logs:/workspace/pipeline_runs \\
  --container-env=WANDB_API_KEY,HF_TOKEN,WANDB_PROJECT,WANDB_RUN_GROUP,WANDB_NAME,PIPELINE_SUFFIX,SKIP_STAGES,NUM_TASKS,TAU2_DATA_DIR,HF_HOME,HUGGINGFACE_HUB_CACHE,HF_HUB_ENABLE_HF_TRANSFER,TOKENIZERS_PARALLELISM,PYTORCH_CUDA_ALLOC_CONF,NCCL_P2P_DISABLE,NCCL_DEBUG{frontier_env} \\
  /workspace/onprem/scripts/run_art_rl.sh
"""


def _tail_slurm_log(pod: str, ns: str, container: str, log_glob: str) -> None:
    """Wait for the log file to appear then stream it until the job ends."""
    print(f"[slurm_submit] waiting for log file matching {log_glob} …")
    for _ in range(60):
        r = _kubectl_exec(pod, ns, container, f"ls {log_glob} 2>/dev/null | head -1")
        if r.returncode == 0 and r.stdout.strip():
            log_path = r.stdout.strip()
            break
        time.sleep(3)
    else:
        print("[slurm_submit] WARNING: log file never appeared; giving up on tail")
        return

    print(f"[slurm_submit] streaming {log_path}")
    proc = subprocess.Popen(
        ["kubectl", "exec", "-n", ns, pod, "-c", container, "--",
         "tail", "-f", "-n", "+1", log_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image-tag", required=True,
                        help="Full image ref, e.g. ghcr.io/kaiii66/tau2-art:abc1234")
    parser.add_argument("--suffix", default=None,
                        help="MMDDHHMM pipeline suffix (default: now in US/Pacific)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the rendered sbatch script; do not submit")
    parser.add_argument("--tail", action="store_true",
                        help="After submit, stream the job log until it finishes")
    parser.add_argument("--skip-stages", default="",
                        help='Space-separated stages to skip inside the job (e.g. "upload_rl leaderboard")')
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Pass --num-tasks N to the RL stage (smoke test)")
    # Slurm topology
    parser.add_argument("--slurm-namespace", default="tenant-slurm",
                        help="K8s namespace of the Slurm login pod (default: tenant-slurm)")
    parser.add_argument("--slurm-login-pod", default="slurm-login-0",
                        help="Name of the Slurm login pod (default: slurm-login-0)")
    parser.add_argument("--slurm-login-container", default="sshd",
                        help="Container name inside the login pod (default: sshd)")
    parser.add_argument("--nfs-base", default="/mnt/data/kai",
                        help="Base path on the NFS share accessible from compute nodes (default: /mnt/data/kai)")
    parser.add_argument("--partition", default="h100",
                        help="Slurm partition to submit to (default: h100)")
    parser.add_argument("--time-limit", default="08:00:00",
                        help="Slurm wall-clock limit (default: 08:00:00)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    suffix = args.suffix or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")
    snapshot_dir = RUNS_DIR / suffix
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(LOCAL_CONFIG)
    wandb_project: str = cfg.get("project", "tau2-ART-distill")

    wandb_api_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not wandb_api_key or not hf_token:
        raise SystemExit("WANDB_API_KEY and HF_TOKEN must be set in .env")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not openai_api_key:
        print("[slurm_submit] WARNING: OPENAI_API_KEY not in .env — gpt-4.1-mini leaderboard row will be skipped")
    if not gemini_api_key:
        print("[slurm_submit] WARNING: GEMINI_API_KEY not in .env — gemini leaderboard row will be skipped")

    num_tasks_str = "" if args.num_tasks is None else str(args.num_tasks)

    script = _render_sbatch(
        image_tag=args.image_tag,
        suffix=suffix,
        wandb_project=wandb_project,
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        nfs_base=args.nfs_base,
        partition=args.partition,
        time_limit=args.time_limit,
        skip_stages=args.skip_stages,
        num_tasks=num_tasks_str,
    )

    script_path = snapshot_dir / "art_rl_job.sbatch"
    script_path.write_text(script)

    print(f"[slurm_submit] suffix              : {suffix}")
    print(f"[slurm_submit] job name            : tau2-art-rl-{suffix}")
    print(f"[slurm_submit] image               : {args.image_tag}")
    print(f"[slurm_submit] wandb project       : {wandb_project}")
    print(f"[slurm_submit] partition           : {args.partition}")
    print(f"[slurm_submit] login pod           : {args.slurm_namespace}/{args.slurm_login_pod}")
    print(f"[slurm_submit] nfs base            : {args.nfs_base}")
    print(f"[slurm_submit] local script copy   : {script_path}")

    if args.dry_run:
        print()
        print("=== rendered sbatch script (dry-run) ===")
        print(script)
        print("=== (dry-run; sbatch not called) ===")
        return 0

    # Setup NFS dirs + enroot credentials
    creds = _get_ghcr_token()
    _setup_nfs(
        args.slurm_login_pod, args.slurm_namespace, args.slurm_login_container,
        args.nfs_base, creds,
    )

    # Copy sbatch script to NFS and submit
    nfs_script = f"{args.nfs_base}/tau2-art-rl.sbatch"
    escaped = script.replace("'", "'\\''")
    write_cmd = f"cat > {nfs_script} << 'SLURM_SCRIPT_EOF'\n{script}\nSLURM_SCRIPT_EOF\nchmod 600 {nfs_script}"
    r = _kubectl_exec(args.slurm_login_pod, args.slurm_namespace, args.slurm_login_container, write_cmd)
    if r.returncode != 0:
        raise SystemExit(f"Failed to write sbatch script to NFS: {r.stderr.strip()}")

    r = _kubectl_exec(args.slurm_login_pod, args.slurm_namespace, args.slurm_login_container,
                      f"sbatch {nfs_script}")
    if r.returncode != 0:
        raise SystemExit(f"sbatch failed: {r.stderr.strip()}\n{r.stdout.strip()}")

    output = r.stdout.strip()
    print(f"[slurm_submit] {output}")  # e.g. "Submitted batch job 4896"

    job_id = output.split()[-1] if output else "unknown"
    log_glob = f"{args.nfs_base}/logs/tau2-{suffix}-{job_id}.log"

    # Persist manifest so --skip-build can reuse image_tag
    manifest_path = snapshot_dir / "k8s_manifest.json"
    manifest_path.write_text(json.dumps({
        "submitted_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "suffix": suffix,
        "job_name": f"tau2-art-rl-{suffix}",
        "slurm_job_id": job_id,
        "image_tag": args.image_tag,
        "wandb_project": wandb_project,
        "backend": "slurm",
        "slurm_namespace": args.slurm_namespace,
        "slurm_login_pod": args.slurm_login_pod,
        "nfs_base": args.nfs_base,
        "log_path": log_glob,
    }, indent=2))

    print(f"[slurm_submit] Monitor with:")
    print(f"    kubectl exec -n {args.slurm_namespace} {args.slurm_login_pod} -c {args.slurm_login_container} -- squeue")
    print(f"    kubectl exec -n {args.slurm_namespace} {args.slurm_login_pod} -c {args.slurm_login_container} -- tail -f {log_glob}")

    if args.tail:
        _tail_slurm_log(args.slurm_login_pod, args.slurm_namespace, args.slurm_login_container, log_glob)

    return 0


if __name__ == "__main__":
    sys.exit(main())

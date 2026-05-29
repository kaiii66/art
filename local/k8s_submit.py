"""Submit the tau2-art RL pipeline as a Kubernetes Job.

Workflow
--------
1. Generates a PIPELINE_SUFFIX (MMDDHHMM in US/Pacific) — or uses --suffix.
2. Reads WANDB_PROJECT from train_config_local.yaml.
3. Renders onprem/k8s/art-rl-job.yaml.template → pipeline_runs/<SUFFIX>/art_rl_job.rendered.yaml.
4. Runs ``kubectl apply -f <rendered>`` (unless --dry-run).
5. Optionally tails the Job logs (``kubectl logs -f job/<name>``).

Usage
-----
    # Build + push the image first:
    cd art
    IMAGE_TAG=$(git rev-parse --short HEAD)
    docker build -f onprem/Dockerfile.art-rl -t ghcr.io/kaiii66/tau2-art:$IMAGE_TAG .
    docker push ghcr.io/kaiii66/tau2-art:$IMAGE_TAG

    # Submit the Job:
    uv run python local/k8s_submit.py --image-tag ghcr.io/kaiii66/tau2-art:$IMAGE_TAG

    # Dry-run (print rendered YAML, skip kubectl):
    uv run python local/k8s_submit.py --image-tag ghcr.io/kaiii66/tau2-art:abc1234 --dry-run

    # Override suffix (useful when re-submitting a specific pipeline run):
    uv run python local/k8s_submit.py --image-tag ... --suffix 05121500

    # Submit then tail logs until Job finishes:
    uv run python local/k8s_submit.py --image-tag ... --tail
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
TEMPLATE_PATH = REPO_ROOT / "onprem" / "k8s" / "art-rl-job.yaml.template"
LOCAL_CONFIG = REPO_ROOT / "train_config_local.yaml"
RUNS_DIR = REPO_ROOT / "pipeline_runs"

# Keys in the YAML template that get replaced (TEMPLATE_ prefix avoids collisions
# with real YAML content).
_REPLACEMENTS = {
    "TEMPLATE_JOB_NAME": "",
    "TEMPLATE_IMAGE_TAG": "",
    "TEMPLATE_PIPELINE_SUFFIX": "",
    "TEMPLATE_WANDB_PROJECT": "",
    "TEMPLATE_WANDB_RUN_GROUP": "",
    "TEMPLATE_WANDB_NAME": "",
    "TEMPLATE_SKIP_STAGES": "",
    "TEMPLATE_NUM_TASKS": "",
}


def _load_config(path: Path) -> dict:
    if _yaml is None:
        raise RuntimeError("PyYAML is not installed; run: uv add pyyaml")
    with path.open() as f:
        return _yaml.safe_load(f)


def _render_template(template_text: str, values: dict[str, str]) -> str:
    result = template_text
    for key, val in values.items():
        result = result.replace(key, val)
    # Sanity check: warn if any TEMPLATE_ markers remain unreplaced.
    remaining = [word for word in result.split() if "TEMPLATE_" in word]
    if remaining:
        print(f"[k8s_submit] WARNING: unreplaced template markers: {remaining}", file=sys.stderr)
    return result


def _upsert_provider_secret(
    namespace: str,
    snapshot_dir: Path,
    *,
    env_var: str,
    secret_name: str,
    row_label: str,
) -> None:
    """Idempotently create/update a `{secret_name}` k8s secret from the local env.

    Resolution order:
      1. ``{env_var}`` in the current process environment.
      2. ``{env_var}`` in art/.env (dotenv file next to train_config_local.yaml).

    If neither source has the key, a warning is printed but the function returns
    without error — the Job template uses ``optional: true`` so pods start even
    when the secret is absent; the corresponding leaderboard row will simply be
    skipped.
    """
    key = os.environ.get(env_var)
    if not key:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{env_var}="):
                    _, _, val = line.partition("=")
                    key = val.strip().strip('"').strip("'")
                    break

    if not key:
        print(
            f"[k8s_submit] WARNING: {env_var} not found in env or .env — "
            f"{secret_name} k8s secret NOT created; {row_label} leaderboard row will be skipped.",
            file=sys.stderr,
        )
        return

    # --dry-run=client | kubectl apply is idempotent: create-or-update.
    secret_path = snapshot_dir / f"{secret_name}_secret.yaml"
    dry_run = subprocess.run(
        [
            "kubectl", "create", "secret", "generic", secret_name,
            f"--from-literal=api={key}",
            "-n", namespace,
            "--dry-run=client", "-o", "yaml",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if dry_run.returncode != 0:
        print(
            f"[k8s_submit] WARNING: could not render {secret_name} secret: {dry_run.stderr.strip()}",
            file=sys.stderr,
        )
        return

    secret_path.write_text(dry_run.stdout)
    result = subprocess.run(
        ["kubectl", "apply", "-f", str(secret_path), "-n", namespace],
        check=False,
    )
    if result.returncode == 0:
        print(f"[k8s_submit] {secret_name} k8s secret upserted in namespace {namespace!r}")
    else:
        print(
            f"[k8s_submit] WARNING: kubectl apply for {secret_name} secret failed (exit {result.returncode})",
            file=sys.stderr,
        )


def _upsert_openai_secret(namespace: str, snapshot_dir: Path) -> None:
    _upsert_provider_secret(
        namespace, snapshot_dir,
        env_var="OPENAI_API_KEY", secret_name="openai",
        row_label="gpt-4.1-mini",
    )


def _upsert_gemini_secret(namespace: str, snapshot_dir: Path) -> None:
    _upsert_provider_secret(
        namespace, snapshot_dir,
        env_var="GEMINI_API_KEY", secret_name="gemini",
        row_label="gemini-3.5-flash",
    )


def _git_short_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image-tag",
        required=True,
        help="Full image ref, e.g. ghcr.io/kaiii66/tau2-art:abc1234",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Override the MMDDHHMM pipeline suffix (default: now in US/Pacific)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the Job YAML and print it, but do not call kubectl",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="After submit, tail kubectl logs until the Job finishes",
    )
    parser.add_argument(
        "--namespace",
        default="tau2",
        help="Kubernetes namespace (default: tau2)",
    )
    parser.add_argument(
        "--skip-stages",
        default="",
        help='Space-separated stage names to skip inside the Job (e.g. "upload_rl leaderboard")',
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Pass --num-tasks N to the RL stage (smoke test). 4 = quick smoke.",
    )
    args = parser.parse_args()

    suffix = args.suffix or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")

    # Read WANDB_PROJECT from the local RL config.
    cfg = _load_config(LOCAL_CONFIG)
    wandb_project: str = cfg.get("project", "tau2-ART-distill")

    job_name = f"tau2-art-rl-{suffix}"
    run_group = f"pipeline-{suffix}"
    run_name = f"art-rl-{suffix}"

    values = {
        "TEMPLATE_JOB_NAME": job_name,
        "TEMPLATE_IMAGE_TAG": args.image_tag,
        "TEMPLATE_PIPELINE_SUFFIX": suffix,
        "TEMPLATE_WANDB_PROJECT": wandb_project,
        "TEMPLATE_WANDB_RUN_GROUP": run_group,
        "TEMPLATE_WANDB_NAME": run_name,
        "TEMPLATE_SKIP_STAGES": args.skip_stages,
        "TEMPLATE_NUM_TASKS": "" if args.num_tasks is None else str(args.num_tasks),
    }

    template_text = TEMPLATE_PATH.read_text()
    rendered = _render_template(template_text, values)

    # Write rendered YAML alongside the snapshot.
    snapshot_dir = RUNS_DIR / suffix
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = snapshot_dir / "art_rl_job.rendered.yaml"
    rendered_path.write_text(rendered)

    # Write a small manifest so run_pipeline_local.py --resume can pick this up.
    manifest_path = snapshot_dir / "k8s_manifest.json"
    manifest = {
        "submitted_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "suffix": suffix,
        "job_name": job_name,
        "image_tag": args.image_tag,
        "wandb_project": wandb_project,
        "namespace": args.namespace,
        "rendered_yaml": str(rendered_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[k8s_submit] suffix         : {suffix}")
    print(f"[k8s_submit] job name       : {job_name}")
    print(f"[k8s_submit] image          : {args.image_tag}")
    print(f"[k8s_submit] wandb project  : {wandb_project}")
    print(f"[k8s_submit] rendered yaml  : {rendered_path}")

    if args.dry_run:
        print()
        print("=== rendered Job YAML (dry-run) ===")
        print(rendered)
        print("=== (dry-run; kubectl not called) ===")
        return 0

    # Upsert provider secrets so the corresponding frontier-baseline leaderboard
    # rows get their keys (both are optional; the row is skipped if the key is
    # missing in .env and the secret cannot be created).
    _upsert_openai_secret(args.namespace, snapshot_dir)
    _upsert_gemini_secret(args.namespace, snapshot_dir)

    print()
    print(f"[k8s_submit] kubectl apply -f {rendered_path}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", str(rendered_path), "-n", args.namespace],
        check=False,
    )
    if result.returncode != 0:
        print(f"[k8s_submit] ERROR: kubectl apply failed (exit {result.returncode})", file=sys.stderr)
        return result.returncode

    print(f"[k8s_submit] Job submitted: {job_name}")
    print(f"[k8s_submit] Monitor with:")
    print(f"    kubectl get job {job_name} -n {args.namespace} -w")
    print(f"    kubectl logs -f job/{job_name} -n {args.namespace}")

    if args.tail:
        print()
        print(f"[k8s_submit] Tailing logs for job/{job_name} …")
        subprocess.run(
            ["kubectl", "logs", "-f", f"job/{job_name}", "-n", args.namespace],
            check=False,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
K8s job orchestration helpers used by run_pipeline.py to drive the on-prem
SFT and RL stages.

Responsibilities:
  - Render the templated job manifest (SFT or RL) by substituting __PLACEHOLDERS__.
  - kubectl apply / kubectl wait --for=condition=complete (or --for=condition=failed).
  - Stream pod logs with `kubectl logs -f` while the job runs so the user
    sees training output in their terminal in real time.
  - Read the artifact URI written by the pod into the artifacts PVC, by
    `kubectl cp`-ing it out to the local snapshot dir.

Assumptions:
  - The local kubectl is configured to talk to the on-prem cluster via
    KUBECONFIG=/Users/ktan/.kube/config-cwb607-ray (set by the caller or
    inherited from the env).
  - All Jobs run in the `tau2` namespace.
  - Jobs write their LoRA URI to /artifacts/.<sft|rl>_lora_artifact_uri-<suffix>
    on the tau2-artifacts PVC.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

NAMESPACE = "tau2"
KUBECONFIG_DEFAULT = "/Users/ktan/.kube/config-cwb607-ray"


def _kubectl_env() -> dict:
    env = os.environ.copy()
    env.setdefault("KUBECONFIG", KUBECONFIG_DEFAULT)
    return env


def _run(cmd: list[str], *, check: bool = True, capture: bool = False, **kw) -> subprocess.CompletedProcess:
    """Run a kubectl command, defaulting KUBECONFIG into the env."""
    env = _kubectl_env()
    env.update(kw.pop("env", {}) or {})
    if capture:
        return subprocess.run(cmd, env=env, check=check, capture_output=True, text=True, **kw)
    return subprocess.run(cmd, env=env, check=check, **kw)


def render_manifest(template_path: Path, out_path: Path, substitutions: dict[str, str]) -> Path:
    """Substitute __KEY__ placeholders in a templated YAML manifest."""
    text = Path(template_path).read_text()
    for key, value in substitutions.items():
        token = f"__{key}__"
        if token not in text:
            raise KeyError(f"placeholder {token!r} not found in {template_path}")
        text = text.replace(token, value)
    # Sanity: any remaining __FOO__ tokens are bugs.
    leftover = [
        line for line in text.splitlines()
        if "__" in line and any(seg.isupper() for seg in line.split("__"))
        and "__" + line.split("__")[1] + "__" not in substitutions
    ]
    # The above heuristic is loose; the real check is: do we still have any
    # placeholder tokens of the form __ALL_CAPS__ ?
    import re
    remaining = re.findall(r"__[A-Z][A-Z0-9_]+__", text)
    if remaining:
        raise KeyError(
            f"placeholders not substituted in {template_path}: {sorted(set(remaining))}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


def apply(manifest_path: Path) -> None:
    print(f"[k8s] kubectl apply -f {manifest_path}")
    _run(["kubectl", "-n", NAMESPACE, "apply", "-f", str(manifest_path)])


def wait_for_pod_ready(job_name: str, *, timeout_seconds: int = 1800) -> str:
    """Wait until the job's first pod is in Running or has terminated. Returns pod name."""
    print(f"[k8s] waiting for pod of job/{job_name} to start (timeout {timeout_seconds}s)")
    deadline = time.time() + timeout_seconds
    last_pod = ""
    while time.time() < deadline:
        result = _run(
            ["kubectl", "-n", NAMESPACE, "get", "pod",
             "-l", f"job-name={job_name}",
             "-o", "jsonpath={range .items[*]}{.metadata.name}|{.status.phase}{'\\n'}{end}"],
            capture=True,
        )
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            name, _, phase = line.partition("|")
            last_pod = name
            if phase in {"Running", "Succeeded", "Failed"}:
                print(f"[k8s] pod {name} phase={phase}")
                return name
        time.sleep(5)
    raise TimeoutError(f"pod for job/{job_name} did not start within {timeout_seconds}s")


def stream_logs(pod_name: str) -> None:
    """Tail logs from the pod until the container exits.

    Use `kubectl logs -f` directly so the user sees training output in real
    time. Returns when the stream closes (i.e. the container terminated).
    """
    print(f"[k8s] streaming logs from pod {pod_name} (Ctrl-C to detach without killing the job)")
    cmd = ["kubectl", "-n", NAMESPACE, "logs", "-f", pod_name, "-c", "trainer"]
    # Don't check return code -- if pod failed we still want to read condition below.
    _run(cmd, check=False)


def wait_for_job_done(job_name: str, *, timeout: str = "24h") -> str:
    """Block on `kubectl wait` for either Complete or Failed; return the condition."""
    print(f"[k8s] waiting for job/{job_name} to reach a terminal state (timeout {timeout})")
    # `kubectl wait --for=condition=complete OR failed` isn't supported; do two waits in parallel via `--for=jsonpath`.
    cmd_complete = [
        "kubectl", "-n", NAMESPACE, "wait", f"job/{job_name}",
        "--for=condition=complete", f"--timeout={timeout}",
    ]
    proc = subprocess.Popen(cmd_complete, env=_kubectl_env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    rc = proc.wait()
    if rc == 0:
        return "Complete"
    # Likely failed -- confirm explicitly so we surface a clean error to the caller.
    failed = _run(
        ["kubectl", "-n", NAMESPACE, "get", "job", job_name,
         "-o", "jsonpath={.status.conditions[?(@.type=='Failed')].status}"],
        capture=True, check=False,
    )
    if failed.stdout.strip() == "True":
        return "Failed"
    return "Unknown"


def cp_from_pod(pod_name: str, src_path: str, dst_path: Path) -> bool:
    """`kubectl cp <pod>:<src> <dst>`. Returns True iff the file was copied."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["kubectl", "-n", NAMESPACE, "cp",
           f"{pod_name}:{src_path}", str(dst_path), "-c", "trainer"]
    result = _run(cmd, check=False, capture=True)
    if result.returncode != 0:
        print(f"[k8s] kubectl cp failed: {result.stderr.strip()}")
        return False
    return dst_path.exists()


def read_uri_from_pod(pod_name: str, suffix: str, kind: str, snapshot_dir: Path) -> str | None:
    """Pull the artifact URI marker file the pod wrote to /artifacts.

    Returns the URI string, or None if the file isn't present.
    """
    if kind not in {"sft", "rl"}:
        raise ValueError(f"unexpected kind: {kind}")
    pod_path = f"/artifacts/.{kind}_lora_artifact_uri-{suffix}"
    local_path = snapshot_dir / f".{kind}_lora_artifact_uri"
    if cp_from_pod(pod_name, pod_path, local_path):
        uri = local_path.read_text().strip()
        return uri or None
    return None


def submit_and_wait(
    *,
    template_path: Path,
    rendered_path: Path,
    substitutions: dict[str, str],
    job_name: str,
    suffix: str,
    kind: str,
    snapshot_dir: Path,
    pod_ready_timeout_s: int = 1800,
    job_done_timeout: str = "24h",
) -> str:
    """End-to-end: render, apply, wait for pod, stream logs, wait for done, fetch URI."""
    render_manifest(template_path, rendered_path, substitutions)
    apply(rendered_path)
    pod_name = wait_for_pod_ready(job_name, timeout_seconds=pod_ready_timeout_s)
    stream_logs(pod_name)
    condition = wait_for_job_done(job_name, timeout=job_done_timeout)
    if condition == "Failed":
        raise RuntimeError(
            f"job/{job_name} reported Failed. Inspect with "
            f"`kubectl -n {NAMESPACE} describe job/{job_name}` and "
            f"`kubectl -n {NAMESPACE} logs {pod_name} -c trainer`."
        )

    uri = read_uri_from_pod(pod_name, suffix, kind, snapshot_dir)
    if not uri:
        raise RuntimeError(
            f"job/{job_name} reached {condition} but no /artifacts/.{kind}_lora_artifact_uri-{suffix} "
            "marker file was found. Did the post-train upload step run?"
        )
    print(f"[k8s] {kind} artifact URI: {uri}")
    return uri

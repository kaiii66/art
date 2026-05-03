"""
K8s job orchestration helpers used by run_pipeline.py to drive the on-prem
SFT and RL stages.

Responsibilities:
  - Render the templated job manifest (SFT or RL) by substituting __PLACEHOLDERS__.
  - kubectl apply / kubectl wait --for=condition=complete (or --for=condition=failed).
  - Stream pod logs with `kubectl logs -f` while the job runs so the user
    sees training output in their terminal in real time.
  - Read the artifact URI the trainer wrote to the tau2-artifacts PVC by
    spawning a tiny ephemeral busybox pod that mounts the same PVC and
    `cat`s the marker file (we read the result via `kubectl logs`).

Assumptions:
  - The local kubectl is configured to talk to the on-prem cluster via
    KUBECONFIG=/Users/ktan/.kube/config-cwb607-ray (set by the caller or
    inherited from the env).
  - All Jobs run in the `tau2` namespace.
  - Jobs write their LoRA URI to /artifacts/.<sft|rl>_lora_artifact_uri-<suffix>
    on the tau2-artifacts PVC.

Why an ephemeral reader pod instead of `kubectl cp`/`exec`:
  Both `cp` and `exec` require the source pod's container to still be
  running (cp is implemented as `tar | exec` under the hood). The trainer
  pod almost always reaches Completed before we get a chance to read the
  marker, and Kubernetes refuses `exec` against Completed pods with
  "cannot exec into a container in a completed pod". The PVC, however,
  outlives the pod -- so we just mount it again from a new pod for ~5s.
"""
from __future__ import annotations

import os
import subprocess
import textwrap
import time
import uuid
from pathlib import Path

NAMESPACE = "tau2"
KUBECONFIG_DEFAULT = "/Users/ktan/.kube/config-cwb607-ray"
ARTIFACTS_PVC = "tau2-artifacts"
# Small, ubiquitous image with `sh` and `cat`. Tag pinned for reproducibility;
# this only ever runs `cat` so any maintained busybox release works.
READER_IMAGE = "busybox:1.36"


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
    """`kubectl cp <pod>:<src> <dst>`. Returns True iff the file was copied.

    NOTE: This requires the source pod to still be Running -- `kubectl cp`
    is implemented as `tar | kubectl exec`, and `exec` is rejected on
    Completed/Failed pods. For reading marker files written by a finished
    Job, use `read_marker_from_pvc` instead.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["kubectl", "-n", NAMESPACE, "cp",
           f"{pod_name}:{src_path}", str(dst_path), "-c", "trainer"]
    result = _run(cmd, check=False, capture=True)
    if result.returncode != 0:
        print(f"[k8s] kubectl cp failed: {result.stderr.strip()}")
        return False
    return dst_path.exists()


def _reader_pod_manifest(pod_name: str, marker_path: str, pvc_name: str) -> str:
    """YAML for an ephemeral pod that cats one file off a PVC and exits.

    The pod is restartPolicy: Never, prints the file (or a sentinel) to
    stdout, then exits. We capture stdout via `kubectl logs`.

    A unique sentinel ('<<<MARKER_MISSING>>>') is used instead of an
    empty stdout so we can distinguish "file not found" from "file
    exists but is empty" -- both are bugs, but they call for different
    diagnostics.
    """
    return textwrap.dedent(f"""
        apiVersion: v1
        kind: Pod
        metadata:
          name: {pod_name}
          namespace: {NAMESPACE}
          labels:
            app.kubernetes.io/part-of: tau2-onprem
            app.kubernetes.io/component: marker-reader
        spec:
          restartPolicy: Never
          # Tolerate any common GPU-node taints so we can land on the same
          # nodes the trainer used (the PVC is RWX so this isn't strictly
          # required, but it avoids surprises on tightly-tainted clusters).
          tolerations:
            - operator: Exists
          containers:
            - name: reader
              image: {READER_IMAGE}
              imagePullPolicy: IfNotPresent
              command: ["sh", "-c"]
              args:
                - 'if [ -f "{marker_path}" ]; then cat "{marker_path}"; else echo "<<<MARKER_MISSING>>>"; fi'
              resources:
                requests: {{ cpu: "10m", memory: "16Mi" }}
                limits:   {{ cpu: "100m", memory: "64Mi" }}
              volumeMounts:
                - name: artifacts
                  mountPath: /artifacts
                  readOnly: true
          volumes:
            - name: artifacts
              persistentVolumeClaim:
                claimName: {pvc_name}
                readOnly: true
        """).lstrip()


def read_marker_from_pvc(
    marker_path: str,
    *,
    pvc_name: str = ARTIFACTS_PVC,
    timeout_seconds: int = 120,
) -> str | None:
    """Read a single text file off a PVC by spawning a one-shot reader pod.

    Returns the file contents stripped of trailing whitespace, or None
    if the file does not exist on the PVC.

    The reader pod is always deleted, even on error.
    """
    pod_name = f"marker-reader-{uuid.uuid4().hex[:8]}"
    manifest = _reader_pod_manifest(pod_name, marker_path, pvc_name)
    print(f"[k8s] spawning marker-reader pod {pod_name} to read {marker_path} from PVC {pvc_name}")
    try:
        # apply via stdin so we don't have to write a temp file
        proc = subprocess.run(
            ["kubectl", "-n", NAMESPACE, "apply", "-f", "-"],
            input=manifest, text=True, env=_kubectl_env(),
            capture_output=True, check=False,
        )
        if proc.returncode != 0:
            print(f"[k8s] failed to create reader pod: {proc.stderr.strip()}")
            return None

        # Wait for the pod to finish (Succeeded or Failed). We don't use
        # `kubectl wait --for=condition=Ready` because the container exits
        # so quickly it may never be observed Ready -- it goes straight
        # from Pending to Succeeded.
        deadline = time.time() + timeout_seconds
        terminal_phase = None
        while time.time() < deadline:
            phase_proc = _run(
                ["kubectl", "-n", NAMESPACE, "get", "pod", pod_name,
                 "-o", "jsonpath={.status.phase}"],
                capture=True, check=False,
            )
            phase = phase_proc.stdout.strip()
            if phase in {"Succeeded", "Failed"}:
                terminal_phase = phase
                break
            time.sleep(1)
        if terminal_phase is None:
            print(f"[k8s] reader pod {pod_name} did not finish within {timeout_seconds}s")
            return None

        logs = _run(
            ["kubectl", "-n", NAMESPACE, "logs", pod_name],
            capture=True, check=False,
        )
        if logs.returncode != 0:
            print(f"[k8s] kubectl logs {pod_name} failed: {logs.stderr.strip()}")
            return None
        out = logs.stdout.strip()
        if out == "<<<MARKER_MISSING>>>":
            print(f"[k8s] marker {marker_path} not found on PVC {pvc_name}")
            return None
        if not out:
            print(f"[k8s] marker {marker_path} exists but is empty")
            return None
        return out
    finally:
        # Best-effort cleanup. Wait=false so we don't block on graceful term.
        _run(
            ["kubectl", "-n", NAMESPACE, "delete", "pod", pod_name,
             "--ignore-not-found", "--wait=false"],
            check=False, capture=True,
        )


def read_uri_from_pod(pod_name: str, suffix: str, kind: str, snapshot_dir: Path) -> str | None:
    """Read the artifact URI marker file the trainer wrote to /artifacts.

    `pod_name` is accepted for backwards compatibility / diagnostics but is
    no longer used for the read path: by the time we call this, the trainer
    pod is almost always Completed and `kubectl cp`/`exec` would fail with
    "cannot exec into a container in a completed pod". We mount the PVC
    via an ephemeral reader pod instead. The result is also cached to
    `snapshot_dir/.{kind}_lora_artifact_uri` so reruns of the orchestrator
    can find it without round-tripping to the cluster.

    Returns the URI string, or None if the marker file isn't present on
    the PVC.
    """
    if kind not in {"sft", "rl"}:
        raise ValueError(f"unexpected kind: {kind}")
    marker_path = f"/artifacts/.{kind}_lora_artifact_uri-{suffix}"
    uri = read_marker_from_pvc(marker_path)
    if uri:
        local_path = snapshot_dir / f".{kind}_lora_artifact_uri"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(uri + "\n")
    return uri


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

#!/usr/bin/env bash
# Create the K8s Secrets the tau2 on-prem Jobs need.
#
# Reads from .env at the repo root. Required keys:
#   WANDB_API_KEY    -> Secret/wandb (key=api)
#   HF_TOKEN         -> Secret/hf    (key=token)
#   RLLM_API_KEY     -> Secret/rllmui (key=api)         [for cloud rllm-ui]
#   GHCR_USER + GHCR_TOKEN -> Secret/ghcr (docker-registry)  [optional, only if
#                            you make the GHCR images private]
#
# Idempotent: each secret is recreated if it already exists.
#
# Usage:
#   export KUBECONFIG=/Users/ktan/.kube/config-cwb607-ray
#   bash onprem/k8s/setup_secrets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
NAMESPACE="${NAMESPACE:-tau2}"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found." >&2
  echo "Create it with at least WANDB_API_KEY, HF_TOKEN, RLLM_API_KEY." >&2
  exit 1
fi

# Source the .env file. Keys with `=` in the value would need shell-quoting in
# the .env; for the API keys we use this is fine.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [ -z "${KUBECONFIG:-}" ]; then
  echo "WARNING: KUBECONFIG is not set. Defaulting to ~/.kube/config-cwb607-ray." >&2
  export KUBECONFIG="$HOME/.kube/config-cwb607-ray"
fi

echo "==> namespace: $NAMESPACE"
kubectl apply -f "$REPO_ROOT/onprem/k8s/namespace.yaml" --validate=false

# ---- helpers ----
recreate_generic() {
  local name="$1"
  local key="$2"
  local value="$3"
  if [ -z "$value" ]; then
    echo "  [skip] $name (empty value in .env)"
    return
  fi
  kubectl -n "$NAMESPACE" delete secret "$name" --ignore-not-found >/dev/null
  kubectl -n "$NAMESPACE" create secret generic "$name" \
    --from-literal="$key=$value" >/dev/null
  echo "  [ok]   $name (key=$key)"
}

recreate_docker_registry() {
  local name="$1"
  local server="$2"
  local user="$3"
  local pass="$4"
  if [ -z "$user" ] || [ -z "$pass" ]; then
    echo "  [skip] $name (GHCR_USER / GHCR_TOKEN not set; only needed for private images)"
    return
  fi
  kubectl -n "$NAMESPACE" delete secret "$name" --ignore-not-found >/dev/null
  kubectl -n "$NAMESPACE" create secret docker-registry "$name" \
    --docker-server="$server" \
    --docker-username="$user" \
    --docker-password="$pass" >/dev/null
  echo "  [ok]   $name (docker-registry, server=$server)"
}

# ---- create secrets ----
echo "==> secrets in namespace $NAMESPACE"
recreate_generic "wandb"  "api"   "${WANDB_API_KEY:-}"
recreate_generic "hf"     "token" "${HF_TOKEN:-}"
recreate_generic "rllmui" "api"   "${RLLM_API_KEY:-}"
recreate_docker_registry "ghcr" "ghcr.io" "${GHCR_USER:-}" "${GHCR_TOKEN:-}"

# ---- PVCs ----
echo "==> persistent volume claims"
kubectl apply -f "$REPO_ROOT/onprem/k8s/pvcs.yaml" --validate=false

# ---- summary ----
echo
echo "==> verification"
kubectl -n "$NAMESPACE" get secrets,pvc -o wide

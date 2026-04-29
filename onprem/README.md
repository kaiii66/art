# tau2 on-prem training stack

Replaces the OpenPipe ART `ServerlessBackend` with a local 8xH100 K8s cluster
(`cwb607-ray`) plus W&B Inference for serving the resulting LoRA adapter.

```
              +-----------------------------------+
              |   run_pipeline.py --backend onprem |
              +-----------------------------------+
                              |
   +--------------------------+--------------------------+
   | upload   prepare-sft         sft-job        rl-job        leaderboard
   | (local)  (local CPU)         (K8s 8xH100)   (K8s 8xH100)  (local)
   |          ^                   |              |              |
   |   teacher rollouts            Axolotl LoRA   rLLM/verl GRPO   Tau2 eval
   |   -> JSONL artifact           -> /artifacts  -> /artifacts    via W&B Inference
   +--------------------------+   ----+  ----+   ----+  ----+
                                  publish .lora    publish .lora
                                  -> W&B artifact  -> W&B artifact
                                  type=lora        type=lora
```

## Layout

```
onprem/
├── docker/
│   ├── Dockerfile.sft          # axolotl base + light helpers
│   └── Dockerfile.rl           # CUDA + vLLM + verl + rllm + tau2 + repo
├── k8s/
│   ├── namespace.yaml          # Namespace `tau2`
│   ├── pvcs.yaml               # tau2-data + tau2-artifacts PVCs
│   ├── setup_secrets.sh        # creates wandb / hf / rllmui / ghcr Secrets
│   ├── sft_job.yaml            # templated K8s Job for SFT
│   └── rl_job.yaml             # templated K8s Job for RL
├── configs/
│   ├── axolotl_sft.yaml        # full Axolotl recipe (LoRA r=16, FSDP)
│   └── rllm_train_config.yaml  # rLLM AgentTrainer recipe
├── scripts/                    # things that run inside the containers
│   ├── trajectory_to_jsonl.py
│   ├── prepare_sft_data.py     # local: Phase A teacher rollouts -> dataset artifact
│   ├── tau2_rllm_rollout.py    # the @rllm.rollout function
│   ├── rllm_train_tau2.py      # RL trainer entrypoint (in-pod)
│   ├── wandb_lora_upload.py    # uploads PEFT dir as type=lora artifact
│   ├── run_axolotl_sft.sh      # SFT pod entrypoint
│   └── run_rllm_rl.sh          # RL pod entrypoint
└── pipeline/
    └── k8s_runner.py           # render manifest, kubectl apply / wait / cp
```

## One-time setup

1. Add to `.env` at the repo root (alongside the existing `WANDB_API_KEY`):

   ```
   WANDB_API_KEY=...
   WANDB_ENTITY=...                # your W&B team / username
   HF_TOKEN=...                    # for HF model downloads
   RLLM_API_KEY=...                # `rllm login` -> ui.rllm-project.com
   GHCR_USER=...                   # your GitHub username (owner of the GHCR images)
   GHCR_TOKEN=...                  # PAT with write:packages (only needed if images are private)
   ```

2. Verify cluster access:

   ```bash
   export KUBECONFIG=/Users/ktan/.kube/config-cwb607-ray
   kubectl get nodes -o wide
   kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'   # expect 8
   ```

3. Provision Namespace + Secrets + PVCs:

   ```bash
   bash onprem/k8s/setup_secrets.sh
   ```

4. Build + push the two images (from repo root):

   ```bash
   TAG=$(git rev-parse --short HEAD)
   docker buildx build --platform linux/amd64 \
     -t ghcr.io/$GHCR_USER/tau2-sft:$TAG \
     -f onprem/docker/Dockerfile.sft --push .
   docker buildx build --platform linux/amd64 \
     -t ghcr.io/$GHCR_USER/tau2-rllm:$TAG \
     -f onprem/docker/Dockerfile.rl  --push .
   ```

   (Mac users: cross-arch build is slow on first run. After the base layers
   cache, incremental pushes only ship the small app layer.)

## Running the pipeline

```bash
# Full pipeline (snapshot -> upload -> sft -> rl -> leaderboard)
uv run python run_pipeline.py --backend onprem --image-tag $(git rev-parse --short HEAD)

# Re-run leaderboard alone against an existing snapshot
uv run python run_pipeline.py --backend onprem --resume pipeline_runs/04270927 --skip upload sft rl

# Dry-run: render manifests + print stage commands without executing
uv run python run_pipeline.py --backend onprem --dry-run
```

The serverless ART path is unchanged — `--backend serverless` (default)
still calls the original `train_tau2_distill.py` and `train_tau2.py`.

## Observability

- **W&B**: project `tau2-ART-autoresearch-telecom`, group
  `pipeline-<MMDDHHMM>` (one bundle per pipeline run, same convention as
  the existing serverless flow).
- **rllm-ui (cloud)**: `ui.rllm-project.com`. Enabled via `RLLM_API_KEY` env
  var in the RL pod. Click into a session to see per-episode reward
  breakdowns and full assistant/user/tool message traces step by step.
- **kubectl**:
  ```bash
  kubectl -n tau2 get jobs,pods
  kubectl -n tau2 logs -f $(kubectl -n tau2 get pod -l job-name=tau2-rl-04270927 -o name)
  ```

## What gets published to W&B Inference

Each successful run creates two `type=lora` artifacts in the W&B project:

```
wandb-artifact:///<entity>/tau2-ART-autoresearch-telecom/tau2-sft-Qwen3-30B-A3B-Instruct-2507-<suffix>:vN
wandb-artifact:///<entity>/tau2-ART-autoresearch-telecom/tau2-rl-Qwen3-30B-A3B-Instruct-2507-<suffix>:vN
```

Both carry `metadata.wandb.base_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"` and
live in `storage_region="coreweave-us"`. Hit them like any other model:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=WANDB_API_KEY,
    project=f"{WB_TEAM}/{WB_PROJECT}",
)
resp = client.chat.completions.create(
    model="wandb-artifact:///team/project/tau2-rl-...:latest",
    messages=[{"role": "user", "content": "Hello"}],
)
```

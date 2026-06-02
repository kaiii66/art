# $\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment

[![python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![arXiv](http://img.shields.io/badge/cs.AI-arXiv%3A2506.07982-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2506.07982)
[![blog](https://img.shields.io/badge/blog-tau2--bench-green)](https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/sierra.svg?style=social&label=Follow%20%40SierraPlatform)](https://x.com/SierraPlatform/status/1932464265207889974)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/posts/sierra_last-year-we-introduced-%F0%9D%9C%8F-bench-a-benchmark-activity-7338229693898231809-F8L4?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAdc8goBmhEsiEo1_t_XSJbAnY4_zMfAWcE)
[![Leaderboard](https://img.shields.io/badge/🏆_Live_Leaderboard-taubench.com-brightgreen?style=flat)](https://taubench.com)

<div align="center">
<img src="figs/overview.png" width="95%" alt="System Overview"><br>
<em>Figure 1: τ²-bench allows users to interact with the agent and the environment</em>
</div>

<div align="center">
<img src="figs/traj.png" width="95%" alt="Trajectory"><br>
<em>Figure 2: Trajectory of a conversation between an agent and a user</em>
</div>

## Quick Start

1. **Create and activate environment, install package:**
   ```bash
   uv venv .venv
   source .venv/bin/activate   
   uv pip install -e .
   ```

2. **Set API keys** — Copy `.env.example` to `.env` and set `WANDB_API_KEY` (and any LLM keys you need). Get your key at [wandb.ai/authorize](https://wandb.ai/authorize).

3. **Upload datasets to W&B** (train/val/base splits for your config):
   ```bash
   python upload_dataset_to_wandb.py
   ```

4. **Train** (GRPO via ART serverless backend):
   ```bash
   python train_tau2.py
   ```

5. **Build leaderboard** (evaluate base + trained, then publish):
   ```bash
   python create_leaderboard.py --models all --publish-leaderboard
   ```

Run in that order. For more options, see each script's `--help` or docstring.

> The Quick Start above uses ART's W&B serverless backend. For the on-prem
> GRPO pipeline (Kubernetes + 8×H100 + KL-anchored LoRA training), see the
> next section.

## End-to-End Pipeline: SFT → RL → Leaderboard (on-prem GPU)

This repo includes a full on-prem RL pipeline that takes a teacher-distilled
SFT checkpoint, runs KL-anchored GRPO training on an 8×H100 node, and
publishes a Weave leaderboard comparing **base / SFT / RL** rows (plus
optional frontier-baseline rows for gpt-4.1-mini and Gemini).

```
[teacher trajectories] ──► [SFT (serverless)] ──► SFT LoRA on W&B
                                                          │
                                                          ▼
          ┌──────────── local/run_pipeline_local.py ────────────┐
          │ pull_sft → rl (LocalBackend) → upload_rl → leaderboard │
          └────────────────────────────────────────────────────────┘
                                                          │
                                                          ▼
                                              Weave leaderboard
                                         (base / SFT / RL / frontier rows)
```

The pipeline supports two GPU submission backends:

| Backend | Flag | When to use |
|---------|------|-------------|
| **SUNK / Slurm** | `--slurm` | Clusters running [SUNK](https://docs.coreweave.com/products/sunk) (Slurm-on-Kubernetes) — current production path |
| **Vanilla Kubernetes** | _(default)_ | Plain K8s clusters with a `tau2` namespace and GPU nodes |

> **Full step-by-step instructions, known-issue playbook, and monitoring commands:**
> **[`local/RUNBOOK.md`](local/RUNBOOK.md)**

Architecture write-up: [`local/RUN_SUMMARY.md`](local/RUN_SUMMARY.md).

### Prerequisites

```bash
# 1. Clone + venv (per Quick Start above)
git clone <this-repo> && cd tau2-bench
uv venv .venv && source .venv/bin/activate && uv pip install -e .

# 2. .env (in repo root)
WANDB_API_KEY=wandb_v1_...     # personal W&B token
HF_TOKEN=hf_...                # for downloading Qwen3-30B-A3B-Instruct-2507
WANDB_ENTITY=kwt
GHCR_USER=<gh-user>            # GitHub Container Registry username
OPENAI_API_KEY=sk-...          # optional — adds gpt-4.1-mini leaderboard row
GEMINI_API_KEY=...             # optional — adds Gemini leaderboard row

# 3. Docker daemon + GHCR auth
# Must write a base64 auth entry (not a credsStore pointer) so enroot can pull
# the image on SUNK compute nodes without a credential helper.
echo "$CR_PAT" | docker login ghcr.io -u <gh-user> --password-stdin

# 4. Kubernetes namespace + PVCs + secrets (one-time bootstrap)
NS=tau2
kubectl create namespace "$NS" 2>/dev/null || true

# PVCs — adjust storageClassName + sizes for your cluster.
# 500 Gi covers a persistent HF cache for Qwen3-30B + multiple RL checkpoints;
# 100 Gi holds intermediate data the pipeline streams in/out.
kubectl apply -n "$NS" -f - <<'YAML'
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: tau2-artifacts }
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: shared-vast      # change to your cluster's class
  resources: { requests: { storage: 500Gi } }
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: tau2-data }
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: shared-vast
  resources: { requests: { storage: 100Gi } }
YAML

# GHCR image-pull secret (so the Job can pull ghcr.io/<gh-user>/tau2-art:...)
kubectl create secret docker-registry ghcr -n "$NS" \
    --docker-server=ghcr.io \
    --docker-username=<gh-user> \
    --docker-password="$CR_PAT"

# W&B API key — pipeline expects key `api`
kubectl create secret generic wandb -n "$NS" \
    --from-literal=api="$WANDB_API_KEY"

# HuggingFace token — pipeline expects key `token`
kubectl create secret generic hf -n "$NS" \
    --from-literal=token="$HF_TOKEN"

# Sanity-check everything is in place
kubectl get pvc,secret -n "$NS"
# Expect: pvc/tau2-artifacts, pvc/tau2-data, secret/ghcr, secret/wandb, secret/hf
```

### Quick (single command)

After the one-time bootstrap (PVCs + secrets + `.env`), the whole pipeline
runs from a single entry point. Set `KUBECONFIG` to match your cluster first.

```bash
# ── SUNK / Slurm (current production path) ────────────────────────────────
export KUBECONFIG=/home/coder/.kube/training
SUFFIX=$(TZ=America/Los_Angeles date +%m%d%H%M)
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --tail \
    2>&1 | tee pipeline_runs/full_run_$SUFFIX.log
```

```bash
# ── Vanilla Kubernetes ─────────────────────────────────────────────────────
export KUBECONFIG=/home/coder/.kube/config-cwb607-ray
SUFFIX=$(TZ=America/Los_Angeles date +%m%d%H%M)
uv run python run_full_pipeline.py --suffix $SUFFIX --tail \
    2>&1 | tee pipeline_runs/full_run_$SUFFIX.log
```

Both commands run all stages end-to-end (~4–6 h): SFT → patch config →
docker build/push → submit GPU job → pull_sft → RL → upload → leaderboard.
The suffix is auto-generated as `MMDDHHMM` (US/Pacific) and shared across all
stages so cross-stage state stays correlated. A snapshot copy of every
patched config lives at `pipeline_runs/<SUFFIX>/` for the audit trail.

**Resume after failure** (reuse the same `$SUFFIX`):
```bash
# SFT done — fix RL/code, rebuild image, resubmit:
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --skip-sft --tail

# SFT done + image correct — resubmit only:
uv run python run_full_pipeline.py --slurm --suffix $SUFFIX --skip-sft --skip-build --tail
```

**Key flags:**

| Flag | Effect |
|---|---|
| `--slurm` | Submit via Slurm (`sbatch` through the SUNK login pod) instead of `kubectl apply`. |
| `--suffix MMDDHHMM` | Reuse a prior suffix (e.g. to retry after a failure). |
| `--skip-sft` | Skip SFT — requires `--suffix` pointing at an existing snapshot. |
| `--skip-build` | Skip docker build + push; reuse the image tag from a prior submit. |
| `--smoke` | 4-task RL only; skips upload_rl + leaderboard. For debugging only. |
| `--tail` | Stream job logs until the job ends. |
| `--slurm-time-limit HH:MM:SS` | Slurm wall-clock limit (default `08:00:00`; use `16:00:00` for safety). |
| `--nfs-base PATH` | NFS base path on SUNK cluster (default `/mnt/data/kai`). |

> See **[`local/RUNBOOK.md`](local/RUNBOOK.md)** for the full flag reference,
> cluster bootstrap steps, monitoring commands, and known-issue playbook.

The rest of this section (Steps 1–5) is the canonical manual flow for
running the pipeline stages independently.

### Step 1 — SFT (skip if you already have an SFT LoRA in W&B)

Uploads the train/val datasets to W&B, then trains a LoRA on top of the base
model using teacher-distilled trajectories. Output: a W&B LoRA artifact at
`kwt/<project>/<collection>:step{N}`.

```bash
# Runs upload → sft, stops before the rl + leaderboard stages.
uv run python run_pipeline.py --skip rl leaderboard
```

When it finishes, the SFT collection name is written to
`pipeline_runs/<MMDDHHMM>/.last_trained_model`. You'll plug that into
`train_config_local.yaml > sft_source.name` in Step 2.

### Step 2 — Configure the RL run

Edit `train_config_local.yaml` so `sft_source` points at the artifact from
Step 1:

```yaml
project: "tau2-ART-distill-05111101"
base_model: "Qwen/Qwen3-30B-A3B-Instruct-2507"

sft_source:
  entity: "kwt"
  project: "tau2-ART-distill-05111101"
  name:   "tau2-distill-Qwen3-30B-A3B-Instruct-2507-20260511-1101"
  step:   "latest"   # or a specific int

kl_penalty_coef: 0.04            # KL anchor against SFT
learning_rate:   1.0e-7
groups_per_step: 1
rollouts_per_group: 16
```

Leave the other hyperparameters at their defaults — they are the validated
values from this work (see [`local/RUN_SUMMARY.md`](local/RUN_SUMMARY.md) for the
rationale behind each one).

### Step 3 — Build & push the on-prem image

```bash
IMAGE=ghcr.io/<gh-user>/tau2-art:$(git rev-parse --short HEAD)
docker build --progress=plain -f onprem/Dockerfile.art-rl -t "$IMAGE" .
docker push "$IMAGE"
```

> ⚠️ kubelet caches images by tag. If you rebuild without bumping the git
> SHA (e.g. uncommitted changes), append a fresh suffix:
> `IMAGE=...:$(git rev-parse --short HEAD)-$(date +%s)`

### Step 4 — Smoke test (optional, ~30 min)

Runs 4 tasks, skips upload + leaderboard:

```bash
uv run python local/k8s_submit.py \
    --image-tag "$IMAGE" \
    --skip-stages "upload_rl leaderboard" \
    --num-tasks 4
```

Success means: `loss/kl_policy_ref` non-zero in the W&B run, `[best-step]`
line appears, pod exits `phase=Succeeded`.

### Step 5 — Full RL run (~3 h)

```bash
uv run python local/k8s_submit.py --image-tag "$IMAGE"

# Watch
kubectl get job  -n tau2 -l app.kubernetes.io/component=art-rl -w
kubectl logs -f job/tau2-art-rl-<MMDDHHMM> -n tau2
```

When you see `Leaderboard published: ObjectRef(…)` the pipeline is done.

### Outputs

| Artifact | Where |
|---|---|
| RL LoRA checkpoints | `kwt/<project>/<collection>-rl-<MMDDHHMM>:step{N}` on W&B |
| Best step | Printed as `[rl] best step: N best val/reward: X` |
| W&B run | `https://wandb.ai/kwt/<project>/runs/<id>` (link in log) |
| **Weave leaderboard** | `https://wandb.ai/kwt/<project>/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1` |

The leaderboard has rows for **base**, **SFT @ best step**, **RL @ best step**,
plus optional frontier-baseline rows (gpt-4.1-mini, Gemini) when the
corresponding API keys are in `.env`. Latest validated result
(`kwt/tau2-ART-distill-06011448`, 40-task test set, 3 trials each):

| Row | success.mean | task_reward.mean |
|---|---|---|
| base Qwen3-30B-A3B-Instruct-2507 | 5.8% | 0.141 |
| SFT @ step 12 | 46.7% | 0.639 |
| **RL @ step 15 (GRPO)** | **53.3%** | **0.668** (+6.6 pp over SFT) |

Weave leaderboard:
`https://wandb.ai/kwt/tau2-ART-distill-06011448/weave/leaderboards/tau2-telecom-leaderboard-shaped-v1`

## 🆕 What's New

### 🤖 Reinforcement Learning Support (New!)
τ²-bench now supports RL training with a Gymnasium-compatible interface:

- **🏋️ Train RL Agents**: Use the gym interface to train agents with popular RL frameworks. 
- **🎮 Play as Agent or User**: Interactive mode lets you control either the agent or the user in conversations
- **📊 Train/Test Splits**: To help support experiments around training Agents and evaluating them, all domains include standardized task splits for proper train/test evaluation.

> **⚠️ IMPORTANT FOR BACKWARD COMPATIBILITY**: If you are just evaluating an agent (not training), you **MUST** use the `base` task split to evaluate on the complete task set that matches the original τ²-bench structure. This ensures your results are comparable to previous evaluations and maintains consistency with the established benchmark. (If you don't specify a task split, it will default to `base`.)
- **🔧 Gymnasium Compatible**: Standard gym interface works with existing RL tools and libraries

[**→ See Gym Documentation**](src/tau2/gym/README.md) | [**→ Try CLI Play Mode**](#interactive-play-mode)

### 🏆 Live Leaderboard (v0.2.0)
The τ²-bench leaderboard is now live at **[taubench.com](https://taubench.com)**! 

- **📊 Interactive Rankings**: Compare model performance across all domains
- **📱 Mobile-Friendly**: View results on any device  
- **🔍 Detailed Analysis**: Explore trajectories and conversation flows
- **📥 Easy Submission**: Submit your results directly through the interface

[**→ Visit the Leaderboard**](https://taubench.com) | [**→ Submit Your Results**](#leaderboard-submission)

## Overview

$\tau^2$-bench implements a simulation framework for evaluating customer service agents across various domains.

**$\tau^2$-bench is the new iteration of the original $\tau$-bench**, featuring code fixes and an additional telecom domain.

Each domain specifies:
- a policy that the agent must follow
- a set of tools that the agent can use
- a set of tasks to evaluate the agent's performance
- Optionally: A set of tools that the user simulator can use

Domains are:
- `mock`
- `airline`
- `retail`
- `telecom`

All the information that an agent developer needs to build an agent for a domain can be accessed through the domain's API docs. See [View domain documentation](#view-domain-documentation) for more details.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

2. Create a new environment (optional)

$\tau^2$-bench requires Python 3.10 or higher. You may create and activate a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install tau2

```bash
pip install -e .
```

This will enable you to run the `tau2` command.

**Note:** If you use `pip install .` (without `-e`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:

```bash
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

**Check your data directory setup:**

After installation, you can verify that your data directory is correctly configured by running:

```bash
tau2 check-data
```

This command will check if the data directory exists and print instructions if it is missing.

To remove all the generated files and the virtual environment, run:
```bash
make clean
```

## Quick Start

### Setup LLM API keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

To provide your API keys, copy `.env.example` as `.env` and edit it to include your API keys.

### Run agent evaluation

To run a test evaluation on only 5 tasks with 1 trial per task, run:

```bash
tau2 run \ 
--domain airline \
--agent-llm gpt-4.1 \
--user-llm gpt-4.1 \
--num-trials 1 \
--num-tasks 5
```

Results will be saved in `data/tau2/simulations/`.

> **💡 Tip**: For full agent evaluation that matches the original τ²-bench methodology, remove `--num-tasks` and use `--task-split base` to evaluate on the complete task set.

## Command Line Interface

The `tau2` command provides a unified interface for all functionality:

### Running Benchmark 
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --task-ids <task_ids> \
  --max-concurrency <concurrent_sims> \
  ...
```

### Interactive Play Mode
```bash
tau2 play
```
Experience τ²-bench from either perspective! The play mode allows you to:
- **Play as Agent**: Manually control the agent's responses and tool calls
- **Play as User**: Control the user while an LLM agent handles requests (available in domains with user tools like telecom)
- **Understand tasks** by walking through scenarios step-by-step
- **Test strategies** before implementing them in code
- **Choose task splits** to practice on training data or test on held-out tasks

This is perfect for:
- Getting familiar with domain policies and tools from both perspectives
- Debugging task scenarios and conversation flows
- Developing intuition for agent strategies
- Testing user behavior and agent responses
- Training yourself before training your model!

See the [Gym Documentation](src/tau2/gym/README.md) for more details on using the gymnasium interface programmatically, including the `AgentGymEnv` (play as agent) and `UserGymEnv` (play as user).

### Viewing Results
```bash
tau2 view
```
This tool allows you to:
- Browse simulation files (in `data/tau2/simulations/`)
- View agent performance metrics
- View a particular simulation
- View task details

### View domain documentation
```bash
tau2 domain <domain>
```
Visit http://127.0.0.1:8004/redoc to see the domain policy and API documentation.

![domain_viewer1](figs/domain_viewer.png)

### Check data configuration
```bash
tau2 check-data
```
This command checks if your data directory is properly configured and all required files are present.

## Leaderboard Submission

To submit your agent results to the τ²-bench leaderboard, you need to prepare a valid submission package that meets specific requirements.

### Requirements for Valid Submissions

Your trajectory runs must follow these constraints:

1. **Complete domain coverage**: Include results for all three domains:
   - `retail`
   - `airline` 
   - `telecom`

2. **Consistent model configuration**: All trajectory files must use:
   - The same agent LLM with identical arguments across all domains
   - The same user simulator LLM with identical arguments across all domains

3. **One result per domain**: Each domain should appear exactly once in your submission

4. **All tasks completed**: Run evaluation on all tasks within each domain (don't use `--task-ids` or `--num-tasks` filters)

> **📝 Note**: For consistency with the original τ²-bench evaluation methodology, use the `base` task split when evaluating your agent to ensure you're testing on the complete, standard task set.

### Preparing Your Submission

#### Step 1: Run Evaluations
First, run your agent evaluation on all domains with consistent settings:

```bash
# Example: Run complete evaluation for all domains
tau2 run --domain retail --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_retail
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_airline  
tau2 run --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4 --save-to my_model_telecom
```

**Important**: Use identical `--agent-llm`, `--user-llm`, and their arguments across all runs.

#### Step 2: Prepare Submission Package
Use the submission preparation tool to create your leaderboard submission:

```bash
tau2 submit prepare data/tau2/simulations/my_model_*.json --output ./my_submission
```

This command will:
- Verify all trajectory files are valid
- Check that submission requirements are met
- Compute performance metrics (Pass^k rates)
- Prompt for required metadata (model name, organization, contact email)
- Create a structured submission directory with:
  - `submission.json`: Metadata and metrics
  - `trajectories/`: Your trajectory files

#### Step 3: Validate Your Submission
Before submitting, validate your submission package:

```bash
tau2 submit validate ./my_submission
```

This will verify:
- All required files are present
- Trajectory files are valid
- Domain coverage is complete
- Model configurations are consistent

### Additional Options

#### Skip Verification (if needed)
```bash
tau2 submit prepare data/tau2/simulations/my_model_*.json --output ./my_submission --no-verify
```

#### Verify Individual Trajectory Files
```bash
tau2 submit verify-trajs data/tau2/simulations/my_model_*.json
```

### Submitting to the Leaderboard

Once your submission package is prepared and validated:

1. Review the generated `submission.json` file
2. Follow the submission guidelines in [web/leaderboard/public/submissions/README.md](web/leaderboard/public/submissions/README.md) to create a Pull Request
3. Keep your `trajectories/` directory for reference

The leaderboard will display your model's Pass^k success rates (k=1,2,3,4) across all domains.

## Experiments

### Experimental Code Directory

The `@experiments/` directory contains experimental features and research code that extends beyond the core tau2 benchmark. This directory is designed for community contributions of innovative approaches, prototypes, and new features that are not part of the core evaluation framework.

- **Purpose**: Research code and experimental features
- **Location**: `src/experiments/`
- **Usage**: Each experimental component has its own README with documentation
- **Status**: Experimental code is provided as-is and may not be fully tested or supported

For more details, see the [experiments README](src/experiments/README.md).

### Running Ablation Studies (No User, or Agent with Oracle Plan)
`telecom` domain enables running ablation studies.

1. Running an LLM in `no-user` mode. In this mode, the LLM is given all the tools and the information upfront.
Just choose `llm_agent_solo` as the agent and `dummy_user` as the user.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_solo \
  --agent-llm gpt-4.1 \
  --user dummy_user \
  ...
```

2. Running an LLM in `oracle-plan` mode. In this mode, the LLM is given an oracle plan ahead of time alleviating the need for action planning.
Just choose `llm_agent_gt` as the agent.

```bash
tau2 run \
  --domain telecom \
  --agent llm_agent_gt \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

### Running Telecom Domain with Workflow Policy
To test the impact of policy format, we provide an additional "workflow" policy for the telecom domain.
To run using this policy, use the `telecom-workflow` domain.

```bash
tau2 run \
  --domain telecom-workflow \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  ...
```

## Domains

For all the details see the domains [README](src/tau2/domains/README.md).

### Basics

- Code is located in `src/tau2/domains/`
- Data is located in `data/tau2/domains/`
- Each domain has its own configuration and task definitions

#### View domain-specific policy and API docs:
Run the following command to see the domain policy and API documentation.
```bash
tau2 env <domain>
```

Then visit http://127.0.0.1:8004/redoc

### Environment CLI (beta)

An interactive command-line interface for directly querying and testing domain environments. Features:
- Interactive query interface with domain-specific tools
- Support for multiple domains (airline, mock, etc.)
- Session management with history

To use:
```bash
make env-cli
```

Available commands:
- `:q` - quit the program
- `:d` - change domain
- `:n` - start new session (clears history)

Example usage:
```bash
$ make env-cli

Welcome to the Environment CLI!
Connected to airline domain.

Query (:n new session, :d change domain, :q quit)> What flights are available from SF to LA tomorrow?
Assistant: Let me check the flight availability for you...
[Flight details will appear here]
```

The Environment CLI is useful for:
- Testing domain tools and queries
- Debugging environment responses
- Exploring available domain functionality
- Quick domain interaction without starting the full server stack


## Run tests
To run the test suite use the command

```sh
make test
```

## Config

To configure the framework, see the [config](src/tau2/config.py) file.

### LLM Calls caching
LLM call caching is disabled by default.

To enable LLM calls caching:
    - Make sure `redis` is running.
    - Update the redis config in `config.py` if necessary.
    - Set `LLM_CACHE_ENABLED` to `True` in `config.py`


## Evaluate Your Own Agent
For local or remote agent evaluation, see our [agent developer guide](src/tau2/agent/README.md).

## Contributing

We welcome contributions to τ²-bench! Whether you're fixing bugs, adding new features, creating new domains, or contributing experimental research code, please see our [Contributing Guide](CONTRIBUTING.md) for detailed guidelines on:

- **Opening issues** before starting work
- **Branch naming conventions** and development workflow  
- **Code quality standards** and testing requirements
- **Pull request guidelines** for clean, reviewable contributions
- **Domain and experimental contributions** specific guidelines

For experimental features and research code, check out the [`@experiments/`](src/experiments/) directory.

## Orchestration Sequence Diagram

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant A as Agent
    participant U as UserSimulator
    participant E as Environment

    Note over O: Initialize(task)
    rect rgb(100, 150, 150)
        O->>A: get_init_state_info(message_history)
        A->>O: agent_state_info
        O->>U: get_init_state_info(message_history)
        U->>O: user_state_info
        O->>E: set_state(initialization_data, initialization_actions, message_history)
    end
    Note over O: Start simulation
    loop Pass messages between Agent, User, and Environment

        alt Agent/Env to User
            rect rgb(200, 150, 150)
            O->>U: generate_next_message(msg, user_state_info)
            U-->>O: (user_msg, user_state_info)
            end
            Note over O: Check if user_msg is STOP
        else User/Env to Agent
            rect rgb(100, 200, 100)
            O->>A: generate_next_message(msg, agent_state_info)
            A-->>O: (assistant_msg, agent_state_info)
            Note over O: Check if too many errors
            end
        else User/Agent to Environment
            rect rgb(150, 150, 200)
            O->>E: get_response(tool_call)
            E-->>O: tool_message
            end
        end
        Note over O: Check if max turns reached.
    end
    Note over O: Return simulation run
```

## Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment}, 
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982}, 
}
```

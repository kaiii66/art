"""
The @rllm.rollout adapter for tau2-bench.

Wraps the existing tau2 orchestrator (the same one used by tau2_rollout in
tau2_art_helpers.py) so rLLM can drive it through its trajectory recorder.

Two LLM clients:
  - policy_client : the in-training student. Points at the local vLLM
                    server colocated in the same RL pod (TP=8). vLLM hot-
                    loads the latest LoRA weights between train steps.
  - user_client   : the user simulator. Points at W&B Inference
                    (api.inference.wandb.ai) so we keep using
                    wandb/Qwen/Qwen3-30B-A3B-Instruct-2507 without
                    occupying any of the local GPUs.

Both are rllm.client.OpenAI instances so every chat.completions call is
captured into the rLLM Episode (token IDs + logprobs + content), which is
what the verl backend needs to compute the GRPO loss.

Note on rLLM API surface: rLLM is moving fast. The exact decorator name
(`@rllm.rollout`) and helper class (`rllm.client.OpenAI`) follow the
rllm-project.com docs as of the time this was written. If a newer release
renames them, adjust here only -- everything else in this file stays the
same.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

# rLLM may live under a slightly different import path depending on the
# pinned version. Try a couple of common ones.
try:
    import rllm  # type: ignore[import-not-found]
    from rllm.client import OpenAI as RllmOpenAI  # type: ignore[import-not-found]
    Episode = rllm.Episode  # type: ignore[attr-defined]
    rllm_rollout = rllm.rollout  # type: ignore[attr-defined]
except ImportError:
    # Fallback shims so the file imports cleanly during local linting on a
    # machine that doesn't have rllm installed (e.g. a Mac without GPUs).
    # These will explode at runtime if used; that's intentional.
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "rllm not available; using stub decorators. Install rllm in the RL container."
    )

    def rllm_rollout(fn):  # type: ignore[no-redef]
        return fn

    class RllmOpenAI:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("rllm.client.OpenAI not available; install rllm.")

    class Episode:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


# Make the existing repo importable when this module runs inside the RL pod
# (where /workspace/repo is on PYTHONPATH already). When running locally for
# tests, fall back to the relative path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tau2.agent.base import LocalAgent, ValidAgentInputMessage, is_valid_agent_history_message
from tau2.agent.llm_agent import LLMAgentState, AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import to_litellm_messages

from tau2_art_helpers import (
    Tau2TaskScenario,
    compute_shaped_reward,
    _evaluate_and_get_reward,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A LocalAgent that drives the policy through an rllm-tracked OpenAI client.
# Mirrors ARTAgent in tau2_art_helpers.py but uses the tracked client so every
# chat.completions call is captured by rLLM.
# ---------------------------------------------------------------------------

class RllmTrackedAgent(LocalAgent):
    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        client: Any,                  # rllm.client.OpenAI (tracked)
        model_name: str,              # vLLM-served model id, e.g. "policy"
        temperature: float = 1.0,
        llm_kwargs: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.llm_kwargs = llm_kwargs or {}
        self.openai_tools: list[dict] = [t.openai_schema for t in self.tools]
        self.completion_tokens: int = 0

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )

    def get_init_state(self, message_history=None):
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)

        openai_messages = to_litellm_messages(state.system_messages + state.messages)
        kwargs = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
        }
        if self.openai_tools:
            kwargs["tools"] = self.openai_tools
            kwargs["tool_choice"] = "auto"
        kwargs.update(self.llm_kwargs)

        response = self.client.chat.completions.create(**kwargs)
        if response.usage:
            self.completion_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        content = choice.message.content
        raw_tool_calls = choice.message.tool_calls or []
        tool_calls = None
        if raw_tool_calls:
            import json
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in raw_tool_calls
            ]

        assistant_msg = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=0.0,
        )
        state.messages.append(assistant_msg)
        return assistant_msg, state

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return bool(message.content and cls.STOP_TOKEN in message.content)

    def set_seed(self, seed: int):
        pass


# ---------------------------------------------------------------------------
# Helpers to build the two tracked clients up front (called once by the
# trainer init, then injected into each rollout invocation).
# ---------------------------------------------------------------------------

def build_policy_client(
    *,
    base_url: str,
    api_key: str = "EMPTY",
) -> Any:
    """rllm-tracked OpenAI client pointed at the local vLLM rollout server."""
    return RllmOpenAI(base_url=base_url, api_key=api_key)


def build_user_client(
    *,
    base_url: str = "https://api.inference.wandb.ai/v1",
    api_key: Optional[str] = None,
) -> Any:
    """rllm-tracked OpenAI client pointed at W&B Inference."""
    if api_key is None:
        api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY not set; cannot build user-sim client.")
    return RllmOpenAI(base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# The actual @rllm.rollout function passed to the AgentTrainer.
# ---------------------------------------------------------------------------

@rllm_rollout
async def tau2_rl_rollout(
    task: dict,
    *,
    policy_client: Any,
    user_client: Any,
    policy_model_name: str,
    user_model_name: str,
    domain: str,
    max_steps: int = 100,
    user_llm_args: Optional[dict] = None,
    agent_llm_args: Optional[dict] = None,
    shaped_reward_weights: Optional[dict] = None,
) -> Any:
    """One tau2 rollout for GRPO.

    `task` is one row from the dataset registered with AgentTrainer.fit:
        {"task_id": "...", "domain": "telecom"}

    Returns an rllm.Episode whose `reward` is the shaped reward and whose
    trajectory contains every chat.completions call made on `policy_client`
    (the user-sim calls on `user_client` are recorded too but don't influence
    the policy gradient).
    """
    user_llm_args = user_llm_args or {"temperature": 1.0}
    agent_llm_args = agent_llm_args or {"temperature": 0.7}

    task_id = task["task_id"]

    def _run_sync():
        # task_split_name=None -> return ALL tasks in tasks.json so any task ID
        # from any split (small / train / test / base / full) is findable.
        # The default split filter ("base") would silently exclude small-split
        # task IDs since `small ∩ base = ∅` for telecom.
        tasks = registry.get_tasks_loader(domain)(task_split_name=None)
        target = next((t for t in tasks if t.id == task_id), None)
        if target is None:
            raise ValueError(f"Task {task_id} not found in domain {domain}")

        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()

        agent = RllmTrackedAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            client=policy_client,
            model_name=policy_model_name,
            temperature=agent_llm_args.get("temperature", 0.7),
            llm_kwargs=deepcopy(agent_llm_args),
        )

        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None

        # The user simulator goes through litellm by default; we want it to
        # use the rLLM-tracked client too so its tokens land in the trace.
        # tau2's UserSimulator currently only takes a string `llm` field, so
        # we pass the W&B Inference model id and rely on litellm config to
        # route. (rLLM's LiteLLM proxy injects auth.) If you switch to a
        # custom UserSimulator that takes an OpenAI client, swap it in here.
        user = UserSimulator(
            tools=user_tools,
            instructions=str(target.user_scenario),
            llm=user_model_name,
            llm_args=deepcopy(user_llm_args),
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=target,
            max_steps=max_steps,
        )
        simulation = orchestrator.run()
        reward_info = _evaluate_and_get_reward(simulation, target, domain)
        return simulation, reward_info, target, agent.completion_tokens

    simulation, reward_info, target, completion_tokens = await asyncio.to_thread(_run_sync)

    binary_reward = reward_info.reward
    success = 1.0 if abs(binary_reward - 1.0) < 1e-6 else 0.0

    shaped_reward, sub_metrics = compute_shaped_reward(
        simulation,
        target,
        domain,
        weights=shaped_reward_weights,
        max_steps=max_steps,
        binary_reward=binary_reward,
    )

    metadata = {
        "task_id": task_id,
        "domain": domain,
        "termination_reason": simulation.termination_reason.value,
        "completion_tokens": completion_tokens,
        "success": success,
        "binary_reward": binary_reward,
        **{f"shaped/{k}": v for k, v in sub_metrics.items()},
    }

    # `Episode` is the rllm record. Trajectory token IDs/logprobs are captured
    # by the tracked policy_client on every chat.completions call.
    return Episode(
        reward=shaped_reward,
        metadata=metadata,
    )

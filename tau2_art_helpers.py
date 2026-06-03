"""
Helpers for training tau2-bench agents with ART (Absolute Reasoning Training).

Provides:
- ARTAgent: a tau2 LocalAgent that uses ART's model inference endpoint
- tau2_rollout: async function that runs a tau2 simulation and returns an ART Trajectory
- Tau2BaseModelWrapper: weave.Model wrapper for Weave Evaluation framework
- PassAtKScorer: Weave Scorer that computes pass^k metrics across trials
- score_task_reward / score_success: Weave scorers for tau2 evaluation
"""
import asyncio
import json
import logging
import os
import time
from copy import deepcopy
from typing import Any, List, Optional

import weave
from httpx import Timeout
from openai import OpenAI
from pydantic import BaseModel

import art
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
from tau2.data_model.simulation import TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import to_litellm_messages

logger = logging.getLogger(__name__)


class ARTAgent(LocalAgent):
    """tau2 agent that uses an ART model's inference endpoint.

    Records raw OpenAI Choice objects alongside regular messages so we can
    build an ART Trajectory after the orchestrator finishes.
    """

    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        inference_base_url: str,
        inference_api_key: str,
        model_name: str,
        temperature: float = 1.0,
        max_retries: int = 30,
        llm_kwargs: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.client = OpenAI(
            base_url=inference_base_url,
            api_key=inference_api_key,
            timeout=Timeout(300.0, connect=60.0),
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.llm_kwargs = llm_kwargs or {}

        self.art_messages_and_choices: list = []
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

        self.art_messages_and_choices.append(
            {"role": "system", "content": self.system_prompt}
        )
        for msg in message_history:
            if isinstance(msg, AssistantMessage):
                d = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in msg.tool_calls
                    ]
                self.art_messages_and_choices.append(d)
            elif isinstance(msg, UserMessage):
                self.art_messages_and_choices.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                self.art_messages_and_choices.append(
                    {"role": "tool", "tool_call_id": msg.id, "content": msg.content or ""}
                )

        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
        )

    @weave.op()
    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                self.art_messages_and_choices.append(
                    {"role": "tool", "tool_call_id": tm.id, "content": tm.content or ""}
                )
            state.messages.extend(message.tool_messages)
        elif message is not None:
            if isinstance(message, UserMessage):
                self.art_messages_and_choices.append(
                    {"role": "user", "content": message.content}
                )
            elif isinstance(message, ToolMessage):
                self.art_messages_and_choices.append(
                    {"role": "tool", "tool_call_id": message.id, "content": message.content or ""}
                )
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

        response = None
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                last_error = e
                # Fail-fast on deterministic context-length-overflow 400s:
                # retrying the same prompt produces the same error and wastes
                # up to ~30 min/rollout.  But "Already borrowed" and similar
                # transient 400s from vllm LoRA concurrency DO recover on
                # retry, so don't short-circuit those.
                msg = str(e)
                if (
                    ("Error code: 400" in msg or "BadRequestError" in msg)
                    and (
                        "context length" in msg
                        or "input_tokens" in msg
                        or "model ID is invalid" in msg
                    )
                ):
                    logger.warning(
                        "Inference API 400 (no retry, fail-fast): %s", e
                    )
                    raise
                if attempt < self.max_retries - 1:
                    delay = min(60, 2 ** attempt)
                    logger.warning(
                        "Inference API request failed (attempt %s/%s): %s; retrying in %ss.",
                        attempt + 1, self.max_retries, e, delay,
                    )
                    time.sleep(delay)
                    continue
                raise

            api_error = getattr(response, "error", None)
            if api_error is not None:
                err_msg = (
                    api_error.get("message", str(api_error))
                    if isinstance(api_error, dict)
                    else getattr(api_error, "message", str(api_error))
                )
                if attempt < self.max_retries - 1:
                    delay = min(60, 2 ** attempt)
                    logger.warning(
                        "Inference API error (attempt %s/%s): %s; retrying in %ss.",
                        attempt + 1, self.max_retries, err_msg, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Inference API error: {err_msg}") from None
            if not response.choices:
                if attempt < self.max_retries - 1:
                    delay = min(60, 2 ** attempt)
                    logger.warning(
                        "Inference API returned no choices (attempt %s/%s); retrying in %ss.",
                        attempt + 1, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    "Inference API returned no choices (response.choices is None or empty). "
                    "Check API rate limits or try again."
                ) from None
            break

        if response.usage:
            self.completion_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        self.art_messages_and_choices.append(choice)

        content = choice.message.content
        raw_tool_calls = choice.message.tool_calls or []
        tool_calls = None
        if raw_tool_calls:
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
        if message.content and cls.STOP_TOKEN in message.content:
            return True
        return False

    def set_seed(self, seed: int):
        pass


@weave.op()
def _evaluate_and_get_reward(simulation, task, domain):
    """Traced wrapper around tau2's evaluate_simulation."""
    return evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )


DEFAULT_SHAPED_WEIGHTS = {
    "action": 0.55,
    "termination": 0.20,
    "tool_accuracy": 0.20,
    "tool_arg_accuracy": 0.05,
    "step_penalty": -0.10,
    "max_step_penalty": -0.20,
    "repeat_message_penalty": -0.05,
}


@weave.op()
def compute_shaped_reward(
    simulation,
    task,
    domain: str,
    weights: Optional[dict] = None,
    max_steps: int = 30,
    binary_reward: Optional[float] = None,
) -> tuple[float, dict]:
    """Compute a continuous shaped reward from tau2 sub-evaluators.

    Unlike evaluate_simulation (which returns binary 0/1), this runs the
    individual evaluators regardless of termination reason and computes
    fractional scores so GRPO gets gradient signal from partial progress.

    Components with no evaluator data (None) are excluded and their weights
    redistributed proportionally so a perfect rollout always scores 1.0.

    `binary_reward`, when supplied, is the unshaped 0/1 task reward from
    `evaluate_simulation`. It is used to gate the termination bonus on
    actual task success (preventing the "give up immediately and collect
    the bonus" reward-hacking failure mode).

    Returns (shaped_reward, sub_metrics) where sub_metrics has per-component
    scores for observability.
    """
    w = {**DEFAULT_SHAPED_WEIGHTS, **(weights or {})}

    action_info = ActionEvaluator.calculate_reward(
        task=task, full_trajectory=simulation.messages,
    )

    action_checks = action_info.action_checks or []
    action_fraction = (
        sum(1 for c in action_checks if c.action_match) / len(action_checks)
        if action_checks else None
    )

    proper_termination = simulation.termination_reason in {
        TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP,
    }
    task_solved = binary_reward is not None and abs(binary_reward - 1.0) < 1e-6
    termination_bonus = 1.0 if (proper_termination and task_solved) else 0.0

    total_tool_calls = 0
    tool_not_found = 0
    tool_other_errors = 0
    for msg in simulation.messages:
        if isinstance(msg, ToolMessage):
            total_tool_calls += 1
            if msg.error:
                if msg.content and "not found" in msg.content.lower():
                    tool_not_found += 1
                else:
                    tool_other_errors += 1
    tool_accuracy = (1.0 - tool_not_found / total_tool_calls) if total_tool_calls > 0 else None
    tool_arg_accuracy = (1.0 - tool_other_errors / total_tool_calls) if total_tool_calls > 0 else None

    num_agent_steps = sum(1 for msg in simulation.messages if isinstance(msg, AssistantMessage))
    step_fraction = num_agent_steps / max_steps if max_steps > 0 else 0.0
    step_penalty = max(0.0, step_fraction - 0.5)

    # max_step_penalty: 1.0 if the episode hit the orchestrator step limit (unresolved loop),
    # else 0.0. Applied as a fixed penalty (weight -0.20).
    max_step_penalty = 1.0 if simulation.termination_reason == TerminationReason.MAX_STEPS else 0.0

    # repeat_message_penalty: fraction of consecutive identical assistant messages
    # (capped at 5 repeats → 1.0). Targets "I'll send the payment request..." loops.
    assistant_contents = [
        msg.content or ""
        for msg in simulation.messages
        if isinstance(msg, AssistantMessage)
    ]
    max_consecutive_repeats = 0
    current_repeats = 0
    prev_content = None
    for content in assistant_contents:
        if content == prev_content:
            current_repeats += 1
            max_consecutive_repeats = max(max_consecutive_repeats, current_repeats)
        else:
            current_repeats = 0
        prev_content = content
    repeat_message_penalty = min(max_consecutive_repeats, 5) / 5.0

    # Components: None means "not applicable for this task".
    # These are excluded from the weighted sum and their weight is redistributed.
    components = {
        "action": action_fraction,
        "termination": termination_bonus,
        "tool_accuracy": tool_accuracy,
        "tool_arg_accuracy": tool_arg_accuracy,
        "step_penalty": step_penalty,
        "max_step_penalty": max_step_penalty,
        "repeat_message_penalty": repeat_message_penalty,
    }

    # Separate positive-weight (reward) and negative-weight (penalty) components.
    # Redistribute positive weights among active (non-None) positive components
    # so a perfect rollout always scores 1.0 regardless of which evaluators apply.
    pos_active = {k: v for k, v in components.items() if v is not None and w.get(k, 0) >= 0}
    neg_active = {k: v for k, v in components.items() if v is not None and w.get(k, 0) < 0}

    pos_weight_sum = sum(w[k] for k in pos_active)
    if pos_weight_sum > 0:
        pos_reward = sum((w[k] / pos_weight_sum) * v for k, v in pos_active.items())
    else:
        pos_reward = 0.0

    neg_reward = sum(w[k] * v for k, v in neg_active.items())

    shaped_reward = max(0.0, pos_reward + neg_reward)

    sub_metrics = {
        "action_fraction": action_fraction if action_fraction is not None else -1.0,
        "termination_bonus": termination_bonus,
        "tool_accuracy": tool_accuracy if tool_accuracy is not None else -1.0,
        "tool_arg_accuracy": tool_arg_accuracy if tool_arg_accuracy is not None else -1.0,
        "step_penalty": step_penalty,
        "max_step_penalty": max_step_penalty,
        "repeat_message_penalty": repeat_message_penalty,
        "shaped_reward": shaped_reward,
    }
    return shaped_reward, sub_metrics


class Tau2TaskScenario(BaseModel):
    """Wraps a tau2 Task for use with ART's iterate_dataset."""
    step: int = 0
    task_id: str
    domain: str

def _redact_model_key(inputs: dict[str, Any]) -> dict[str, Any]:
    model = inputs.get("model")
    if model is not None and hasattr(model, "inference_api_key") and model.inference_api_key:
        from copy import copy
        redacted = copy(model)
        redacted.inference_api_key = "REDACTED"
        return {**inputs, "model": redacted}
    return inputs

def _tau2_messages_to_openai_dicts(messages) -> list[dict]:
    """Convert tau2 simulation messages to strict OpenAI chat-format dicts.

    Unlike tau2.utils.llm_utils.to_litellm_messages, this emits tool_calls
    using only the OpenAI-spec fields ({id, type, function:{name, arguments}})
    so they can be used directly as SFT training data per
    https://art.openpipe.ai/fundamentals/sft-training.
    """
    out: list[dict] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            d: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            out.append(d)
        elif isinstance(msg, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.id,
                    "content": msg.content or "",
                }
            )
    return out


@weave.op()
async def tau2_teacher_rollout(
    task_scenario: Tau2TaskScenario,
    teacher_llm: str,
    teacher_llm_args: Optional[dict] = None,
    user_llm: str = "openai/gpt-4o-mini",
    user_llm_args: Optional[dict] = None,
    max_steps: int = 30,
) -> art.Trajectory:
    """Run a tau2 simulation using `teacher_llm` as the agent and return an
    SFT-ready ART Trajectory.

    The returned Trajectory's `messages_and_choices` is a list of plain dicts
    in OpenAI chat format (system + user/assistant/tool turns), and `tools`
    is the domain's OpenAI tool schema. metrics["task_reward"] / "success"
    are populated from tau2's evaluator so callers can filter for successful
    teacher trajectories before training.
    """
    if teacher_llm_args is None:
        teacher_llm_args = {"temperature": 0.7}
    if user_llm_args is None:
        user_llm_args = {"temperature": 1.0}

    domain = task_scenario.domain
    task_id = task_scenario.task_id

    def _run_sync():
        # task_split_name=None loads all tasks from tasks.json (not just base),
        # so val tasks drawn from full-base are resolvable during rollout.
        tasks = registry.get_tasks_loader(domain)(task_split_name=None)
        task = None
        for t in tasks:
            if t.id == task_id:
                task = t
                break
        if task is None:
            raise ValueError(f"Task {task_id} not found in domain {domain}")

        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()

        AgentConstructor = registry.get_agent_constructor("llm_agent")
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=teacher_llm,
            llm_args=deepcopy(teacher_llm_args),
        )
        openai_tools = [t.openai_schema for t in environment.get_tools()]
        domain_policy = environment.get_policy()

        try:
            user_tools = environment.get_user_tools()
        except (ValueError, Exception):
            user_tools = None

        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=deepcopy(user_llm_args),
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
        )
        simulation = orchestrator.run()

        reward_info = _evaluate_and_get_reward(simulation, task, domain)
        simulation.reward_info = reward_info

        return (
            simulation,
            reward_info,
            openai_tools,
            domain_policy,
            task,
        )

    (
        simulation,
        reward_info,
        openai_tools,
        domain_policy,
        task,
    ) = await asyncio.to_thread(_run_sync)

    binary_reward = reward_info.reward
    success = 1.0 if abs(binary_reward - 1.0) < 1e-6 else 0.0

    sys_msg = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(
            domain_policy=domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        ),
    }
    convo = _tau2_messages_to_openai_dicts(simulation.messages)
    # tau2 simulation messages may already include the system message; if so
    # drop ours to avoid duplication.
    if convo and convo[0].get("role") == "system":
        messages = convo
    else:
        messages = [sys_msg] + convo

    traj = art.Trajectory(
        reward=binary_reward,
        messages_and_choices=messages,
        metadata={
            "task_id": task.id,
            "step": task_scenario.step,
            "domain": task_scenario.domain,
            "termination_reason": simulation.termination_reason.value,
            "teacher_llm": teacher_llm,
        },
        metrics={
            "task_reward": binary_reward,
            "success": success,
        },
    )
    traj.tools = openai_tools
    return traj


@weave.op(postprocess_inputs=_redact_model_key)
async def tau2_rollout(
    model: Optional[Any],
    task_scenario: Tau2TaskScenario,
    user_llm: str = "openai/gpt-4o-mini",
    user_llm_args: Optional[dict] = None,
    max_steps: int = 30,
    agent_llm: Optional[str] = None,
    agent_llm_args: Optional[dict] = None,
    use_shaped_reward: bool = False,
    shaped_reward_weights: Optional[dict] = None,
    task_reward_blend: Optional[float] = None,
    pinned_step: Optional[int] = None,
    pinned_alias: Optional[str] = None,
    seed: Optional[int] = None,
) -> art.Trajectory:
    """Run a tau2-bench simulation and return an ART Trajectory.

    When agent_llm is set (e.g. "openai/gpt-5.2"), uses tau2's LLMAgent with that
    LLM (same pattern as run.run_task), so no ART backend is needed. Otherwise
    uses ARTAgent with the given model (inference_base_url, etc.).

    When use_shaped_reward is True, traj.reward is set to the continuous shaped
    reward (for GRPO training signal) while the original binary reward is kept
    in traj.metrics["task_reward"].

    task_reward_blend (0-1): when set alongside use_shaped_reward, blends binary
    and shaped rewards: final = blend * binary + (1-blend) * shaped.
    This makes task completion dominate while still propagating a gradient signal
    through partial progress. Ignored when use_shaped_reward is False.
    """
    if user_llm_args is None:
        user_llm_args = {"temperature": 1.0}
    if agent_llm_args is None:
        agent_llm_args = {}

    domain = task_scenario.domain
    task_id = task_scenario.task_id

    def _run_sync():
        # task_split_name=None loads all tasks from tasks.json (not just base),
        # so val tasks drawn from full-base are resolvable during rollout.
        tasks = registry.get_tasks_loader(domain)(task_split_name=None)
        task = None
        for t in tasks:
            if t.id == task_id:
                task = t
                break
        if task is None:
            raise ValueError(f"Task {task_id} not found in domain {domain}")

        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()
        
        if agent_llm is not None:
            AgentConstructor = registry.get_agent_constructor("llm_agent")
            agent = AgentConstructor(
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
                llm=agent_llm,
                llm_args=deepcopy(agent_llm_args),
            )
            openai_tools = [t.openai_schema for t in environment.get_tools()]
            completion_tokens = 0
            messages_and_choices = []
        else:
            # Prefer the model's inference_api_key (set by the backend during
            # prepare_backend_for_training).  For LocalBackend this is the
            # local vLLM server's key ("default"); for ServerlessBackend it's
            # the W&B Inference key.  Fall back to env for eval-time callers
            # who pass an unregistered model.
            inference_api_key = model.inference_api_key or os.getenv("WANDB_API_KEY")
            if pinned_alias is not None:
                # Pin directly to any W&B artifact alias on this model's collection
                # (e.g. "v1", "latest", "best"). Useful when the desired checkpoint
                # exists in W&B but doesn't carry a `step{N}` alias (for example,
                # an artifact orphaned by a buggy fork).
                inference_name = (
                    f"wandb-artifact:///{model.entity}/{model.project}/"
                    f"{model.name}:{pinned_alias}"
                )
            elif pinned_step is not None:
                inference_name = model.get_inference_name(step=pinned_step)
            else:
                inference_name = model.get_inference_name()
            agent = ARTAgent(
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
                inference_base_url=model.inference_base_url,
                inference_api_key=inference_api_key,
                model_name=inference_name,
                temperature=agent_llm_args.get("temperature", 1.0),
                llm_kwargs=deepcopy(agent_llm_args),
            )
            openai_tools = agent.openai_tools
            completion_tokens = 0
            messages_and_choices = []

        try:
            user_tools = environment.get_user_tools()
        except (ValueError, Exception):
            user_tools = None

        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=deepcopy(user_llm_args),
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            seed=seed,
        )
        simulation = orchestrator.run()

        reward_info = _evaluate_and_get_reward(simulation, task, domain)
        simulation.reward_info = reward_info

        if agent_llm is None:
            completion_tokens = agent.completion_tokens
            messages_and_choices = agent.art_messages_and_choices

        return (
            simulation,
            reward_info,
            messages_and_choices,
            completion_tokens,
            task,
            openai_tools,
        )

    (
        simulation,
        reward_info,
        messages_and_choices,
        completion_tokens,
        task,
        openai_tools,
    ) = await asyncio.to_thread(_run_sync)

    binary_reward = reward_info.reward
    success = 1.0 if abs(binary_reward - 1.0) < 1e-6 else 0.0

    metrics = {
        "task_reward": binary_reward,
        "success": success,
    }

    if use_shaped_reward:
        shaped_reward, sub_metrics = compute_shaped_reward(
            simulation, task, task_scenario.domain,
            weights=shaped_reward_weights,
            max_steps=max_steps,
            binary_reward=binary_reward,
        )
        if task_reward_blend is not None:
            blend = float(task_reward_blend)
            reward = blend * binary_reward + (1.0 - blend) * shaped_reward
        else:
            reward = shaped_reward
        metrics.update(sub_metrics)
        metrics["shaped_reward"] = shaped_reward
    else:
        reward = binary_reward

    traj = art.Trajectory(
        reward=reward,
        messages_and_choices=messages_and_choices,
        metadata={
            "task_id": task.id,
            "step": task_scenario.step,
            "domain": task_scenario.domain,
            "termination_reason": simulation.termination_reason.value,
            "completion_tokens": completion_tokens,
        },
        metrics=metrics,
    )
    traj.tools = openai_tools
    return traj


# ─────────────────────────────────────────────────────────────────────
# Weave Evaluation wrappers and scorers
# ─────────────────────────────────────────────────────────────────────

class Tau2BaseModelWrapper(weave.Model):
    """Weave Model wrapper for tau2-bench evaluation (base and trained).

    Used with weave.Evaluation so per-task results and aggregate metrics
    appear in the Weave console.  When agent_llm is set (e.g. "openai/gpt-5.2"),
    uses tau2's LLMAgent with that LLM (no ART backend needed). Otherwise
    uses the given ART model (e.g. a GRPO-trained TrainableModel).
    """

    model: Optional[Any] = None
    model_name: str
    domain: str
    user_llm: str
    user_llm_args: dict
    agent_llm_args: dict = {}
    max_steps: int = 30
    agent_llm: Optional[str] = None
    use_shaped_reward: bool = False
    shaped_reward_weights: dict = {}
    # When set, evaluate a specific LoRA checkpoint of `self.model` instead
    # of the collection's `:latest`. Resolves to the W&B artifact alias
    # `step{pinned_step}` (e.g. step0 = freshly forked SFT, pre-RL).
    pinned_step: Optional[int] = None
    # When set, pin directly to an arbitrary W&B artifact alias on this
    # collection (e.g. "v1", "latest"). Takes precedence over `pinned_step`.
    # Useful for evaluating artifacts that were uploaded but never received
    # a `step{N}` alias (e.g. orphaned fork checkpoints).
    pinned_alias: Optional[str] = None
    # CRN: when set, passes a fixed seed to the Orchestrator so the user
    # simulator produces deterministic behaviour.  Both SFT and RL wrappers
    # must receive the same seed value for the pairing to be valid.
    seed: Optional[int] = None
    # Concurrency cap: limits how many concurrent tau2_rollout calls this
    # wrapper issues.  Uses a per-instance asyncio.Semaphore lazily created
    # on first predict() call.  None means unlimited (Weave default).
    max_concurrency: Optional[int] = None

    # asyncio.Semaphore and result lists are not serialisable by Weave/pydantic;
    # keep them outside the model fields using private instance variables.
    _semaphore: Optional[Any] = None
    # Accumulated per-trial results from predict() calls during evaluate().
    # Each entry: {task_id, reward, success, dropped, trial_idx}.
    # Read after evaluate() returns to get per-task data for paired stats and
    # saving to data/simulations/.
    _task_results: Optional[Any] = None

    def _get_semaphore(self):
        if self.max_concurrency is None:
            return None
        if self._semaphore is None:
            object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.max_concurrency))
        return self._semaphore

    def _record_result(self, result: dict) -> None:
        """Append a predict() result to the instance-level result list."""
        if self._task_results is None:
            object.__setattr__(self, "_task_results", [])
        self._task_results.append(result)

    @weave.op()
    async def predict(self, task_id: str, domain: str) -> dict:
        sem = self._get_semaphore()
        scenario = Tau2TaskScenario(step=0, task_id=task_id, domain=domain)
        try:
            if sem is not None:
                async with sem:
                    traj = await tau2_rollout(
                        self.model,
                        scenario,
                        user_llm=self.user_llm,
                        user_llm_args=self.user_llm_args,
                        max_steps=self.max_steps,
                        agent_llm=self.agent_llm,
                        agent_llm_args=self.agent_llm_args,
                        use_shaped_reward=self.use_shaped_reward,
                        shaped_reward_weights=self.shaped_reward_weights,
                        pinned_step=self.pinned_step,
                        pinned_alias=self.pinned_alias,
                        seed=self.seed,
                    )
            else:
                traj = await tau2_rollout(
                    self.model,
                    scenario,
                    user_llm=self.user_llm,
                    user_llm_args=self.user_llm_args,
                    max_steps=self.max_steps,
                    agent_llm=self.agent_llm,
                    agent_llm_args=self.agent_llm_args,
                    use_shaped_reward=self.use_shaped_reward,
                    shaped_reward_weights=self.shaped_reward_weights,
                    pinned_step=self.pinned_step,
                    pinned_alias=self.pinned_alias,
                    seed=self.seed,
                )
        except Exception as e:
            logger.warning("Leaderboard eval failed for task_id=%s: %s", task_id, e)
            result = {
                "task_id": task_id,
                "reward": 0.0,
                "success": 0.0,
                "termination_reason": "error",
                "completion_tokens": 0,
                "dropped": True,
                "error_reason": str(e)[:200],
            }
            self._record_result(result)
            return result
        result = {
            "task_id": task_id,
            "reward": traj.reward,
            "success": traj.metrics.get("success", 0.0),
            "termination_reason": traj.metadata.get("termination_reason", "unknown"),
            "completion_tokens": traj.metadata.get("completion_tokens", 0),
            "dropped": False,
            "error_reason": "",
        }
        self._record_result(result)
        return result


class PassAtKScorer(weave.Scorer):
    """Weave scorer that computes pass^k metrics across multiple trials per task.

    Uses the same is_successful() and pass_hat_k() functions as the CLI
    (tau2.metrics.agent_metrics) to guarantee identical results.
    """
    num_trials: int = 3

    @weave.op()
    def score(self, *, output: dict) -> dict:
        from tau2.metrics.agent_metrics import is_successful
        # Use the precomputed BINARY success field. output["reward"] may be the
        # continuous SHAPED reward (when shaped_reward is enabled); calling
        # is_successful() on a shaped score like 0.99 wrongly counts it as a
        # failure and collapses pass^k toward 0. The binary `success` field is
        # itself is_successful(binary_reward) computed inside tau2_rollout, so
        # results remain identical to the CLI for binary rewards. Fall back to
        # is_successful(reward) only for older result dicts that lack `success`.
        if "success" in output:
            success = bool(output["success"])
        else:
            success = is_successful(output.get("reward", 0.0))
        return {
            "task_id": output["task_id"],
            "success": success,
        }

    @weave.op()
    def summarize(self, score_rows: list) -> dict:
        from collections import defaultdict
        from tau2.metrics.agent_metrics import pass_hat_k

        task_results: dict[str, list[bool]] = defaultdict(list)
        skipped = 0
        for row in score_rows:
            if not isinstance(row, dict) or "task_id" not in row or "success" not in row:
                skipped += 1
                continue
            task_results[row["task_id"]].append(row["success"])
        if skipped:
            logger.warning(
                "PassAtKScorer: skipped %s score row(s) missing task_id/success (e.g. API failures after retries). "
                "Included %s tasks with at least one valid trial.",
                skipped, len(task_results),
            )

        pass_hat_ks = {}
        num_tasks = len(task_results)
        for k in range(1, self.num_trials + 1):
            per_task = []
            for task_id, successes in task_results.items():
                n = len(successes)
                s = sum(successes)
                if n < k:
                    continue
                per_task.append(pass_hat_k(n, s, k))
            pass_hat_ks[f"pass^{k}"] = {
                "mean": sum(per_task) / num_tasks if num_tasks else 0.0
            }
        return pass_hat_ks


@weave.op()
def score_task_reward(model_output: dict) -> dict:
    """Weave scorer: extracts the binary task reward (0/1)."""
    return {"task_reward": model_output.get("reward", 0.0)}


@weave.op()
def score_success(model_output: dict) -> dict:
    """Weave scorer: extracts the binary success flag (0/1).

    Returns None for dropped trials so infra errors are excluded from the
    mean rather than counted as 0.
    """
    if model_output.get("dropped"):
        return {"success": None}
    return {"success": model_output.get("success", 0.0)}


@weave.op()
def score_error(model_output: dict) -> dict:
    """Weave scorer: flags task-trials dropped due to API/infra errors.

    dropped.mean on the leaderboard reads as the fraction of task-trials
    that errored out. 0.00 is clean; anything above 0.02 deserves a footnote.
    """
    return {"dropped": int(model_output.get("dropped", False))}


# NOTE: A `fork_model_weights` helper used to live here, wrapping ART's
# `_experimental_fork_checkpoint`. It was removed because forked artifacts
# upload bytes to W&B Artifacts but are NOT registered with W&B Inference,
# making them unservable. To start RL from an existing SFT, set
# `continue_from_model: <sft-collection-name>` in train_config.yaml so GRPO
# continues that collection's checkpoint history (step 7, 8, …) through the
# proper server-side checkpoint-registration path.

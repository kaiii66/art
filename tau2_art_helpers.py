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
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
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
    "action": 0.35,
    "communicate": 0.10,
    "nl_assertions": 0.30,
    "termination": 0.15,
    "tool_accuracy": 0.15,
    "tool_arg_accuracy": 0.05,
    "step_penalty": -0.10,
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
    communicate_info = CommunicateEvaluator.calculate_reward(
        task=task, full_trajectory=simulation.messages,
    )
    nl_info = NLAssertionsEvaluator.calculate_reward(
        task=task, full_trajectory=simulation.messages,
    )

    action_checks = action_info.action_checks or []
    action_fraction = (
        sum(1 for c in action_checks if c.action_match) / len(action_checks)
        if action_checks else None
    )

    comm_checks = communicate_info.communicate_checks or []
    communicate_fraction = (
        sum(1 for c in comm_checks if c.met) / len(comm_checks)
        if comm_checks else None
    )

    nl_checks = nl_info.nl_assertions or []
    nl_fraction = (
        sum(1 for c in nl_checks if c.met) / len(nl_checks)
        if nl_checks else None
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

    # Components: None means "not applicable for this task" (e.g. nl_assertions in telecom).
    # These are excluded from the weighted sum and their weight is redistributed.
    components = {
        "action": action_fraction,
        "communicate": communicate_fraction,
        "nl_assertions": nl_fraction,
        "termination": termination_bonus,
        "tool_accuracy": tool_accuracy,
        "tool_arg_accuracy": tool_arg_accuracy,
        "step_penalty": step_penalty,
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
        "communicate_fraction": communicate_fraction if communicate_fraction is not None else -1.0,
        "nl_fraction": nl_fraction if nl_fraction is not None else -1.0,
        "termination_bonus": termination_bonus,
        "tool_accuracy": tool_accuracy if tool_accuracy is not None else -1.0,
        "tool_arg_accuracy": tool_arg_accuracy if tool_arg_accuracy is not None else -1.0,
        "step_penalty": step_penalty,
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
        # task_split_name=None -> return ALL tasks in tasks.json so any task ID
        # from any split (small / train / test / base / full) is findable.
        # The default split filter ("base") would silently exclude small-split
        # task IDs since `small ∩ base = ∅` for telecom.
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
    pinned_step: Optional[int] = None,
    pinned_alias: Optional[str] = None,
) -> art.Trajectory:
    """Run a tau2-bench simulation and return an ART Trajectory.

    When agent_llm is set (e.g. "openai/gpt-5.2"), uses tau2's LLMAgent with that
    LLM (same pattern as run.run_task), so no ART backend is needed. Otherwise
    uses ARTAgent with the given model (inference_base_url, etc.).

    When use_shaped_reward is True, traj.reward is set to the continuous shaped
    reward (for GRPO training signal) while the original binary reward is kept
    in traj.metrics["task_reward"].
    """
    if user_llm_args is None:
        user_llm_args = {"temperature": 1.0}
    if agent_llm_args is None:
        agent_llm_args = {}

    domain = task_scenario.domain
    task_id = task_scenario.task_id

    def _run_sync():
        # task_split_name=None -> return ALL tasks in tasks.json so any task ID
        # from any split (small / train / test / base / full) is findable.
        # The default split filter ("base") would silently exclude small-split
        # task IDs since `small ∩ base = ∅` for telecom.
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
            inference_api_key = os.getenv("WANDB_API_KEY")
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
        reward = shaped_reward
        metrics.update(sub_metrics)
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

    @weave.op()
    async def predict(self, task_id: str, domain: str) -> dict:
        scenario = Tau2TaskScenario(step=0, task_id=task_id, domain=domain)
        try:
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
            )
        except Exception as e:
            logger.warning("Leaderboard eval failed for task_id=%s (will be excluded from pass^k): %s", task_id, e)
            raise
        return {
            "task_id": task_id,
            "reward": traj.reward,
            "success": traj.metrics.get("success", 0.0),
            "termination_reason": traj.metadata.get("termination_reason", "unknown"),
            "completion_tokens": traj.metadata.get("completion_tokens", 0),
        }


class PassAtKScorer(weave.Scorer):
    """Weave scorer that computes pass^k metrics across multiple trials per task.

    Uses the same is_successful() and pass_hat_k() functions as the CLI
    (tau2.metrics.agent_metrics) to guarantee identical results.
    """
    num_trials: int = 3

    @weave.op()
    def score(self, *, output: dict) -> dict:
        from tau2.metrics.agent_metrics import is_successful
        return {
            "task_id": output["task_id"],
            "success": is_successful(output.get("reward", 0.0)),
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
    """Weave scorer: extracts the binary success flag (0/1)."""
    return {"success": model_output.get("success", 0.0)}


# NOTE: A `fork_model_weights` helper used to live here, wrapping ART's
# `_experimental_fork_checkpoint`. It was removed because forked artifacts
# upload bytes to W&B Artifacts but are NOT registered with W&B Inference,
# making them unservable. To start RL from an existing SFT, set
# `continue_from_model: <sft-collection-name>` in train_config.yaml so GRPO
# continues that collection's checkpoint history (step 7, 8, …) through the
# proper server-side checkpoint-registration path.

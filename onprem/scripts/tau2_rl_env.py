"""rLLM `BaseEnv` adapter for tau2-bench.

Owns the orchestration loop that tau2's `Orchestrator` would normally drive,
but factored as `(reset, step)` so rLLM's `MultiTurnWorkflow` can treat it as
a Gym-style environment. Routing rules mirror tau2's orchestrator exactly:

  * Agent text  -> `UserSimulator.generate_next_message` -> next user content
  * Agent tools -> `Environment.get_response` per `ToolCall` -> tool outputs
  * Empty / mixed agent message -> terminate as `AGENT_ERROR`
  * `STOP` / `TRANSFER` / `OUT_OF_SCOPE` in user text -> `USER_STOP`
  * `###STOP###` in agent text -> `AGENT_STOP`

On terminate we synthesise a tau2 `SimulationRun` and feed it to
`tau2_art_helpers.compute_shaped_reward` (the same scoring function the
ART/SFT rollout pipeline uses) so the RL gradient signal matches the SFT
benchmark.

The user simulator runs in-process via tau2's litellm wrapper and is pointed
at W&B Inference (`wandb/Qwen/...`) so it never competes with the trainable
policy for local GPUs.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

# Make the existing repo importable when running inside the RL pod
# (/workspace/repo is on PYTHONPATH at runtime; the explicit insert lets
# tests + smoke runs work from a checkout too).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rllm.agents.agent import Action  # noqa: E402
from rllm.environments.base.base_env import BaseEnv  # noqa: E402

from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason  # noqa: E402
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.user.base import OUT_OF_SCOPE, STOP, TRANSFER  # noqa: E402
from tau2.user.user_simulator import UserSimulator  # noqa: E402
from tau2.utils.utils import get_now  # noqa: E402

from tau2_art_helpers import _evaluate_and_get_reward, compute_shaped_reward  # noqa: E402

logger = logging.getLogger(__name__)

# Agent stop sentinel matches the one tau2's LLMAgent.is_stop checks for.
_AGENT_STOP_TOKEN = "###STOP###"

# Per-domain caches so we don't re-resolve the task list / env constructor on
# every rollout. The env constructor itself is called fresh per task to get
# clean DB state (mirrors `Orchestrator(...)` in tau2_rllm_rollout.py).
_TASKS_CACHE: dict[str, list[Any]] = {}


def _load_tasks(domain: str) -> list[Any]:
    if domain not in _TASKS_CACHE:
        # task_split_name=None -> all tasks in tasks.json so we can resolve
        # any task_id regardless of which split it came from. The default
        # "base" filter would silently drop small-split task IDs.
        _TASKS_CACHE[domain] = registry.get_tasks_loader(domain)(task_split_name=None)
    return _TASKS_CACHE[domain]


class Tau2Env(BaseEnv):
    """tau2-bench wrapped as an rLLM `BaseEnv`.

    The same instance is reused across rollouts inside one workflow pool slot
    (`MultiTurnWorkflow.__init__` builds it once via `env_cls(**env_args)` and
    then calls `env.reset(task)` for each new task), so domain-level state is
    cached and per-task state is rebuilt on every `reset()`.

    Action shapes accepted from the agent (post `update_from_model`):
      * `list[dict]` of OpenAI-style tool calls -> dispatch to environment
      * `str` -> assistant text reply, routed to the user simulator
      * `dict` with key `"_assistant_text"` -> explicit text reply
      * `Action(action=...)` wrapper -> unwrapped first
    """

    def __init__(
        self,
        domain: str,
        max_steps: int = 50,
        user_llm: str = "wandb/Qwen/Qwen3-30B-A3B-Instruct-2507",
        user_llm_args: Optional[dict] = None,
        shaped_reward_weights: Optional[dict] = None,
        max_user_errors: int = 10,
    ):
        self.domain = domain
        self.max_steps = max_steps
        self.user_llm = user_llm
        self.user_llm_args = user_llm_args or {"temperature": 1.0, "max_tokens": 16384}
        self.shaped_reward_weights = shaped_reward_weights
        self.max_user_errors = max_user_errors

        # Built lazily on the first reset() since constructing tau2's
        # environment for a domain pulls in domain-specific data files.
        self._tasks: Optional[list[Any]] = None

        # Per-rollout state (re-initialised in reset()).
        self.task_id: Optional[str] = None
        self.target: Optional[Any] = None
        self.environment = None
        self.user_simulator: Optional[UserSimulator] = None
        self.user_state = None
        self.tools_openai: list[dict] = []

        self.step_count: int = 0
        self.num_user_errors: int = 0
        self.done: bool = False
        self.termination_reason: Optional[TerminationReason] = None
        self._messages: list[Any] = []
        self._start_perf: float = 0.0
        self._start_time: str = ""

    # ------------------------------------------------------------------
    # rllm BaseEnv API
    # ------------------------------------------------------------------

    def reset(self, task: dict | None = None) -> tuple[dict, dict]:
        """Initialise a fresh rollout.

        `Workflow.reset` (auto-discovery) AND `MultiTurnWorkflow.reset` both
        call this with the same task; doing the work once on each call is
        cheap (a few hundred ms for the first user-sim turn) and keeps the
        contract stateless from the caller's perspective.
        """
        if task is None:
            # Auto-discovery first call without a task -- defer real work
            # until MultiTurnWorkflow.reset passes the task explicitly.
            return {}, {}

        task_id = task.get("task_id")
        if not task_id:
            raise ValueError(f"Tau2Env.reset: task missing 'task_id': {task!r}")
        domain = task.get("domain", self.domain)
        if domain != self.domain:
            # Crossing domains within a single workflow pool slot is unsupported
            # because tools/policy are baked into the agent at __init__ time.
            raise ValueError(
                f"Tau2Env.reset: task domain {domain!r} != env domain {self.domain!r}; "
                "use one workflow per domain."
            )

        self._tasks = _load_tasks(self.domain)
        target = next((t for t in self._tasks if t.id == task_id), None)
        if target is None:
            raise ValueError(f"Tau2Env.reset: task_id {task_id!r} not found in domain {self.domain!r}")

        # Fresh environment per rollout -> fresh DB / tool state.
        env_constructor = registry.get_env_constructor(self.domain)
        environment = env_constructor()

        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None

        user_simulator = UserSimulator(
            tools=user_tools,
            instructions=str(target.user_scenario),
            llm=self.user_llm,
            llm_args=deepcopy(self.user_llm_args),
        )
        user_state = user_simulator.get_init_state(message_history=[])

        # Per-rollout state.
        self.task_id = task_id
        self.target = target
        self.environment = environment
        self.user_simulator = user_simulator
        self.user_state = user_state
        self.tools_openai = [t.openai_schema for t in environment.get_tools()]
        self.step_count = 0
        self.num_user_errors = 0
        self.done = False
        self.termination_reason = None
        self._start_perf = time.perf_counter()
        self._start_time = get_now()

        # Mirror tau2 Orchestrator.initialize() in non-solo mode: the agent
        # opens with DEFAULT_FIRST_AGENT_MESSAGE ("Hi! How can I help you
        # today?"), the user simulator responds. Whatever the user says is
        # the first observation the trainable agent sees.
        first_agent_msg = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        first_agent_msg.timestamp = get_now()
        user_msg, self.user_state = user_simulator.generate_next_message(
            first_agent_msg, self.user_state
        )
        try:
            user_msg.validate()
        except Exception as exc:
            logger.warning("user-sim emitted invalid first message: %s", exc)
        self._messages = [first_agent_msg, user_msg]

        if self.user_simulator.is_stop(user_msg):
            self.done = True
            self.termination_reason = TerminationReason.USER_STOP
            obs = self._format_observation_user(user_msg)
            return obs, self._terminal_info()

        return self._format_observation_user(user_msg), {}

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Drive one tau2 turn.

        `action` comes straight from `Tau2AssistantAgent.update_from_model` and
        is one of:
          * `Action(action=list[dict])` -- tool calls (OpenAI-spec)
          * `Action(action=str)` -- plain assistant text
          * `Action(action={"_assistant_text": str})` -- explicit text reply
          * raw `list` / `str` / `dict` (tolerated for back-compat)
        """
        if self.done:
            raise RuntimeError("Tau2Env.step called after termination")

        action_payload = self._unwrap_action(action)
        self.step_count += 1

        tool_calls, assistant_text = self._classify_action(action_payload)

        # Communication-protocol violations (mirrors tau2 _check_communication_error):
        #   * empty (no text and no tool calls)
        #   * mixed (text AND tool calls in the same response)
        if not tool_calls and not assistant_text:
            return self._terminate(TerminationReason.AGENT_ERROR, info_extra={
                "violation": "empty_assistant_message",
            })
        if tool_calls and assistant_text:
            return self._terminate(TerminationReason.AGENT_ERROR, info_extra={
                "violation": "mixed_assistant_message",
            })

        if tool_calls:
            return self._step_tool_calls(tool_calls)

        # Plain assistant text: check for the agent stop sentinel first.
        assistant_msg = AssistantMessage(
            role="assistant", content=assistant_text, cost=0.0,
            timestamp=get_now(),
        )
        self._messages.append(assistant_msg)
        if _AGENT_STOP_TOKEN in (assistant_text or ""):
            return self._terminate(TerminationReason.AGENT_STOP)

        return self._step_user_turn(assistant_msg)

    @staticmethod
    def from_dict(env_args: dict) -> "Tau2Env":
        """rLLM training pipelines use this when they need to construct an
        env from a single dict (e.g. legacy verl agent path). The
        MultiTurnWorkflow path uses ``env_cls(**env_args)`` directly so we
        just forward the same kwargs."""
        env_args = dict(env_args)
        # Drop dataset-row keys if a task-row dict was passed in by mistake.
        env_args.pop("task_id", None)
        return Tau2Env(**env_args)

    @staticmethod
    def is_multithread_safe() -> bool:
        # tau2 environments are independent per instance, and we never share
        # one across rollouts. Each Tau2Env lives in its own MultiTurnWorkflow
        # pool slot.
        return True

    def close(self) -> None:
        self.environment = None
        self.user_simulator = None
        self.user_state = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_action(action: Any) -> Any:
        if isinstance(action, Action):
            return action.action
        return action

    @staticmethod
    def _classify_action(action: Any) -> tuple[list[dict], str]:
        """Return (tool_calls, assistant_text). Either may be empty."""
        if action is None:
            return [], ""
        if isinstance(action, str):
            return [], action
        if isinstance(action, dict):
            # Two shapes accepted:
            #   * {"_assistant_text": "..."} (explicit text reply)
            #   * a single OpenAI tool call (rare; rllm parsers emit lists)
            if "_assistant_text" in action:
                return [], action["_assistant_text"] or ""
            return [action], ""
        if isinstance(action, list):
            # Filter out the synthetic "finish" tool call ToolAgent emits
            # when no tools were parsed -- that's not a real tau2 tool call.
            real_calls = [
                c for c in action
                if isinstance(c, dict)
                and c.get("function", {}).get("name") not in (None, "finish")
            ]
            return real_calls, ""
        # Unknown payload -> treat as text repr so the rollout still terminates
        # cleanly (will trigger AGENT_ERROR via the empty/mixed guard upstream).
        return [], str(action)

    def _step_tool_calls(self, tool_calls: list[dict]) -> tuple[Any, float, bool, dict]:
        """Dispatch each tool call through tau2's environment + record a
        single MultiToolMessage-style assistant + tool sequence."""
        assert self.environment is not None

        # Build a tau2 AssistantMessage that carries the tool calls. We need
        # this in self._messages so the shaped-reward evaluator sees the
        # action history.
        tau2_calls: list[ToolCall] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                import json as _json
                try:
                    args = _json.loads(args)
                except _json.JSONDecodeError:
                    args = {"_raw": args}
            tau2_calls.append(
                ToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    name=name,
                    arguments=args or {},
                    requestor="assistant",
                )
            )
        assistant_msg = AssistantMessage(
            role="assistant", content=None, tool_calls=tau2_calls, cost=0.0,
            timestamp=get_now(),
        )
        self._messages.append(assistant_msg)

        tool_outputs: dict[str, str] = {}
        for tau2_call in tau2_calls:
            tool_msg: ToolMessage = self.environment.get_response(tau2_call)
            self._messages.append(tool_msg)
            tool_outputs[tau2_call.id] = tool_msg.content or ""

        # tau2 calls sync_tools after every step; the environment subclass
        # may use it to refresh derived tool state.
        try:
            self.environment.sync_tools()
        except Exception as exc:
            logger.debug("environment.sync_tools() failed: %s", exc)

        if self.step_count >= self.max_steps:
            return self._terminate(TerminationReason.MAX_STEPS)

        return {"tool_outputs": tool_outputs}, 0.0, False, {"step": self.step_count}

    def _step_user_turn(self, assistant_msg: AssistantMessage) -> tuple[Any, float, bool, dict]:
        assert self.user_simulator is not None
        try:
            user_msg, self.user_state = self.user_simulator.generate_next_message(
                assistant_msg, self.user_state
            )
            user_msg.validate()
        except Exception as exc:
            logger.warning("user-sim error in step %d: %s", self.step_count, exc)
            self.num_user_errors += 1
            if self.num_user_errors >= self.max_user_errors:
                return self._terminate(
                    TerminationReason.USER_ERROR,
                    info_extra={"user_sim_error": str(exc)},
                )
            # Inject a placeholder so the agent gets *some* feedback.
            placeholder = UserMessage(
                role="user",
                content="[user-sim error; continue helping]",
                timestamp=get_now(),
            )
            self._messages.append(placeholder)
            user_msg = placeholder
        else:
            self._messages.append(user_msg)

        if user_msg.is_tool_call():
            # User-tool calls (rare: only some domains expose user tools).
            # Dispatch via the same environment endpoint and feed the result
            # back to the agent on the next step.
            assert self.environment is not None
            tool_outputs: dict[str, str] = {}
            for tc in user_msg.tool_calls or []:
                tool_msg: ToolMessage = self.environment.get_response(tc)
                self._messages.append(tool_msg)
                tool_outputs[tc.id] = tool_msg.content or ""
            try:
                self.environment.sync_tools()
            except Exception:
                pass
            if self.step_count >= self.max_steps:
                return self._terminate(TerminationReason.MAX_STEPS)
            return {"tool_outputs": tool_outputs}, 0.0, False, {"step": self.step_count}

        if self.user_simulator.is_stop(user_msg):
            return self._terminate(TerminationReason.USER_STOP)

        if self.step_count >= self.max_steps:
            return self._terminate(TerminationReason.MAX_STEPS)

        return self._format_observation_user(user_msg), 0.0, False, {"step": self.step_count}

    def _format_observation_user(self, user_msg: UserMessage) -> dict:
        # Match ToolAgent's observation shape so a single update_from_env can
        # handle either path: "user_content" for text, "tool_outputs" for
        # tool-call dispatch results.
        return {"user_content": user_msg.content or ""}

    def _terminate(
        self,
        reason: TerminationReason,
        info_extra: Optional[dict] = None,
    ) -> tuple[Any, float, bool, dict]:
        self.done = True
        self.termination_reason = reason
        info = self._terminal_info(info_extra=info_extra)
        return {}, info["shaped_reward"], True, info

    def _terminal_info(self, info_extra: Optional[dict] = None) -> dict:
        """Compute the shaped reward and pack diagnostics into the info dict."""
        reason = self.termination_reason or TerminationReason.AGENT_ERROR
        duration = time.perf_counter() - self._start_perf
        end_time = get_now()

        simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task_id or "unknown",
            start_time=self._start_time or end_time,
            end_time=end_time,
            duration=duration,
            termination_reason=reason,
            messages=list(self._messages),
            seed=None,
        )

        try:
            reward_info = _evaluate_and_get_reward(simulation, self.target, self.domain)
            binary_reward = float(reward_info.reward)
        except Exception as exc:
            logger.warning("evaluate_simulation failed for task %s: %s", self.task_id, exc)
            binary_reward = 0.0

        try:
            shaped_reward, sub_metrics = compute_shaped_reward(
                simulation,
                self.target,
                self.domain,
                weights=self.shaped_reward_weights,
                max_steps=self.max_steps,
                binary_reward=binary_reward,
            )
        except Exception as exc:
            logger.warning("compute_shaped_reward failed for task %s: %s", self.task_id, exc)
            shaped_reward = 0.0
            sub_metrics = {"error": str(exc)}

        success = 1.0 if abs(binary_reward - 1.0) < 1e-6 else 0.0
        info = {
            "task_id": self.task_id,
            "domain": self.domain,
            "termination_reason": reason.value,
            "binary_reward": binary_reward,
            "shaped_reward": shaped_reward,
            "success": success,
            "step_count": self.step_count,
            **{f"shaped/{k}": v for k, v in sub_metrics.items()},
        }
        if info_extra:
            info.update(info_extra)
        return info


# Convenience alias used by tests/ + Hydra config target paths.
__all__ = ["Tau2Env"]

"""rLLM `BaseAgent` adapter for tau2-bench.

Reuses tau2's official `SYSTEM_PROMPT` + `AGENT_INSTRUCTION` and an
`Environment.get_tools()` snapshot so the trainable policy sees the *same*
prompt the SFT teacher saw. Tool-call extraction goes through rLLM's
`QwenToolParser` (matches Qwen3-30B-A3B's native chat template), turning
`<tool_call>{...}</tool_call>` blocks into OpenAI-spec dicts the env can
dispatch directly.

The cumulative `chat_completions` snapshot recorded on every step is what
verl's tokenizer reads at training time (see
`AgentWorkflowEngine.transform_results_for_verl`'s `tokenize_and_mask_cumulative`
path), so the loss is computed over the full multi-turn conversation.
"""
from __future__ import annotations

import copy
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

# Repo on PYTHONPATH (mirrors tau2_rl_env.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # noqa: E402
from rllm.parser import get_tool_parser  # noqa: E402

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT  # noqa: E402
from tau2.registry import registry  # noqa: E402

logger = logging.getLogger(__name__)


class Tau2AssistantAgent(BaseAgent):
    """tau2 customer-service agent reframed as an rLLM `BaseAgent`.

    Lifecycle inside `MultiTurnWorkflow`:
      1. `__init__`: load the domain policy + tool schemas from the tau2
         registry once and bake them into the system prompt.
      2. `reset()`: wipe per-rollout state; `messages` resets to just the
         system prompt.
      3. `update_from_env(observation, ...)`: append a user message
         (`observation["user_content"]`) or one tool message per
         entry in `observation["tool_outputs"]`. Also stamp the previous
         step's reward / done / info so the workflow's terminal reward
         lands on the last step.
      4. `update_from_model(response)`: parse `<tool_call>` blocks. If any
         present, return them as OpenAI-spec dicts; otherwise pass the raw
         text through. Append the assistant message to the conversation
         and snapshot a cumulative `Step`.

    Notes:
      * `domain` is fixed for the lifetime of the agent (one workflow pool
        slot serves one domain). The SFT pipeline is also single-domain
        (telecom), so this matches the existing data layout.
      * `parser_name` defaults to `"qwen"`; bump to `"r1"` only if you swap
        in a DeepSeek-style base model.
    """

    def __init__(
        self,
        domain: str = "telecom",
        parser_name: str = "qwen",
    ):
        self.domain = domain
        self.parser_name = parser_name

        # Policy + tool schemas come from the same tau2 registry the env uses.
        env_constructor = registry.get_env_constructor(domain)
        env = env_constructor()
        self.domain_policy: str = env.get_policy()
        self.tools_openai: list[dict] = [t.openai_schema for t in env.get_tools()]
        # Throw the throwaway env away; the real per-rollout env lives in
        # Tau2Env. We just needed it to read the policy/tool schemas.
        del env

        parser_cls = get_tool_parser(parser_name)
        self.tool_parser = parser_cls()

        # Bake the tools_prompt the parser expects (Qwen format) into the
        # system message so a vanilla chat-completions call (no `tools=`
        # arg) still emits `<tool_call>` blocks the parser understands.
        tools_schema_json = json.dumps(self.tools_openai, indent=2)
        self.tools_prompt: str = self.tool_parser.get_tool_prompt(tools_schema_json)
        self.system_prompt: str = (
            SYSTEM_PROMPT.format(
                domain_policy=self.domain_policy,
                agent_instruction=AGENT_INSTRUCTION,
            )
            + "\n\n"
            + self.tools_prompt
        )

        # Per-rollout state.
        self._trajectory: Trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation: Any = None
        self.reset()

    # ------------------------------------------------------------------
    # rllm BaseAgent API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs,
    ) -> None:
        # Stamp the previous step's reward / done / info first so the
        # workflow's terminal reward lands on the last cumulative Step.
        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = float(reward) if reward is not None else 0.0
            self._trajectory.steps[-1].done = bool(done)
            if info:
                self._trajectory.steps[-1].info.update(info)

        # Append the new observation as messages so the next chat-completions
        # call has the right tail.
        for msg in self._observation_to_messages(observation):
            self.messages.append(msg)

        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        # Always record the assistant text in the conversation, even if it
        # contains tool calls -- matches Qwen3 chat-template behaviour where
        # `<tool_call>` blocks live INSIDE the assistant message.
        assistant_message = {"role": "assistant", "content": response}
        self.messages.append(assistant_message)

        tool_calls_dict: list[dict] = []
        try:
            parsed = self.tool_parser.parse(response or "")
        except Exception as exc:
            logger.warning("tool parser failed: %s", exc)
            parsed = []

        for tool_call in parsed:
            try:
                payload = tool_call.to_dict()
            except Exception:
                payload = {"name": getattr(tool_call, "name", ""), "arguments": getattr(tool_call, "arguments", {})}
            args = payload.get("arguments", {})
            if isinstance(args, dict):
                payload["arguments"] = json.dumps(args)
            tool_calls_dict.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": payload,
                }
            )

        # When tools fired, the env will dispatch them. Otherwise hand the
        # plain text to the env so the user simulator can respond. We pack
        # text into a dict marker the env knows how to detect.
        if tool_calls_dict:
            action_payload: Any = tool_calls_dict
        else:
            stripped = self._strip_tool_call_markers(response or "")
            action_payload = {"_assistant_text": stripped}

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action_payload,
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        return Action(action=action_payload)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _observation_to_messages(obs: Any) -> list[dict]:
        """Map an env observation dict to OpenAI chat messages."""
        if obs is None or obs == {}:
            return []
        if isinstance(obs, dict):
            if "user_content" in obs:
                content = obs["user_content"] or ""
                return [{"role": "user", "content": content}]
            if "tool_outputs" in obs:
                msgs: list[dict] = []
                for tool_call_id, tool_output in obs["tool_outputs"].items():
                    msgs.append(
                        {
                            "role": "tool",
                            "content": str(tool_output) if tool_output is not None else "",
                            "tool_call_id": tool_call_id,
                        }
                    )
                return msgs
            # Fallback: stringify the whole dict so the trajectory doesn't
            # silently drop information.
            return [{"role": "user", "content": json.dumps(obs)}]
        if isinstance(obs, str):
            return [{"role": "user", "content": obs}]
        return [{"role": "user", "content": str(obs)}]

    @staticmethod
    def _strip_tool_call_markers(text: str) -> str:
        """If the model emitted `<tool_call>` blocks AND text, the text is
        what should be sent to the user. We're a bit defensive here -- in
        practice the model picks one or the other, and the env's
        empty/mixed guard catches violations regardless."""
        if "<tool_call>" not in text:
            return text
        keep = []
        i = 0
        while i < len(text):
            start = text.find("<tool_call>", i)
            if start == -1:
                keep.append(text[i:])
                break
            keep.append(text[i:start])
            end = text.find("</tool_call>", start)
            if end == -1:
                break
            i = end + len("</tool_call>")
        return "".join(keep).strip()


__all__ = ["Tau2AssistantAgent"]

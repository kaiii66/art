"""Offline tests for `Tau2AssistantAgent` -- no GPUs / no rllm install needed.

Skipped when rllm isn't importable. Covers:
  * reset() initialises messages with the system prompt that bakes in
    tau2's domain policy + the qwen tool-call instructions.
  * update_from_env appends user / tool messages in OpenAI shape.
  * update_from_model parses `<tool_call>` blocks into OpenAI dicts.
  * update_from_model preserves plain text via `_assistant_text`.
  * trajectory.steps grow cumulatively (chat_completions strictly extends).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("rllm", reason="rllm v0.2.1 not installed in this env")

from onprem.scripts.tau2_rl_agent import Tau2AssistantAgent  # noqa: E402


@pytest.fixture
def agent():
    try:
        return Tau2AssistantAgent(domain="telecom", parser_name="qwen")
    except Exception as exc:
        pytest.skip(f"telecom registry not loadable for agent test: {exc}")


def test_reset_seeds_with_system_prompt(agent):
    agent.reset()
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    sys_content = agent.messages[0]["content"]
    # tau2's AGENT_INSTRUCTION is wrapped inside SYSTEM_PROMPT under <instructions>.
    assert "<instructions>" in sys_content
    assert "<policy>" in sys_content
    # Qwen tool prompt contributes the <tools> + <tool_call> rubric.
    assert "<tools>" in sys_content
    assert "<tool_call>" in sys_content


def test_update_from_env_user_text(agent):
    agent.reset()
    agent.update_from_env({"user_content": "Hi I need help"}, 0.0, False, {})
    assert len(agent.messages) == 2
    assert agent.messages[1] == {"role": "user", "content": "Hi I need help"}


def test_update_from_env_tool_outputs(agent):
    agent.reset()
    agent.update_from_env(
        {"tool_outputs": {"call_a": "result-a", "call_b": "result-b"}},
        0.0, False, {},
    )
    assert len(agent.messages) == 3
    assert agent.messages[1] == {
        "role": "tool", "content": "result-a", "tool_call_id": "call_a",
    }
    assert agent.messages[2] == {
        "role": "tool", "content": "result-b", "tool_call_id": "call_b",
    }


def test_update_from_model_parses_tool_calls(agent):
    agent.reset()
    agent.update_from_env({"user_content": "What's my plan status?"}, 0.0, False, {})

    response = (
        "Let me check.\n"
        '<tool_call>\n{"name": "get_user_plan", "arguments": {"user_id": "u-1"}}\n</tool_call>'
    )
    action = agent.update_from_model(response)

    payload = action.action
    assert isinstance(payload, list), payload
    assert len(payload) == 1
    fn = payload[0]["function"]
    assert fn["name"] == "get_user_plan"
    # Arguments are JSON-stringified for OpenAI compat.
    assert json.loads(fn["arguments"]) == {"user_id": "u-1"}

    # The trajectory grows by one step + chat_completions snapshot includes
    # the assistant message we just added.
    assert len(agent.trajectory.steps) == 1
    last_step = agent.trajectory.steps[-1]
    assert last_step.chat_completions[-1]["role"] == "assistant"
    assert last_step.chat_completions[-1]["content"] == response


def test_update_from_model_plain_text_packs_into_dict(agent):
    agent.reset()
    agent.update_from_env({"user_content": "Hi"}, 0.0, False, {})
    action = agent.update_from_model("Sure, happy to help.")
    assert isinstance(action.action, dict)
    assert action.action == {"_assistant_text": "Sure, happy to help."}


def test_trajectory_is_cumulative_across_turns(agent):
    agent.reset()
    agent.update_from_env({"user_content": "Hi"}, 0.0, False, {})
    agent.update_from_model("Hello! How can I help?")
    agent.update_from_env({"user_content": "I need to change my plan."}, 0.0, False, {})
    agent.update_from_model(
        '<tool_call>\n{"name": "get_user_plan", "arguments": {"user_id": "u-1"}}\n</tool_call>'
    )
    agent.update_from_env(
        {"tool_outputs": {"call_x": "current plan: basic"}}, 0.0, False, {},
    )
    agent.update_from_model("You're on the basic plan. Want to upgrade?")

    # 3 steps, each carrying a strictly extending chat_completions snapshot.
    assert len(agent.trajectory.steps) == 3
    assert agent.trajectory.is_cumulative()


def test_strip_tool_call_markers():
    text = "Sure, calling now. <tool_call>{...}</tool_call> All done."
    cleaned = Tau2AssistantAgent._strip_tool_call_markers(text)
    assert cleaned == "Sure, calling now.  All done.".strip()


def test_reward_done_info_propagate_to_last_step(agent):
    agent.reset()
    agent.update_from_env({"user_content": "Hi"}, 0.0, False, {})
    agent.update_from_model("Hello!")

    agent.update_from_env(
        {"user_content": "Goodbye"},
        reward=0.7,
        done=True,
        info={"termination_reason": "agent_stop"},
    )

    last = agent.trajectory.steps[-1]
    assert last.reward == 0.7
    assert last.done is True
    assert last.info.get("termination_reason") == "agent_stop"

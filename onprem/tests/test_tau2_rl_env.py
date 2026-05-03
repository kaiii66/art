"""Offline tests for `Tau2Env` -- no GPUs / no rllm install required.

Stubs out:
  * `tau2.user.user_simulator.UserSimulator.generate_next_message` so the
    user-sim never hits W&B Inference (so these run with no API keys).
  * Skips the whole module if `rllm` isn't importable (env code does
    `from rllm.environments.base.base_env import BaseEnv`).

Covers:
  - reset(): first user message is captured + observation shape is right
  - step(text): user-sim invoked, conversation grows, reward stays 0
  - step(tool calls): environment.get_response dispatched + outputs returned
  - step(empty action): terminates with AGENT_ERROR
  - step(mixed action): terminates with AGENT_ERROR
  - step(STOP token): terminates with AGENT_STOP
  - max_steps: terminates with MAX_STEPS at boundary
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("rllm", reason="rllm v0.2.1 not installed in this env")

from rllm.agents.agent import Action  # noqa: E402

from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import TerminationReason  # noqa: E402
from tau2.user.user_simulator import UserSimulator  # noqa: E402

from onprem.scripts.tau2_rl_env import Tau2Env  # noqa: E402


def _stub_user_responses(*responses: str):
    """Patch UserSimulator.generate_next_message to emit a fixed sequence."""
    state_holder = {"i": 0}

    def fake(self, message, state):  # noqa: ARG001
        idx = min(state_holder["i"], len(responses) - 1)
        content = responses[idx]
        state_holder["i"] += 1
        msg = UserMessage(role="user", content=content)
        state.messages.append(msg)
        return msg, state

    return patch.object(UserSimulator, "generate_next_message", fake)


def _make_env(max_steps: int = 5) -> Tau2Env:
    return Tau2Env(
        domain="telecom",
        max_steps=max_steps,
        user_llm="stub",
        user_llm_args={},
        shaped_reward_weights={
            "action": 0.5, "communicate": 0.0, "nl_assertions": 0.0,
            "termination": 0.5, "tool_accuracy": 0.0,
            "tool_arg_accuracy": 0.0, "step_penalty": 0.0,
        },
    )


# ---------------------------------------------------------------------------

def _first_telecom_task_id() -> str:
    """Pull the first telecom task id from tau2's registry. Skip if missing."""
    try:
        from tau2.registry import registry
        tasks = registry.get_tasks_loader("telecom")(task_split_name=None)
    except Exception as exc:
        pytest.skip(f"telecom task registry not loadable: {exc}")
    if not tasks:
        pytest.skip("telecom task list is empty")
    return tasks[0].id


def test_reset_returns_user_observation():
    task_id = _first_telecom_task_id()
    env = _make_env()

    with _stub_user_responses("Hi, I have a problem with my data."):
        obs, info = env.reset({"task_id": task_id, "domain": "telecom"})

    assert "user_content" in obs
    assert obs["user_content"] == "Hi, I have a problem with my data."
    assert info == {} or info.get("termination_reason") is None
    assert env.step_count == 0
    assert env.done is False
    # _messages should hold (first agent stub, first user reply).
    assert len(env._messages) == 2


def test_step_text_action_returns_user_followup():
    task_id = _first_telecom_task_id()
    env = _make_env()

    with _stub_user_responses(
        "Hi, my plan stopped working.",
        "Yes, please look at my account.",
    ):
        env.reset({"task_id": task_id, "domain": "telecom"})
        obs, reward, done, info = env.step(Action(action={"_assistant_text": "Sure, I can help."}))

    assert reward == 0.0
    assert done is False
    assert obs["user_content"] == "Yes, please look at my account."
    # Conversation should be: agent stub, user1, agent reply, user2.
    assert len(env._messages) == 4
    assert isinstance(env._messages[2], AssistantMessage)
    assert env._messages[2].content == "Sure, I can help."


def test_step_tool_calls_dispatched_to_environment():
    task_id = _first_telecom_task_id()
    env = _make_env()

    with _stub_user_responses("Help me change my plan."):
        env.reset({"task_id": task_id, "domain": "telecom"})

    # Pick a tool the telecom environment actually exposes so get_response
    # returns a ToolMessage (its content / error don't matter for this test).
    tool_name = next(iter(env.tools_openai))["function"]["name"]
    tool_calls = [{
        "id": "call_xyz",
        "type": "function",
        "function": {"name": tool_name, "arguments": {}},
    }]

    obs, reward, done, info = env.step(Action(action=tool_calls))

    assert reward == 0.0
    assert done is False
    assert "tool_outputs" in obs
    assert "call_xyz" in obs["tool_outputs"]
    # _messages got: stub, user1, assistant_with_tool_calls, ToolMessage.
    assert len(env._messages) >= 4
    assert isinstance(env._messages[-1], ToolMessage)


def test_empty_action_terminates_with_agent_error():
    task_id = _first_telecom_task_id()
    env = _make_env()
    with _stub_user_responses("Hello."):
        env.reset({"task_id": task_id, "domain": "telecom"})
        obs, reward, done, info = env.step(Action(action={"_assistant_text": ""}))

    assert done is True
    assert info["termination_reason"] == TerminationReason.AGENT_ERROR.value
    assert info["violation"] == "empty_assistant_message"
    assert "shaped_reward" in info


def test_mixed_text_and_tool_calls_terminates_with_agent_error():
    task_id = _first_telecom_task_id()
    env = _make_env()
    with _stub_user_responses("Hello."):
        env.reset({"task_id": task_id, "domain": "telecom"})
        # Crafted bad payload: a non-empty assistant text PLUS a tool_calls
        # list. _classify_action will return both => AGENT_ERROR.
        # We stitch via a raw list+dict payload to simulate a confused agent.
        bad = {"_assistant_text": "let me check that"}
        obs1, _, done1, _info1 = env.step(Action(action=bad))
        assert done1 is False
        # Now actually trigger the mixed case by short-circuiting through
        # the env's classifier with a list-of-calls and assistant text in
        # one synthesised tuple. The agent never actually emits this; we
        # exercise the guard directly.
        env._classify_action_test_hook = True  # type: ignore[attr-defined]
        # Inject by monkeypatching the classifier for a single call:
        from onprem.scripts import tau2_rl_env as env_mod
        original = env_mod.Tau2Env._classify_action
        try:
            env_mod.Tau2Env._classify_action = staticmethod(
                lambda action: ([{"id": "x", "type": "function", "function": {"name": "noop", "arguments": {}}}], "and some chatter")
            )
            obs2, _, done2, info2 = env.step(Action(action="anything"))
        finally:
            env_mod.Tau2Env._classify_action = original
    assert done2 is True
    assert info2["termination_reason"] == TerminationReason.AGENT_ERROR.value
    assert info2["violation"] == "mixed_assistant_message"


def test_agent_stop_token_terminates_with_agent_stop():
    task_id = _first_telecom_task_id()
    env = _make_env()
    with _stub_user_responses("Hi."):
        env.reset({"task_id": task_id, "domain": "telecom"})
        obs, _, done, info = env.step(Action(action={"_assistant_text": "Bye! ###STOP###"}))

    assert done is True
    assert info["termination_reason"] == TerminationReason.AGENT_STOP.value


def test_max_steps_termination():
    task_id = _first_telecom_task_id()
    env = _make_env(max_steps=2)
    user_seq = [f"reply {i}" for i in range(20)]

    with _stub_user_responses(*user_seq):
        env.reset({"task_id": task_id, "domain": "telecom"})
        # step 1 (-> user reply)
        env.step(Action(action={"_assistant_text": "ok"}))
        # step 2 (-> user reply, hits max_steps boundary)
        obs, _, done, info = env.step(Action(action={"_assistant_text": "ok"}))

    assert done is True
    assert info["termination_reason"] == TerminationReason.MAX_STEPS.value


def test_user_stop_terminates_with_user_stop():
    task_id = _first_telecom_task_id()
    env = _make_env()
    with _stub_user_responses("Hi.", "###TRANSFER###"):
        env.reset({"task_id": task_id, "domain": "telecom"})
        obs, _, done, info = env.step(Action(action={"_assistant_text": "anything"}))

    assert done is True
    assert info["termination_reason"] == TerminationReason.USER_STOP.value


def test_from_dict_factory():
    env_args = {
        "domain": "telecom",
        "max_steps": 7,
        "user_llm": "stub",
        "user_llm_args": {"temperature": 1.0},
        "shaped_reward_weights": {},
    }
    env = Tau2Env.from_dict({**env_args, "task_id": "should-be-stripped"})
    assert env.domain == "telecom"
    assert env.max_steps == 7

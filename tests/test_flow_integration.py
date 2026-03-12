"""End-to-end flow tests: full call lifecycle with accumulator and payload build.

These tests simulate a complete call flow (start → user/LLM/TTS/bot events → node
transitions → end) and assert the resulting payload and transcript enrichment.
No pipecat runtime required.
"""

from unittest.mock import MagicMock

import pytest

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.config import TunerConfig


@pytest.fixture
def config():
    return TunerConfig(
        api_key="key",
        workspace_id=1,
        agent_id="agent",
        call_id="call-e2e",
        base_url="https://tuner.test",
        asr_model="asr",
        llm_model="llm",
        tts_model="tts",
    )


def test_full_call_flow_single_turn(config):
    """Simulate one user turn and one bot response; verify payload and transcript."""
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000

    # Start
    acc.on_start(base_ns)
    assert acc.call_start_abs_ns == base_ns

    # Initial node (e.g. greeting)
    acc.on_node_entered(
        from_node=None,
        to_node="greeting",
        node_config={"functions": [], "task_messages": []},
        trigger=None,
        state_snapshot={},
        timestamp_ns=base_ns,
    )
    assert acc._current_node == "greeting"
    assert len(acc.node_transitions) == 1

    # User speaks
    acc.on_user_started(base_ns + 50_000_000)
    acc.on_user_stopped(MagicMock(stop_secs=0), base_ns + 100_000_000)
    acc.on_llm_started(base_ns + 150_000_000)
    acc.on_tts_started(base_ns + 200_000_000)
    acc.on_tts_text_chars(MagicMock(text="Hello there"))
    acc.on_bot_started_speaking(base_ns + 250_000_000)
    acc.on_bot_stopped(base_ns + 400_000_000)

    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node == "greeting"
    assert turn.user_stopped_ms == 100
    assert turn.bot_started_ms == 250
    assert turn.bot_stopped_ms == 400
    assert turn.ttfb_ms == 100
    assert acc._tts_chars == 11

    # End
    acc.on_call_end(base_ns + 500_000_000)
    assert acc.done
    assert acc.call_end_abs_ns == base_ns + 500_000_000

    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there!"},
    ]
    payload = acc.build_payload(config, transcript)

    assert payload.call_id == config.call_id
    assert payload.duration_ms == 500
    assert payload.call_status == "call_ended"
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 11
    assert len(payload.transcript_with_tool_calls) >= 2
    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "user" in roles
    assert "agent" in roles


def test_full_call_flow_with_tool_and_node_transition(config):
    """Simulate user → tool call → node transition → assistant response."""
    acc = FlowsAccumulator()
    base_ns = 2_000_000_000

    acc.on_start(base_ns)
    acc.on_node_entered(
        from_node=None,
        to_node="greeting",
        node_config={"functions": [{"name": "transfer"}], "task_messages": []},
        trigger=None,
        state_snapshot={},
        timestamp_ns=base_ns,
    )

    # First turn: user asks for transfer
    acc.on_user_started(base_ns + 10_000_000)
    acc.on_user_stopped(MagicMock(stop_secs=0), base_ns + 50_000_000)
    acc.on_function_call_in_progress(
        MagicMock(function_name="transfer", arguments={"to": "sales"}),
        base_ns + 100_000_000,
    )
    assert acc.get_pending_transition() is not None
    assert acc.get_pending_transition().function_name == "transfer"

    # Node transition (flow manager calls patched _set_node)
    acc.on_node_entered(
        from_node="greeting",
        to_node="transfer",
        node_config={"functions": [{"name": "hangup"}], "task_messages": []},
        trigger=acc.get_pending_transition(),
        state_snapshot={"transferred": True},
        timestamp_ns=base_ns + 120_000_000,
    )
    assert acc._current_node == "transfer"
    assert acc.get_pending_transition() is None
    assert len(acc.node_transitions) == 2

    acc.on_llm_started(base_ns + 130_000_000)
    acc.on_tts_started(base_ns + 180_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    acc.on_bot_stopped(base_ns + 350_000_000)
    acc.on_call_end(base_ns + 400_000_000)

    transcript = [
        {"role": "user", "content": "Transfer me to sales"},
        {"role": "assistant", "tool_calls": [{
            "id": "tc-1",
            "function": {"name": "transfer", "arguments": '{"to": "sales"}'},
        }]},
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Transferring you now."},
    ]
    payload = acc.build_payload(config, transcript)

    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "agent_function" in roles
    assert "agent_result" in roles
    assert "node_transition" in roles
    # Initial + one triggered transition
    transitions = [s for s in payload.transcript_with_tool_calls if s.role == "node_transition"]
    assert len(transitions) >= 2  # initial → greeting, greeting → transfer

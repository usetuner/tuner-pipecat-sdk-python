"""End-to-end flow tests: runtime observer flow with accumulator and payload build."""
from types import SimpleNamespace

import pytest

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.config import TunerConfig


def _metric(cls_name: str, **kwargs):
    return type(cls_name, (), kwargs)()


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
        trigger=None,
        timestamp_ns=base_ns,
    )
    assert len(acc.node_transitions) == 1
    assert acc.node_transitions[-1].to_node == "greeting"

    # TurnTrackingObserver fires on_turn_started when user begins speaking
    acc.on_turn_started(1, base_ns + 50_000_000)  # user started at +50ms

    # Runtime observer data
    acc.on_latency_measured(0.15)
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=11)]))
    acc._pending_pipecat_llm_processing_s = 0.05
    acc._pending_pipecat_tts_processing_s = 0.05
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=1.05,
            user_turn_secs=0.05,
            ttfb=[SimpleNamespace(duration_secs=0.1)],
            function_calls=[],
        )
    )
    acc.on_bot_stopped(base_ns + 400_000_000)

    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node == "greeting"
    assert turn.user_stopped_ms == 100
    assert turn.bot_started_ms == 250
    assert turn.bot_stopped_ms == 400
    assert turn.ttfb_ms == 100

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
        trigger=None,
        timestamp_ns=base_ns,
    )

    # First turn: user asks for transfer
    acc.on_function_call_in_progress(
        SimpleNamespace(function_name="transfer", arguments={"to": "sales"}, tool_call_id="tc-1"),
        base_ns + 100_000_000,
    )
    assert acc.get_pending_transition() is not None
    assert acc.get_pending_transition().function_name == "transfer"

    # Node transition (flow manager calls patched _set_node)
    acc.on_node_entered(
        from_node="greeting",
        to_node="transfer",
        trigger=acc.get_pending_transition(),
        timestamp_ns=base_ns + 120_000_000,
    )
    assert acc.node_transitions[-1].to_node == "transfer"
    assert acc.get_pending_transition() is None
    assert len(acc.node_transitions) == 2
    triggered_transition = next(t for t in acc.node_transitions if t.trigger_function)
    assert triggered_transition.trigger_timestamp_ms == 100  # _rel_ms(base_ns + 100ms)

    acc.on_turn_started(1, base_ns + 10_000_000)  # user started speaking at +10ms
    acc.on_latency_measured(0.15)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=2.01,
            user_turn_secs=0.04,
            ttfb=[SimpleNamespace(duration_secs=0.07)],
            function_calls=[
                SimpleNamespace(function_name="transfer", start_time=2.1, duration_secs=0.02)
            ],
        )
    )
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

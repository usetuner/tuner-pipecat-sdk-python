"""Accumulator transcript enrichment with tools and transitions."""

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn, NodeTransitionRecord


def test_enrich_transcript_tool_call_and_result_and_node_transition(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.node_transitions = [
        NodeTransitionRecord(
            from_node="greeting",
            to_node="transfer",
            trigger_function="transfer",
            trigger_args={"to": "sales"},
            state_snapshot={},
            task_messages=[],
            functions_available=[],
            timestamp_ms=100,
            trigger_timestamp_ms=60,
        )
    ]
    transcript = [
        {"role": "user", "content": "Transfer me"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc-1",
                    "function": {"name": "transfer", "arguments": '{"to": "sales"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done."},
    ]
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="greeting",
            ttfb_ms=50,
            llm_ms=30,
            tts_ms=20,
            bot_started_ms=100,
            user_stopped_ms=0,
            user_started_ms=0,
            bot_stopped_ms=150,
        ),
        LatencyTurn(
            turn_index=1,
            node="transfer",
            ttfb_ms=50,
            llm_ms=30,
            tts_ms=20,
            bot_started_ms=200,
            user_stopped_ms=150,
            user_started_ms=100,
            bot_stopped_ms=250,
        ),
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert "agent_function" in roles
    assert "agent_result" in roles
    assert "node_transition" in roles
    user_segments = [
        segment for segment in payload.transcript_with_tool_calls if segment.role == "user"
    ]
    agent_segments = [
        segment for segment in payload.transcript_with_tool_calls if segment.role == "agent"
    ]
    assert "asr_node_ttft" not in user_segments[0].metadata
    assert agent_segments[0].metadata["tts_node_ttfb"] == 50

    # agent_function uses trigger_timestamp_ms (function invocation); agent_result uses timestamp_ms (node switch)
    func_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent_function"]
    result_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent_result"]
    assert func_segments, "expected agent_function segment"
    assert result_segments, "expected agent_result segment"
    assert func_segments[0].start_ms == 60   # trigger_timestamp_ms
    assert result_segments[0].start_ms == 100  # timestamp_ms (node switch)
    assert func_segments[0].start_ms != result_segments[0].start_ms


def test_enrich_transcript_initial_node_transition(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.node_transitions = [
        NodeTransitionRecord(
            from_node=None,
            to_node="start",
            trigger_function=None,
            trigger_args=None,
            state_snapshot={"x": 1},
            task_messages=[],
            functions_available=["next"],
            timestamp_ms=0,
        )
    ]
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi!"},
    ]
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="start",
            ttfb_ms=0,
            llm_ms=0,
            tts_ms=0,
            bot_started_ms=0,
            user_stopped_ms=0,
            user_started_ms=0,
            bot_stopped_ms=100,
        )
    ]
    payload = acc.build_payload(tuner_config, transcript)
    first = payload.transcript_with_tool_calls[0]
    assert first.role == "node_transition"
    assert first.node is not None
    assert first.node.to == "start"

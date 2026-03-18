"""Accumulator transcript enrichment with tools and transitions."""

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn, NodeTransitionRecord


def test_enrich_transcript_tool_call_and_result_and_node_transition(tuner_config):
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-1", base_ns + 60_000_000)
    acc.registry.record_completion_ns("tc-1", base_ns + 90_000_000)
    acc.node_transitions = [
        NodeTransitionRecord(
            from_node="greeting",
            to_node="transfer",
            trigger_function="transfer",
            trigger_args={"to": "sales"},
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
    transition_segments = [
        segment
        for segment in payload.transcript_with_tool_calls
        if segment.role == "node_transition"
    ]
    assert transition_segments
    assert "state_snapshot" not in transition_segments[0].metadata
    assert "functions_available" not in transition_segments[0].metadata
    assert "task_messages" not in transition_segments[0].metadata
    user_segments = [
        segment for segment in payload.transcript_with_tool_calls if segment.role == "user"
    ]
    agent_segments = [
        segment for segment in payload.transcript_with_tool_calls if segment.role == "agent"
    ]
    assert "asr_node_ttft" not in user_segments[0].metadata
    assert agent_segments[0].metadata["tts_node_ttfb"] == 50

    # Tool timings come strictly from tool_call_id registry.
    func_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent_function"]
    result_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent_result"]
    assert func_segments, "expected agent_function segment"
    assert result_segments, "expected agent_result segment"
    assert func_segments[0].start_ms == 60
    assert result_segments[0].start_ms == 90
    assert func_segments[0].start_ms != result_segments[0].start_ms


def test_consecutive_assistant_messages_merged_into_one_segment(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.node_transitions = []
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="greeting",
            ttfb_ms=10,
            llm_ms=20,
            tts_ms=30,
            bot_started_ms=100,
            user_stopped_ms=50,
            user_started_ms=10,
            bot_stopped_ms=300,
        ),
    ]
    transcript = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there,"},
        {"role": "assistant", "content": "how can I help?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Hi there, how can I help?"
    assert agent_segments[0].start_ms == 100
    assert agent_segments[0].end_ms == 300


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
    assert "state_snapshot" not in first.metadata
    assert "functions_available" not in first.metadata


def test_enrich_transcript_uses_assistant_turn_events_to_skip_ghost_messages(tuner_config):
    # Ghost messages appear in the same user-turn window as the spoken text, before a tool call
    # triggers a node transition. The last plain assistant text in the window is the spoken one.
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="n1",
            ttfb_ms=20,
            llm_ms=10,
            tts_ms=10,
            bot_started_ms=300,
            user_stopped_ms=200,
            user_started_ms=100,
            bot_stopped_ms=500,
        )
    ]
    # Real-world ghost structure: ghost comes before a tool call in the same window;
    # the spoken text is the last plain assistant message before the next user turn.
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Draft answer"},  # ghost — not the last in window
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "choose", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "Spoken answer"},  # spoken — last plain text in window
    ]

    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 2
    assert agent_segments[0].text == "Draft answer"
    assert agent_segments[0].start_ms == 0
    assert agent_segments[1].text == "Spoken answer"
    assert agent_segments[1].start_ms == 300


def test_last_plain_assistant_in_window_gets_turn_by_order_not_text(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="n1",
            ttfb_ms=30,
            llm_ms=10,
            tts_ms=10,
            bot_started_ms=300,
            user_stopped_ms=200,
            user_started_ms=100,
            bot_stopped_ms=600,
        )
    ]
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Draft answer"},
        {"role": "assistant", "content": "Final spoken answer"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Draft answer Final spoken answer"
    assert agent_segments[0].start_ms == 300
    assert agent_segments[0].end_ms == 600


def test_agent_metadata_node_prefers_bot_node_at_response_time(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="greeting",
            bot_node="size",
            ttfb_ms=30,
            llm_ms=10,
            tts_ms=10,
            bot_started_ms=300,
            user_stopped_ms=200,
            user_started_ms=100,
            bot_stopped_ms=600,
        )
    ]
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "What size pizza would you like?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent")
    assert agent_seg.metadata["node"] == "size"


def test_all_trailing_assistant_messages_after_last_user_are_spoken(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="confirm",
            bot_node="farewell",
            ttfb_ms=20,
            llm_ms=10,
            tts_ms=10,
            bot_started_ms=800,
            user_stopped_ms=700,
            user_started_ms=600,
            bot_stopped_ms=1200,
        )
    ]
    transcript = [
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "Thank you for your order!"},
        {"role": "assistant", "content": "Enjoy your meal!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Thank you for your order! Enjoy your meal!"
    assert agent_segments[0].start_ms == 800


def test_agent_result_uses_registry_completion_when_available(tuner_config):
    """agent_result.start_ms uses the registry completion time, not invocation_ms."""
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_completion_ns("call_xyz", base_ns + 350_000_000)

    transcript = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_xyz", "function": {"name": "greet", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_xyz", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done!"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    result_segs = [s for s in payload.transcript_with_tool_calls if s.role == "agent_result"]
    assert len(result_segs) == 1
    # start_ms must use registry completion (350), not fall back to invocation_ms (0)
    assert result_segs[0].start_ms == 350
    assert result_segs[0].end_ms is None


def test_parallel_same_name_tools_use_distinct_invocation_ms_by_id(tuner_config):
    """Two add_topping calls with different tool_call_ids get distinct invocation times."""
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-a", base_ns + 100_000_000)
    acc.registry.record_invocation_ns("tc-b", base_ns + 200_000_000)

    transcript = [
        {"role": "user", "content": "add mushrooms and olives"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc-a",
                    "function": {"name": "add_topping", "arguments": '{"topping": "mushrooms"}'},
                },
                {
                    "id": "tc-b",
                    "function": {"name": "add_topping", "arguments": '{"topping": "olives"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc-a", "content": '{"ok": true}'},
        {"role": "tool", "tool_call_id": "tc-b", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Added both!"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    func_segs = [s for s in payload.transcript_with_tool_calls if s.role == "agent_function"]
    assert len(func_segs) == 2
    assert func_segs[0].start_ms == 100
    assert func_segs[1].start_ms == 200


def test_agent_result_does_not_fallback_to_transition_timestamp(tuner_config):
    """Without registry completion, agent_result timing stays unset (0)."""
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-1", base_ns + 75_000_000)
    acc.node_transitions = [
        NodeTransitionRecord(
            from_node="greeting",
            to_node="next",
            trigger_function="transfer",
            trigger_args={},
            timestamp_ms=200,
            trigger_timestamp_ms=75,
        )
    ]
    transcript = [
        {"role": "user", "content": "transfer"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc-1", "function": {"name": "transfer", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "done"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    result_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent_result")
    assert result_seg.start_ms == 0


def test_payload_monotonic_guard_corrects_agent_end_before_start(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="greeting",
            ttfb_ms=10,
            llm_ms=20,
            tts_ms=30,
            bot_started_ms=5000,
            user_stopped_ms=1000,
            user_started_ms=500,
            bot_stopped_ms=2000,
        )
    ]
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent")
    assert agent_seg.start_ms == 5000
    assert agent_seg.end_ms == 5000

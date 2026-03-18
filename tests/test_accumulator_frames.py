"""Accumulator frame-event tests for turns and transitions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn, NodeTransitionRecord, PendingTransition


def test_on_bot_stopped_updates_last_turn():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node=None,
            ttfb_ms=100,
            llm_ms=50,
            tts_ms=50,
            bot_started_ms=200,
            user_stopped_ms=100,
            user_started_ms=50,
            bot_stopped_ms=None,
        )
    ]
    acc._bot_turn_idx = 0
    acc.on_bot_stopped(1 + 500 * 1_000_000)
    assert acc.latency_turns[-1].bot_stopped_ms == 500


def test_on_bot_stopped_no_op_when_done():
    acc = FlowsAccumulator()
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node=None,
            ttfb_ms=0,
            llm_ms=0,
            tts_ms=0,
            bot_started_ms=0,
            user_stopped_ms=0,
            user_started_ms=0,
            bot_stopped_ms=None,
        )
    ]
    acc.on_bot_stopped(999_000_000)
    assert acc.latency_turns[-1].bot_stopped_ms is None


def test_on_function_call_in_progress_sets_pending():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1
    frame = MagicMock(function_name="transfer", arguments={"to": "sales"}, tool_call_id="tc-1")
    acc.on_function_call_in_progress(frame, 1 + 300 * 1_000_000)
    pending = acc.get_pending_transition()
    assert pending is not None
    assert pending.function_name == "transfer"
    assert pending.arguments == {"to": "sales"}
    assert pending.timestamp_ms == 300


def test_on_function_call_in_progress_records_invocation_in_registry():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    frame = MagicMock(function_name="add_topping", arguments={}, tool_call_id="tc-xyz")
    acc.on_function_call_in_progress(frame, 1_000_000_000 + 200_000_000)
    assert acc.get_tool_invocation_ms("tc-xyz") == 200


def test_on_function_call_result_records_completion_in_registry():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_function_call_result("call_abc", 1_000_000_000 + 250_000_000)
    assert acc.get_tool_completion_ms("call_abc") == 250


def test_on_node_entered_appends_transition_and_clears_pending():
    acc = FlowsAccumulator()
    acc._pending_transition = PendingTransition(
        function_name="transfer", arguments={"to": "sales"}, timestamp_ms=100
    )
    acc.call_start_abs_ns = 1
    acc.on_node_entered(
        from_node="greeting",
        to_node="transfer",
        node_config={"functions": [{"name": "hangup"}], "task_messages": []},
        trigger=acc._pending_transition,
        state_snapshot={"x": 1},
        timestamp_ns=1 + 200 * 1_000_000,
    )
    assert len(acc.node_transitions) == 1
    record = acc.node_transitions[0]
    assert record.from_node == "greeting"
    assert record.to_node == "transfer"
    assert record.trigger_function == "transfer"
    assert record.trigger_args == {"to": "sales"}
    assert record.state_snapshot == {"x": 1}
    assert record.functions_available == ["hangup"]
    assert record.timestamp_ms == 200
    assert acc._current_node == "transfer"
    assert acc._pending_transition is None


def test_on_node_entered_extracts_function_names_from_objects():
    acc = FlowsAccumulator()

    class FunctionObject:
        def __init__(self, name: str) -> None:
            self.name = name

    acc.on_node_entered(
        from_node=None,
        to_node="start",
        node_config={
            "functions": [FunctionObject("a"), {"name": "b"}],
            "task_messages": [],
        },
        trigger=None,
        state_snapshot={},
        timestamp_ns=0,
    )
    assert acc.node_transitions[0].functions_available == ["a", "b"]


def test_usage_counter_accessors():
    acc = FlowsAccumulator()
    llm_metric = type("LLMUsageMetricsData", (), {"value": SimpleNamespace(total_tokens=42)})
    tts_metric = type("TTSUsageMetricsData", (), {"value": 99})
    acc.on_metrics_frame(SimpleNamespace(data=[llm_metric(), tts_metric()]))
    assert acc.get_total_llm_tokens() == 42
    assert acc.get_total_tts_characters() == 99


def test_on_turn_started_creates_latency_turn():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc._current_node = "greeting"
    acc.on_turn_started(1, 1_000_000_000 + 400_000_000)
    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node == "greeting"
    assert turn.user_started_ms == 400
    assert acc._active_turn_number == 1


def test_on_turn_started_backfills_initial_transition_timestamp_when_zero():
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.node_transitions.append(
        NodeTransitionRecord(
            from_node=None,
            to_node="greeting",
            trigger_function=None,
            trigger_args=None,
            state_snapshot={},
            task_messages=[],
            functions_available=[],
            timestamp_ms=0,
        )
    )
    acc.on_turn_started(1, base_ns + 350_000_000)
    assert acc.node_transitions[0].timestamp_ms == 350


def test_on_bot_started_speaking_sets_turn_bot_node():
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc._current_node = "delivery_or_pickup"
    acc.on_turn_started(1, base_ns + 100_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    assert acc.latency_turns[0].bot_node == "delivery_or_pickup"


def test_on_turn_ended_sets_was_interrupted():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_turn_ended(1, was_interrupted=True)
    assert acc.latency_turns[0].was_interrupted is True
    assert acc._active_turn_number is None


def test_on_latency_breakdown_enriches_existing_turn():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc._current_node = "greeting"
    acc._pending_pipecat_llm_processing_s = 0.03
    acc._pending_pipecat_tts_processing_s = 0.07
    acc.on_latency_measured(0.2)

    # Turn must already exist (created by on_turn_started)
    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)

    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(duration_secs=0.05)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.node == "greeting"
    assert turn.user_started_ms == 500
    assert turn.user_stopped_ms == 700
    assert turn.bot_started_ms == 900
    assert turn.ttfb_ms == 50
    assert turn.llm_ms == 30
    assert turn.tts_ms == 70


def test_on_latency_breakdown_keeps_bot_started_unset_when_latency_missing():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.2,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(duration_secs=0.04)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)
    assert acc.latency_turns[0].bot_started_ms == 0
    assert acc.latency_turns[0].ttfb_ms == 40


def test_on_latency_breakdown_preserves_user_started_ms_when_user_turn_start_time_missing():
    """Keep on_turn_started timestamp when breakdown start time is missing."""
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)  # user_started_ms = 500
    acc.on_latency_measured(0.2)

    breakdown = SimpleNamespace(
        user_turn_start_time=None,
        user_turn_secs=None,
        ttfb=[SimpleNamespace(duration_secs=0.05)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    turn = acc.latency_turns[0]
    assert turn.user_started_ms == 500  # preserved from on_turn_started
    assert turn.user_stopped_ms == 0   # unknown (no fallback)
    assert turn.bot_started_ms == 200  # 0 + 200ms latency


def test_on_latency_breakdown_skips_when_no_active_turn(caplog):
    """on_latency_breakdown logs a warning and skips when no turn is active."""
    import logging
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[],
        function_calls=[],
    )
    with caplog.at_level(logging.WARNING):
        acc.on_latency_breakdown(breakdown)
    assert not acc.latency_turns


def test_on_call_end_marks_done_without_creating_turns():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert not acc.latency_turns
    acc.on_call_end(1_000_000_000 + 500_000_000)
    assert acc.done
    assert acc.call_end_abs_ns == 1_000_000_000 + 500_000_000
    assert not acc.latency_turns  # no synthetic turns created

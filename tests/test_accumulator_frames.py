"""Accumulator frame-event tests for turns and transitions."""

from unittest.mock import MagicMock

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn, PendingTransition


def test_on_user_started_sets_once():
    acc = FlowsAccumulator()
    acc.on_user_started(100)
    acc.on_user_started(200)
    assert acc._user_started_ns == 100


def test_on_user_stopped_sets_anchor_and_clears_llm_tts_bot():
    acc = FlowsAccumulator()
    acc._current_node = "greeting"
    acc.on_user_stopped(MagicMock(stop_secs=0), 200)
    assert acc._user_stopped_ns == 200
    assert acc._llm_started_ns == 0
    assert acc._tts_started_ns == 0
    assert acc._latency_node == "greeting"


def test_on_user_stopped_applies_stop_correction():
    acc = FlowsAccumulator()
    frame = MagicMock(stop_secs=0.5)
    acc.on_user_stopped(frame, 2_000_000_000)
    assert acc._user_stopped_ns == 1_500_000_000


def test_on_llm_started_sets_once_after_user_stopped():
    acc = FlowsAccumulator()
    acc._user_stopped_ns = 100
    acc.on_llm_started(150)
    acc.on_llm_started(160)
    assert acc._llm_started_ns == 150


def test_on_llm_started_ignored_before_user_stopped():
    acc = FlowsAccumulator()
    acc.on_llm_started(100)
    assert acc._llm_started_ns == 0


def test_on_tts_started_sets_once_after_user_stopped():
    acc = FlowsAccumulator()
    acc._user_stopped_ns = 100
    acc.on_tts_started(200)
    acc.on_tts_started(210)
    assert acc._tts_started_ns == 200


def test_on_tts_text_chars_accumulates():
    acc = FlowsAccumulator()
    acc.on_tts_text_chars(MagicMock(text="Hello"))
    acc.on_tts_text_chars(MagicMock(text=" world"))
    assert acc._tts_chars == 11


def test_on_tts_text_chars_handles_empty():
    acc = FlowsAccumulator()
    acc.on_tts_text_chars(MagicMock(text=""))
    assert acc._tts_chars == 0


def test_on_bot_started_speaking_flushes_latency_turn():
    base_ns = 1_000_000_000
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = base_ns
    acc._user_started_ns = base_ns + 50_000_000
    acc._user_stopped_ns = base_ns + 100_000_000
    acc._llm_started_ns = base_ns + 150_000_000
    acc._tts_started_ns = base_ns + 200_000_000
    acc._latency_node = "greeting"

    acc.on_bot_started_speaking(base_ns + 250_000_000)

    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node == "greeting"
    assert turn.bot_started_ms == 250
    assert turn.user_stopped_ms == 100
    assert turn.user_started_ms == 50
    assert turn.llm_ms == 50
    assert turn.tts_ms == 50
    assert turn.ttfb_ms == 100
    assert acc._user_stopped_ns == 0


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
    frame = MagicMock(function_name="transfer", arguments={"to": "sales"})
    acc.on_function_call_in_progress(frame, 1 + 300 * 1_000_000)
    pending = acc.get_pending_transition()
    assert pending is not None
    assert pending.function_name == "transfer"
    assert pending.arguments == {"to": "sales"}
    assert pending.timestamp_ms == 300


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

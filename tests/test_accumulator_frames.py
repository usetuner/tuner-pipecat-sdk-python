"""Accumulator frame-event tests for turns and transitions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tuner_pipecat_sdk.accumulator import CallAccumulator
from tuner_pipecat_sdk.models import LatencyTurn


def test_on_bot_stopped_updates_last_turn():
    acc = CallAccumulator()
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
    acc = CallAccumulator()
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


def test_on_function_call_in_progress_records_invocation_in_registry():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    frame = MagicMock(function_name="add_topping", arguments={}, tool_call_id="tc-xyz")
    acc.on_function_call_in_progress(frame, 1_000_000_000 + 200_000_000)
    assert acc.get_tool_invocation_ms("tc-xyz") == 200


def test_on_function_call_result_records_completion_in_registry():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_function_call_result("call_abc", 1_000_000_000 + 250_000_000)
    assert acc.get_tool_completion_ms("call_abc") == 250


def test_usage_counter_accessors():
    acc = CallAccumulator()
    llm_metric = type("LLMUsageMetricsData", (), {"value": SimpleNamespace(total_tokens=42)})
    tts_metric = type("TTSUsageMetricsData", (), {"value": 99})
    acc.on_metrics_frame(SimpleNamespace(data=[llm_metric(), tts_metric()]))
    assert acc.get_total_llm_tokens() == 42
    assert acc.get_total_tts_characters() == 99


def test_on_turn_started_creates_latency_turn():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 400_000_000)
    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node is None
    assert turn.user_started_ms == 400
    assert acc._active_turn_number == 1


def test_on_bot_started_speaking_sets_bot_started_ms():
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.on_turn_started(1, base_ns + 100_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    assert acc.latency_turns[0].bot_started_ms == 200


def test_on_turn_ended_sets_was_interrupted():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_turn_ended(1, was_interrupted=True)
    assert acc.latency_turns[0].was_interrupted is True
    assert acc._active_turn_number is None


def test_on_latency_breakdown_enriches_existing_turn():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc._pending_pipecat_llm_processing_s = 0.03
    acc._pending_pipecat_tts_processing_s = 0.07
    acc.on_latency_measured(0.2)

    # Turn must already exist (created by on_turn_started)
    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)

    # user_turn_start_time=1.5, user_turn_secs=0.2 → stopped at 1.7s → 700ms
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(duration_secs=0.05)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.user_started_ms == 500
    assert turn.user_stopped_ms == 700
    assert turn.bot_started_ms == 900
    assert turn.ttfb_ms == 50
    assert turn.llm_ms == 30
    assert turn.tts_ms == 70


def test_on_latency_breakdown_keeps_bot_started_unset_when_latency_missing():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 400_000_000)  # user stopped at +400ms
    acc.on_bot_started_speaking(1_000_000_000 + 600_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.2,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(duration_secs=0.04)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)
    assert acc.latency_turns[0].bot_started_ms == 600
    assert acc.latency_turns[0].ttfb_ms == 40


def test_on_latency_breakdown_preserves_user_started_ms_when_user_turn_start_time_missing():
    """Keep on_turn_started timestamp when breakdown start time is missing."""
    acc = CallAccumulator()
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
    assert turn.user_stopped_ms == 0  # unknown (no fallback)
    assert turn.bot_started_ms == 0  # not set for proactive breakdown (is_real_user_turn=False)


def test_on_latency_breakdown_skips_when_no_active_turn(caplog):
    """on_latency_breakdown logs a warning and skips when no turn is active."""
    import logging

    acc = CallAccumulator()
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


def test_on_latency_breakdown_skips_overwrite_for_initial_proactive_greeting():
    """Initial proactive greeting fires breakdown with user_turn_start_time=None.

    is_real_user_turn=False so bot_started_ms must NOT be overwritten by the
    latency queue, and user timing from on_turn_started must be preserved.
    The latency IS consumed from the queue so subsequent real-turn breakdowns
    get the correct value.
    """
    base_ns = 1_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns

    acc.on_latency_measured(1.132)

    # Ghost turn from pipeline internal frame before user speaks.
    acc.on_turn_started(0, base_ns + 44_000_000)
    assert acc.latency_turns[0].user_started_ms == 44

    # Bot greeting starts at +800ms.
    acc.on_bot_started_speaking(base_ns + 800_000_000)
    assert acc.latency_turns[0].bot_started_ms == 800

    # Greeting breakdown fires with user_turn_start_time=None — real production signal.
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=None,
            user_turn_secs=None,
            ttfb=[SimpleNamespace(duration_secs=0.769)],
            function_calls=[],
        )
    )

    turn = acc.latency_turns[0]
    assert turn.is_proactive is True
    # bot_started_ms must NOT be overwritten by latency queue.
    assert turn.bot_started_ms == 800
    # ttfb from greeting breakdown is captured.
    assert turn.ttfb_ms == 769
    # latency was consumed — queue is empty.
    assert len(acc._pending_latency_ms_queue) == 0


def test_on_call_end_marks_done_without_creating_turns():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert not acc.latency_turns
    acc.on_call_end(1_000_000_000 + 500_000_000)
    assert acc.done
    assert acc.call_end_abs_ns == 1_000_000_000 + 500_000_000
    assert not acc.latency_turns  # no synthetic turns created


def test_on_user_started_speaking_records_interrupted_at_ms():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 300_000_000)
    # User cuts in while bot is speaking
    acc.on_user_started_speaking(1_000_000_000 + 450_000_000)
    assert acc.latency_turns[0].interrupted_at_ms == 450


def test_on_turn_started_collapses_when_llm_not_completed():
    # Second user fragment before bot responds — should collapse into same turn
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    acc.on_turn_started(2, 1_000_000_000 + 300_000_000)  # fragment, bot hasn't responded
    assert len(acc.latency_turns) == 1
    assert acc.latency_turns[0].user_started_ms == 200  # keeps earliest


def test_on_turn_started_opens_new_turn_when_llm_completed():
    # llm_completed=True means bot already processed — should NOT collapse
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    acc.latency_turns[0].llm_completed = True
    acc.on_turn_started(2, 1_000_000_000 + 500_000_000)
    assert len(acc.latency_turns) == 2


def test_on_vad_stopped_records_timestamp():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_vad_stopped(1_000_000_000 + 400_000_000)
    assert acc.latency_turns[0].vad_stopped_ns == 1_000_000_000 + 400_000_000


def test_on_user_turn_stopped_computes_stt_ms():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_vad_stopped(1_000_000_000 + 400_000_000)
    acc.on_user_turn_stopped(1_000_000_000 + 550_000_000)  # 150ms after vad
    assert acc.latency_turns[0].stt_ms == 150


def test_on_call_end_anchors_user_stopped_ms_when_still_speaking():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    # user never stopped — call ends while they're mid-sentence
    acc.on_call_end(1_000_000_000 + 600_000_000)
    assert acc.latency_turns[0].user_stopped_ms == 600


def test_on_latency_breakdown_marks_proactive_when_no_user_turn_start():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_bot_started_speaking(1_000_000_000 + 500_000_000)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=None,
            user_turn_secs=None,
            ttfb=[SimpleNamespace(duration_secs=0.1)],
            function_calls=[],
        )
    )
    assert acc.latency_turns[0].is_proactive is True
    assert acc.latency_turns[0].ttfb_ms == 100


def test_on_bot_started_speaking_creates_proactive_turn():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    # No on_turn_started — bot speaks first
    acc.on_bot_started_speaking(1_000_000_000 + 800_000_000)
    assert len(acc.latency_turns) == 1
    assert acc.latency_turns[0].is_proactive is True
    assert acc.latency_turns[0].bot_started_ms == 800


def test_on_turn_started_before_on_start_is_replayed():
    acc = CallAccumulator()
    # turn_started fires BEFORE start (call_start_abs_ns still 0)
    acc.on_turn_started(1, 2_000_000_000 + 300_000_000)
    assert len(acc.latency_turns) == 0  # not yet processed

    acc.on_start(2_000_000_000)
    assert len(acc.latency_turns) == 1
    assert acc.latency_turns[0].user_started_ms == 300


def test_on_metrics_frame_sets_llm_completed_on_current_turn():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    llm_metric = type(
        "ProcessingMetricsData", (), {"processor": "openaillmservice", "value": 0.5}
    )()
    acc.on_metrics_frame(SimpleNamespace(data=[llm_metric]))
    assert acc.latency_turns[0].llm_completed is True


def test_proactive_greeting_detected_when_user_has_not_spoken():
    """Breakdown with user_turn_start_time=None before any user speech → is_proactive=True."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(0, 1_000_000_000 + 44_000_000)  # ghost turn from pipeline
    acc.on_bot_started_speaking(1_000_000_000 + 800_000_000)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=None,
            user_turn_secs=None,
            ttfb=[SimpleNamespace(duration_secs=0.1)],
            function_calls=[],
        )
    )
    assert acc.latency_turns[0].is_proactive is True
    assert acc.latency_turns[0].bot_started_ms == 800
    assert acc.latency_turns[0].ttfb_ms == 100


def test_mid_conversation_breakdown_not_marked_proactive_when_user_has_spoken():
    """Breakdown with user_turn_start_time=None mid-conversation → is_proactive stays False."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(0, 1_000_000_000 + 44_000_000)
    acc.on_user_started_speaking(1_000_000_000 + 2_000_000_000)  # real user speech
    acc.on_bot_started_speaking(1_000_000_000 + 3_000_000_000)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=None,
            user_turn_secs=None,
            ttfb=[SimpleNamespace(duration_secs=0.1)],
            function_calls=[],
        )
    )
    assert acc.latency_turns[0].is_proactive is False


def test_set_disconnection_reason_stores_value():
    acc = CallAccumulator()
    acc.set_disconnection_reason("user_hangup")
    assert acc.disconnection_reason == "user_hangup"


def test_set_disconnection_reason_write_once():
    acc = CallAccumulator()
    acc.set_disconnection_reason("user_hangup")
    acc.set_disconnection_reason("agent_ended")
    assert acc.disconnection_reason == "user_hangup"


def test_set_disconnection_reason_ignores_empty_string():
    acc = CallAccumulator()
    acc.set_disconnection_reason("")
    assert acc.disconnection_reason == ""


def test_disconnection_reason_default_is_empty():
    acc = CallAccumulator()
    assert acc.disconnection_reason == ""

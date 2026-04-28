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


def test_on_metrics_frame_captures_llm_ttfb():
    """TTFBMetricsData from LLM is stored in _pending_llm_ttfb_ms."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    llm_ttfb = type("TTFBMetricsData", (), {"processor": "openaillmservice", "value": 0.05})()
    acc.on_metrics_frame(SimpleNamespace(data=[llm_ttfb]))
    assert acc._pending_llm_ttfb_ms == 50


def test_on_metrics_frame_captures_tts_ttfb():
    """TTFBMetricsData from TTS is stored in _pending_tts_ttfb_ms."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    tts_ttfb = type("TTFBMetricsData", (), {"processor": "cartesiattsservice", "value": 0.03})()
    acc.on_metrics_frame(SimpleNamespace(data=[tts_ttfb]))
    assert acc._pending_tts_ttfb_ms == 30


def test_on_bot_started_speaking_applies_pending_ttfb():
    """Pending TTFB values are written to the turn when bot starts speaking."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)

    llm_ttfb = type("TTFBMetricsData", (), {"processor": "openaillmservice", "value": 0.06})()
    tts_ttfb = type("TTFBMetricsData", (), {"processor": "cartesiattsservice", "value": 0.04})()
    acc.on_metrics_frame(SimpleNamespace(data=[llm_ttfb, tts_ttfb]))

    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)

    turn = acc.latency_turns[0]
    assert turn.llm_ms == 60
    assert turn.ttfb_ms == 40
    # pending fields consumed
    assert acc._pending_llm_ttfb_ms is None
    assert acc._pending_tts_ttfb_ms is None


def test_proactive_turn_ttfb_from_metrics_frame():
    """TTFB from MetricsFrame before bot's first speech is applied to the proactive turn."""
    base_ns = 1_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns

    # Ghost turn from pipeline internal frame before user speaks.
    acc.on_turn_started(0, base_ns + 44_000_000)
    assert acc.latency_turns[0].user_started_ms == 44

    # LLM+TTS TTFB metrics arrive before bot starts speaking
    llm_ttfb = type("TTFBMetricsData", (), {"processor": "openaillmservice", "value": 0.769})()
    tts_ttfb = type("TTFBMetricsData", (), {"processor": "cartesiattsservice", "value": 0.12})()
    acc.on_metrics_frame(SimpleNamespace(data=[llm_ttfb, tts_ttfb]))

    acc.on_bot_started_speaking(base_ns + 800_000_000)

    turn = acc.latency_turns[0]
    assert turn.bot_started_ms == 800
    assert turn.llm_ms == 769
    assert turn.ttfb_ms == 120


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
    assert acc._vad_stopped_ns_by_turn[0] == 1_000_000_000 + 400_000_000


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


def test_on_bot_started_speaking_safety_net_marks_proactive():
    """Bot speaks before any user turn → safety net creates proactive turn."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000

    tts_ttfb = type("TTFBMetricsData", (), {"processor": "cartesiattsservice", "value": 0.1})()
    acc.on_metrics_frame(SimpleNamespace(data=[tts_ttfb]))

    acc.on_bot_started_speaking(1_000_000_000 + 500_000_000)
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
    """Ghost turn before any user speech: bot responds → is_proactive stays False on the turn
    (is_proactive is True only on the safety-net path, not on the ghost-turn path)."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(0, 1_000_000_000 + 44_000_000)  # ghost turn from pipeline

    tts_ttfb = type("TTFBMetricsData", (), {"processor": "cartesiattsservice", "value": 0.1})()
    acc.on_metrics_frame(SimpleNamespace(data=[tts_ttfb]))

    acc.on_bot_started_speaking(1_000_000_000 + 800_000_000)
    assert acc.latency_turns[0].bot_started_ms == 800
    assert acc.latency_turns[0].ttfb_ms == 100


def test_mid_conversation_bot_response_not_marked_proactive():
    """Mid-conversation bot response (after user speech) is not proactive."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(0, 1_000_000_000 + 44_000_000)
    acc.on_user_started_speaking(1_000_000_000 + 2_000_000_000)  # real user speech
    acc.on_bot_started_speaking(1_000_000_000 + 3_000_000_000)
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
    assert acc.disconnection_reason is None


def test_disconnection_reason_default_is_empty():
    acc = CallAccumulator()
    assert acc.disconnection_reason is None


# ---------------------------------------------------------------------------
# Async-task ordering regression tests
#
# TurnTrackingObserver fires on_turn_started as an asyncio task, meaning it
# runs *after* on_user_started_speaking (which fires inline from process_frame).
# These tests assert that timestamps are captured correctly even when
# on_user_started_speaking fires before on_turn_started.
# ---------------------------------------------------------------------------


def test_user_started_speaking_before_on_turn_started_creates_turn():
    """on_user_started_speaking creates the LatencyTurn before on_turn_started fires."""
    base_ns = 1_000_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns

    # Simulate proactive greeting (safety-net path): bot speaks, no prior on_turn_started
    acc.on_bot_started_speaking(base_ns + 500_000_000)
    assert acc.latency_turns[0].is_proactive is True

    acc.on_bot_stopped(base_ns + 1_000_000_000)

    # Now: on_user_started_speaking fires inline (before on_turn_started async task)
    user_start_ns = base_ns + 2_000_000_000
    acc.on_user_started_speaking(user_start_ns)

    assert len(acc.latency_turns) == 2
    assert acc.latency_turns[1].user_started_ms == 2000
    assert acc.latency_turns[1].is_proactive is False

    # Simulate on_turn_started arriving late (asyncio task)
    acc.on_turn_started(2, user_start_ns + 5_000_000)  # 5ms later

    # Must not create a third turn
    assert len(acc.latency_turns) == 2
    # _active_turn_number must be set so stopped-speaking helpers work
    assert acc._active_turn_number == 2
    assert acc._turn_to_latency_idx[2] == 1


def test_user_stopped_and_vad_stopped_work_before_on_turn_started_fires():
    """Stopped-speaking helpers find the turn via _current_user_turn_latency_idx
    even before the on_turn_started async task sets _active_turn_number."""
    base_ns = 1_000_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns

    # Proactive greeting completes
    acc.on_bot_started_speaking(base_ns + 300_000_000)
    acc.on_bot_stopped(base_ns + 800_000_000)

    # User speaks — on_user_started_speaking inline
    user_start_ns = base_ns + 1_500_000_000
    acc.on_user_started_speaking(user_start_ns)

    # on_turn_started hasn't fired yet → _active_turn_number is still None
    assert acc._active_turn_number is None

    # VAD stopped and user stopped fire inline BEFORE on_turn_started
    vad_ns = base_ns + 2_200_000_000
    acc.on_vad_stopped(vad_ns)
    acc.on_user_stopped_speaking(base_ns + 2_300_000_000)
    acc.on_user_turn_stopped(base_ns + 2_400_000_000)

    turn = acc.latency_turns[1]
    assert turn.user_started_ms == 1500
    assert turn.user_stopped_ms == 2300
    assert turn.stt_ms == 200  # 2400 - 2200 ms


def test_full_proactive_plus_user_turn_flow():
    """Full realistic flow: proactive greeting → user turn → bot response."""
    base_ns = 1_000_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns

    # Turn 1: proactive (on_turn_started via StartFrame)
    acc.on_turn_started(1, base_ns + 10_000_000)
    acc.on_bot_started_speaking(base_ns + 500_000_000)
    acc.on_bot_stopped(base_ns + 1_200_000_000)

    assert acc.latency_turns[0].is_proactive is True

    # Turn 2: user speaks — on_user_started_speaking fires first (inline)
    user_start_ns = base_ns + 2_000_000_000
    acc.on_user_started_speaking(user_start_ns)
    assert len(acc.latency_turns) == 2
    assert acc.latency_turns[1].user_started_ms == 2000

    # on_turn_started fires late (async task)
    acc.on_turn_started(2, user_start_ns + 3_000_000)
    assert len(acc.latency_turns) == 2  # no extra turn
    assert acc._active_turn_number == 2

    # User stops, bot responds
    acc.on_vad_stopped(base_ns + 2_700_000_000)
    acc.on_user_stopped_speaking(base_ns + 2_800_000_000)
    llm_ttfb = type("TTFBMetricsData", (), {"processor": "openaillmservice", "value": 0.25})()
    acc.on_metrics_frame(SimpleNamespace(data=[llm_ttfb]))
    acc.on_bot_started_speaking(base_ns + 3_100_000_000)
    acc.on_bot_stopped(base_ns + 4_000_000_000)

    # latency_offset == 1 (one proactive turn)
    assert sum(1 for t in acc.latency_turns if t.is_proactive) == 1

    user_turn = acc.latency_turns[1]
    assert user_turn.user_started_ms == 2000
    assert user_turn.user_stopped_ms == 2800
    assert user_turn.bot_started_ms == 3100
    assert user_turn.bot_stopped_ms == 4000
    assert user_turn.llm_ms == 250

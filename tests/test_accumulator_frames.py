"""Accumulator frame-event tests for speech segments and latency measurements."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tuner_pipecat_sdk.accumulator import CallAccumulator


def _user_segs(acc):
    return [s for s in acc.speech_segments if s.speaker == "user"]


def _bot_segs(acc):
    return [s for s in acc.speech_segments if s.speaker == "bot"]


def test_on_bot_stopped_updates_current_bot_segment():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1
    acc.on_turn_started(1, 1 + 50 * 1_000_000)
    acc.on_bot_started_speaking(1 + 200 * 1_000_000)
    acc.on_bot_stopped(1 + 500 * 1_000_000)
    assert _bot_segs(acc)[-1].stop_ms == 500


def test_on_bot_stopped_no_op_when_done():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1
    acc.on_turn_started(1, 1 + 50 * 1_000_000)
    acc.on_bot_started_speaking(1 + 200 * 1_000_000)
    acc.done = True
    acc.on_bot_stopped(999_000_000)
    assert _bot_segs(acc)[-1].stop_ms is None


def test_on_function_call_in_progress_records_invocation_in_registry():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    frame = MagicMock(function_name="add_topping", arguments={}, tool_call_id="tc-xyz")
    acc.on_function_call_in_progress(frame, 1_000_000_000 + 200_000_000)
    assert acc.get_tool_invocation_ms("tc-xyz") == 200


def test_user_started_speaking_records_per_utterance_windows():
    """Each VAD utterance opens a window; a coalesced turn accrues several windows."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    acc.on_turn_started(1, base + 100_000_000)
    acc.on_user_started_speaking(base + 100_000_000)
    acc.on_user_stopped_speaking(base + 300_000_000)
    # second utterance, same turn (no new turn-start)
    acc.on_user_started_speaking(base + 1_000_000_000)
    acc.on_user_stopped_speaking(base + 1_400_000_000)
    seg = _user_segs(acc)[0]
    assert seg.windows == [[100, 300], [1000, 1400]]
    assert seg.turn_number == 1


def test_tts_text_captured_even_when_it_arrives_before_bot_started():
    """TTSTextFrames arrive ~200ms before BotStartedSpeakingFrame; they must still be
    captured (the old buffer-clear-at-start dropped them)."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.on_turn_started(1, base_ns + 100_000_000)
    acc.on_tts_text("Hello there,", base_ns + 180_000_000)  # before bot start
    acc.on_tts_text("how can I help?", base_ns + 190_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    acc.on_bot_stopped(base_ns + 500_000_000)
    bot = _bot_segs(acc)[0]
    assert bot.spoken_text == "Hello there, how can I help?"


def test_tts_text_assigned_per_segment_by_window():
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.on_turn_started(1, base_ns + 100_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    acc.on_tts_text("First.", base_ns + 210_000_000)
    acc.on_bot_stopped(base_ns + 300_000_000)
    # second response — must not carry "First."
    acc.on_turn_started(2, base_ns + 400_000_000)
    acc.on_bot_started_speaking(base_ns + 500_000_000)
    acc.on_tts_text("Second.", base_ns + 510_000_000)
    acc.on_bot_stopped(base_ns + 600_000_000)
    bots = _bot_segs(acc)
    assert bots[0].spoken_text == "First."
    assert bots[1].spoken_text == "Second."


def test_tts_text_for_never_voiced_response_is_dropped():
    """Text generated for a response interrupted before any BotStartedSpeaking must not
    contaminate the next (actually voiced) segment."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.on_turn_started(1, base_ns + 100_000_000)
    # "draft" TTS, but the user interrupts before bot audio → no BotStartedSpeaking
    acc.on_tts_text("Draft never voiced.", base_ns + 200_000_000)
    # ~7s later the actually-voiced response starts
    acc.on_turn_started(2, base_ns + 7_000_000_000)
    acc.on_tts_text("Real reply.", base_ns + 7_200_000_000)
    acc.on_bot_started_speaking(base_ns + 7_300_000_000)
    acc.on_bot_stopped(base_ns + 8_000_000_000)
    bot = _bot_segs(acc)[0]
    assert bot.spoken_text == "Real reply."  # draft dropped (outside the window)


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


def test_on_turn_started_creates_user_segment():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 400_000_000)
    segs = _user_segs(acc)
    assert len(segs) == 1
    assert segs[0].id == 0
    assert segs[0].speaker == "user"
    assert segs[0].node is None
    assert segs[0].start_ms == 400
    assert acc._active_turn_number == 1


def test_on_bot_started_speaking_sets_start_ms_and_links_measurement():
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.on_turn_started(1, base_ns + 100_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    bot = _bot_segs(acc)[0]
    assert bot.start_ms == 200
    assert len(acc.latency_measurements) == 1
    assert acc.latency_measurements[0].user_segment_id == _user_segs(acc)[0].id
    assert acc.latency_measurements[0].bot_segment_id == bot.id


def test_on_turn_ended_sets_was_interrupted_on_measurement():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 200_000_000)
    acc.on_turn_ended(1, was_interrupted=True)
    assert acc.latency_measurements[0].was_interrupted is True
    assert _bot_segs(acc)[0].interrupted is True
    assert acc._active_turn_number is None


def test_on_latency_breakdown_enriches_measurement():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc._pending_pipecat_llm_processing_s = 0.03
    acc._pending_pipecat_tts_processing_s = 0.07
    acc.on_latency_measured(0.2)

    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(duration_secs=0.05)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    user = _user_segs(acc)[0]
    bot = _bot_segs(acc)[0]
    meas = acc.latency_measurements[0]
    assert user.start_ms == 500
    assert user.stop_ms == 700
    assert bot.start_ms == 900
    assert meas.ttfb_ms == 50
    assert meas.llm_ms == 30
    assert meas.tts_ms == 70
    assert meas.e2e_ms == 200  # 900 - 700


def test_on_latency_breakdown_keeps_bot_start_when_latency_missing():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 400_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 600_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.2,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(duration_secs=0.04)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)
    assert _bot_segs(acc)[0].start_ms == 600
    assert acc.latency_measurements[0].ttfb_ms == 40


def test_on_latency_breakdown_preserves_user_start_when_user_turn_start_time_missing():
    """Keep on_turn_started timestamp when breakdown start time is missing."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 800_000_000)
    acc.on_latency_measured(0.2)

    breakdown = SimpleNamespace(
        user_turn_start_time=None,
        user_turn_secs=None,
        ttfb=[SimpleNamespace(duration_secs=0.05)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    user = _user_segs(acc)[0]
    bot = _bot_segs(acc)[0]
    assert user.start_ms == 500  # preserved from on_turn_started
    assert user.stop_ms is None  # unknown (no fallback)
    assert bot.start_ms == 800  # not overwritten for proactive breakdown


def test_on_latency_breakdown_skips_when_no_pending_measurement(caplog):
    """on_latency_breakdown logs a warning and skips when no measurement is pending."""
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
    assert not acc.latency_measurements


def test_on_latency_breakdown_skips_overwrite_for_initial_proactive_greeting():
    """Initial proactive greeting fires breakdown with user_turn_start_time=None.

    bot start_ms must NOT be overwritten by the latency queue, and the latency IS
    consumed so subsequent real-turn breakdowns get the correct value.
    """
    base_ns = 1_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns

    acc.on_latency_measured(1.132)

    # Ghost turn from pipeline internal frame before user speaks.
    acc.on_turn_started(0, base_ns + 44_000_000)
    assert _user_segs(acc)[0].start_ms == 44

    acc.on_bot_started_speaking(base_ns + 800_000_000)
    assert _bot_segs(acc)[0].start_ms == 800

    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=None,
            user_turn_secs=None,
            ttfb=[SimpleNamespace(duration_secs=0.769)],
            function_calls=[],
        )
    )

    meas = acc.latency_measurements[0]
    assert meas.is_proactive is True
    assert _bot_segs(acc)[0].is_proactive is True
    assert _user_segs(acc)[0].is_proactive is True  # ghost user segment flagged
    assert _bot_segs(acc)[0].start_ms == 800  # not overwritten by latency queue
    assert meas.ttfb_ms == 769
    assert len(acc._pending_latency_ms_queue) == 0  # latency consumed


def test_on_call_end_marks_done_without_creating_segments():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert not acc.speech_segments
    acc.on_call_end(1_000_000_000 + 500_000_000)
    assert acc.done
    assert acc.call_end_abs_ns == 1_000_000_000 + 500_000_000
    assert not acc.speech_segments


def test_on_user_started_speaking_records_interrupted_at_ms():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 300_000_000)
    # User cuts in while bot is speaking
    acc.on_user_started_speaking(1_000_000_000 + 450_000_000)
    assert _bot_segs(acc)[0].interrupted_at_ms == 450


def test_on_turn_started_never_collapses_consecutive_user_speech():
    """Two user utterances before the bot responds become two separate segments.

    The merge-or-not decision moves to the enricher (gated by gap duration), so the
    accumulator never destroys the second utterance's timing.
    """
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    acc.on_turn_started(2, 1_000_000_000 + 300_000_000)
    segs = _user_segs(acc)
    assert len(segs) == 2
    assert segs[0].start_ms == 200
    assert segs[1].start_ms == 300


def test_on_vad_stopped_records_timestamp_by_segment():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_vad_stopped(1_000_000_000 + 400_000_000)
    seg_id = _user_segs(acc)[0].id
    assert acc._vad_stopped_ns_by_user_segment_id[seg_id] == 1_000_000_000 + 400_000_000


def test_on_user_turn_stopped_computes_stt_ms_on_segment():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 100_000_000)
    acc.on_vad_stopped(1_000_000_000 + 400_000_000)
    acc.on_user_turn_stopped(1_000_000_000 + 550_000_000)  # 150ms after vad
    assert _user_segs(acc)[0].stt_ms == 150


def test_on_call_end_anchors_user_stop_when_still_speaking():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_000_000_000 + 200_000_000)
    # user never stopped — call ends while they're mid-sentence
    acc.on_call_end(1_000_000_000 + 600_000_000)
    assert _user_segs(acc)[0].stop_ms == 600


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
    assert acc.latency_measurements[0].is_proactive is True
    assert acc.latency_measurements[0].ttfb_ms == 100


def test_on_bot_started_speaking_creates_proactive_measurement_when_bot_first():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    # No on_turn_started — bot speaks first
    acc.on_bot_started_speaking(1_000_000_000 + 800_000_000)
    assert len(_bot_segs(acc)) == 1
    assert _bot_segs(acc)[0].is_proactive is True
    assert _bot_segs(acc)[0].start_ms == 800
    assert acc.latency_measurements[0].user_segment_id == -1
    assert acc.latency_measurements[0].is_proactive is True


def test_on_turn_started_before_on_start_is_replayed():
    acc = CallAccumulator()
    acc.on_turn_started(1, 2_000_000_000 + 300_000_000)
    assert len(acc.speech_segments) == 0  # not yet processed

    acc.on_start(2_000_000_000)
    assert len(_user_segs(acc)) == 1
    assert _user_segs(acc)[0].start_ms == 300


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
    assert acc.latency_measurements[0].is_proactive is True
    assert _bot_segs(acc)[0].start_ms == 800
    assert acc.latency_measurements[0].ttfb_ms == 100


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
    assert acc.latency_measurements[0].is_proactive is False


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

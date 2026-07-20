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
    frame = SimpleNamespace(function_name="f", tool_call_id="call_abc", result={"ok": True})
    acc.on_function_call_result(frame, 1_000_000_000 + 250_000_000)
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


def test_on_latency_breakdown_prefers_langchain_duration_over_pipecat_metrics():
    """A LangChain/LangGraph-driven turn (via observer.wrap_chain()/wrap_graph())
    reports LLM duration through record_external_llm_duration_ms(), not pipecat's
    MetricsFrame machinery (LangchainProcessor never emits it). That value must win
    even when pipecat-sourced numbers are also present, since those didn't actually
    come from LangchainProcessor's own frames in this scenario."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc._pending_pipecat_llm_processing_s = 0.03  # should be ignored
    acc.record_external_llm_duration_ms(45)
    acc.on_latency_measured(0.2)

    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(processor="LLM", duration_secs=0.658)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    assert acc.latency_measurements[0].llm_ms == 45
    # Consumed and reset -- a second breakdown with nothing pending falls through
    # to the pipecat-sourced fallbacks instead of reusing a stale value.
    assert acc._pending_external_llm_duration_ms is None


def test_on_latency_breakdown_langchain_duration_zero_still_counts_as_measured():
    """0ms is a valid measured duration, not 'nothing reported' -- must not be
    treated the same as None (unlike the truthy check pipecat's own processing-time
    fallback uses)."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.record_external_llm_duration_ms(0)
    acc.on_latency_measured(0.2)

    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(processor="LLM", duration_secs=0.658)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    assert acc.latency_measurements[0].llm_ms == 0


# Used for Google/Gemini, which emit a per-processor TTFB metric but no LLM ProcessingMetricsData.
def test_on_latency_breakdown_llm_ms_falls_back_to_llm_ttfb():
    """Some providers (e.g. Google/Gemini) emit a per-processor TTFB metric but no LLM
    ProcessingMetricsData. LLM latency must still be populated — from the LLM's TTFB —
    instead of coming through blank."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    # No _pending_pipecat_llm_processing_s set (Google never emitted one).
    acc.on_latency_measured(0.2)

    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[
            SimpleNamespace(processor="GoogleLLMService#0", duration_secs=0.658),
            SimpleNamespace(processor="GoogleTTSService#0", duration_secs=0.132),
        ],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    meas = acc.latency_measurements[0]
    assert meas.llm_ms == 658  # fell back to the LLM processor's TTFB
    assert meas.ttfb_ms == 132  # tts_node_ttfb is the TTS processor's TTFB, not the LLM's


def test_on_latency_breakdown_per_node_ttfb_not_contaminated_by_stt():
    """Each node's TTFB is attributed to the right node. Regression guard: the breakdown lists
    TTFBs in arrival order STT -> LLM -> TTS, and "GoogleSTTService".lower() contains the
    substring "tts" — so a naive match would put the STT TTFB into tts_node_ttfb. The STT entry
    must NOT leak into either the LLM or TTS field."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_latency_measured(0.2)

    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[
            SimpleNamespace(processor="GoogleSTTService#0", duration_secs=0.879),  # first!
            SimpleNamespace(processor="GoogleLLMService#0", duration_secs=0.658),
            SimpleNamespace(processor="GoogleTTSService#0", duration_secs=0.132),
        ],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    meas = acc.latency_measurements[0]
    assert meas.ttfb_ms == 132  # TTS node, NOT the STT 879 nor the LLM 658
    assert meas.llm_ms == 658  # LLM node


def test_on_latency_breakdown_llm_processing_time_preferred_over_ttfb():
    """When a provider DOES emit LLM processing time (e.g. OpenAI), it stays the source of
    LLM latency — the TTFB fallback only fills the gap when processing time is absent."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc._pending_pipecat_llm_processing_s = 0.03
    acc.on_latency_measured(0.2)

    acc.on_turn_started(1, 1_000_000_000 + 500_000_000)
    acc.on_user_stopped_speaking(1_000_000_000 + 700_000_000)
    acc.on_bot_started_speaking(1_000_000_000 + 900_000_000)
    breakdown = SimpleNamespace(
        user_turn_start_time=1.5,
        user_turn_secs=0.2,
        ttfb=[SimpleNamespace(processor="OpenAILLMService#0", duration_secs=0.658)],
        function_calls=[],
    )
    acc.on_latency_breakdown(breakdown)

    assert acc.latency_measurements[0].llm_ms == 30  # processing time wins


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

# ---------------------------------------------------------------------------
# set_turn_eou — event-driven EOU decision (on_user_turn_stopped)
# ---------------------------------------------------------------------------


def _user_turns(acc):
    return [t for t in acc.live_turns if t.kind == "user"]


def _commit_user_turn(acc, base, text="hello", start_ms=100):
    """Drive one committed user turn up to (but not including) the stop: start → transcription.
    The caller then adds vad_stop / user_stop / set_turn_eou as the scenario needs."""
    acc.on_turn_started(1, base + start_ms * 1_000_000)
    acc.on_user_started_speaking(base + start_ms * 1_000_000)
    acc.on_transcription(text, base + (start_ms + 100) * 1_000_000)


def test_set_turn_eou_stt_endpoint_records_reason_only():
    """Server-side STT (Flux) closed the turn: eou_reason='stt_endpoint', no ms (the local
    VAD anchor is meaningless on the server's clock, so it's deliberately not computed)."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    acc.on_vad_user_stopped_speaking(base + 1400 * 1_000_000)  # anchor present…
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)      # …but must be ignored
    acc.set_turn_eou("ExternalUserTurnStopStrategy", base + 1600 * 1_000_000)

    turn = _user_turns(acc)[0]
    assert turn.eou_reason == "stt_endpoint"
    assert turn.eou_ms is None


def test_set_turn_eou_silence_timeout_computes_vad_anchored_delay():
    """A local strategy closed the turn: eou_ms = last VAD stop → turn-stop event."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    acc.on_vad_user_stopped_speaking(base + 1400 * 1_000_000)
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)
    acc.set_turn_eou("SpeechTimeoutUserTurnStopStrategy", base + 1500 * 1_000_000)

    turn = _user_turns(acc)[0]
    assert turn.eou_reason == "silence_timeout"
    assert turn.eou_ms == 100  # 1500 − 1400


def test_set_turn_eou_targets_still_open_turn_when_event_beats_commit():
    """If the event arrives BEFORE UserStoppedSpeaking commits, it targets the open turn."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    acc.on_vad_user_stopped_speaking(base + 1400 * 1_000_000)
    # event first (turn still open), then the commit
    acc.set_turn_eou("SpeechTimeoutUserTurnStopStrategy", base + 1450 * 1_000_000)
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)

    turn = _user_turns(acc)[0]
    assert turn.eou_reason == "silence_timeout"
    assert turn.eou_ms == 50  # 1450 − 1400 (event timestamp is the stop anchor)


def test_set_turn_eou_targets_committed_turn_when_event_after_commit():
    """If the event arrives AFTER commit (turn already appended), it back-targets the most
    recent committed user turn — the race the old frame-driven design mislabeled."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    acc.on_vad_user_stopped_speaking(base + 1400 * 1_000_000)
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)  # commits, _open_user_turn → None
    assert acc._open_user_turn is None
    acc.set_turn_eou("ExternalUserTurnStopStrategy", base + 1550 * 1_000_000)

    assert _user_turns(acc)[0].eou_reason == "stt_endpoint"


def test_set_turn_eou_model_verdict_takes_precedence():
    """A smart-turn model verdict set during the turn wins; the later event no-ops."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    metric = type(
        "TurnMetricsData",
        (),
        {"is_complete": True, "e2e_processing_time_ms": 42, "probability": 0.9},
    )()
    acc.on_metrics_frame(SimpleNamespace(data=[metric]))
    acc.on_vad_user_stopped_speaking(base + 1400 * 1_000_000)
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)
    acc.set_turn_eou("SpeechTimeoutUserTurnStopStrategy", base + 1500 * 1_000_000)

    turn = _user_turns(acc)[0]
    assert turn.eou_reason == "model_verdict"  # not silence_timeout
    assert turn.eou_ms == 42
    assert turn.eou_confidence == 0.9


def test_set_turn_eou_none_strategy_without_anchor_leaves_absent():
    """A None/unknown strategy with no usable VAD anchor leaves eou absent (honest)."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)  # no vad_stop recorded
    acc.set_turn_eou("NoneType", base + 1500 * 1_000_000)

    turn = _user_turns(acc)[0]
    assert turn.eou_ms is None
    assert turn.eou_reason is None


def test_set_turn_eou_no_user_turn_is_safe():
    """No user turn present at all → no crash, nothing to target."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    acc.set_turn_eou("SpeechTimeoutUserTurnStopStrategy", base + 1500 * 1_000_000)
    assert _user_turns(acc) == []


def test_set_turn_eou_stale_vad_anchor_does_not_leak_into_next_turn():
    """A VAD anchor left over from a turn that was already decided by a model_verdict (so
    set_turn_eou early-returns for it) must not survive to be misapplied as the next turn's
    silence-timeout anchor when that next turn has no VAD stop of its own yet."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base

    # Turn 1: VAD stop recorded, then a model_verdict decides it before it commits.
    _commit_user_turn(acc, base, text="first turn", start_ms=100)
    metric = type(
        "TurnMetricsData",
        (),
        {"is_complete": True, "e2e_processing_time_ms": 42, "probability": 0.9},
    )()
    acc.on_metrics_frame(SimpleNamespace(data=[metric]))
    acc.on_vad_user_stopped_speaking(base + 1400 * 1_000_000)
    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)
    # Late-arriving event for turn 1: already decided (model_verdict) → early return.
    acc.set_turn_eou("SpeechTimeoutUserTurnStopStrategy", base + 1550 * 1_000_000)

    # Turn 2: commits with NO vad_stop of its own (its VAD frame hasn't reached the
    # observer yet — the exact race set_turn_eou's docstring anticipates).
    _commit_user_turn(acc, base, text="second turn", start_ms=3000)
    acc.on_user_stopped_speaking(base + 3500 * 1_000_000)
    acc.set_turn_eou("SpeechTimeoutUserTurnStopStrategy", base + 3500 * 1_000_000)

    turns = _user_turns(acc)
    assert turns[0].eou_reason == "model_verdict"
    assert turns[0].eou_ms == 42
    # Turn 2 must be left honestly absent, not computed from turn 1's stale VAD anchor.
    assert turns[1].eou_ms is None
    assert turns[1].eou_reason is None


def test_metrics_verdict_does_not_overwrite_already_decided_open_turn():
    """A still-open turn already decided by set_turn_eou (event beat the commit, e.g.
    stt_endpoint — which leaves eou_ms unset) must not be clobbered by a late-arriving
    TurnMetricsData model_verdict for what TurnMetricsData (no turn id) can't tell is a
    different signal for the same turn."""
    acc = CallAccumulator()
    base = 1_000_000_000
    acc.call_start_abs_ns = base
    _commit_user_turn(acc, base)
    # Event beats the commit: the still-open turn is decided as stt_endpoint (eou_ms unset).
    acc.set_turn_eou("ExternalUserTurnStopStrategy", base + 1450 * 1_000_000)

    metric = type(
        "TurnMetricsData",
        (),
        {"is_complete": True, "e2e_processing_time_ms": 42, "probability": 0.9},
    )()
    acc.on_metrics_frame(SimpleNamespace(data=[metric]))

    acc.on_user_stopped_speaking(base + 1500 * 1_000_000)

    turn = _user_turns(acc)[0]
    assert turn.eou_reason == "stt_endpoint"  # not overwritten by model_verdict
    assert turn.eou_ms is None

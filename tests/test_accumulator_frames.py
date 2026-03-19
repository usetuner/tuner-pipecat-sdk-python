"""Accumulator frame-event tests for turns and transitions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn


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
    acc.on_turn_started(1, 1_000_000_000 + 400_000_000)
    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node is None
    assert turn.user_started_ms == 400
    assert acc._active_turn_number == 1



def test_on_bot_started_speaking_sets_bot_started_ms():
    acc = FlowsAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.on_turn_started(1, base_ns + 100_000_000)
    acc.on_bot_started_speaking(base_ns + 200_000_000)
    assert acc.latency_turns[0].bot_started_ms == 200


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


def test_on_latency_breakdown_skips_overwrite_for_initial_proactive_greeting():
    """Initial proactive greeting fires breakdown with user_turn_start_time ≈ call_start.

    computed_started_ms will be 0 (or very close), which must NOT overwrite the
    user timing captured by on_turn_started, and must NOT corrupt bot_started_ms
    by using the initial greeting latency.  The latency IS consumed from the queue
    so the subsequent real-turn breakdown gets the correct value.
    """
    base_ns = 1_000_000_000  # 1.0s unix time
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = base_ns

    # on_latency_measured fires for the initial proactive greeting (1132ms latency)
    acc.on_latency_measured(1.132)

    # User says "Hi." at +1200ms — on_turn_started creates latency_turn[0]
    acc.on_turn_started(0, base_ns + 1_200_000_000)
    assert acc.latency_turns[0].user_started_ms == 1200

    # Bot says "Hi there!" starts at +2400ms (after the greeting was interrupted)
    acc.on_bot_started_speaking(base_ns + 2_400_000_000)
    assert acc.latency_turns[0].bot_started_ms == 2400

    # Breakdown fires for the initial proactive greeting while user "Hi." is active.
    # user_turn_start_time ≈ call_start → computed_started_ms = 0 → skip overwrite.
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=1.0,   # ≈ call start → computed_started_ms = 0
            user_turn_secs=0.0,
            ttfb=[SimpleNamespace(duration_secs=0.769)],
            function_calls=[],
        )
    )

    turn = acc.latency_turns[0]
    # user timing must NOT be overwritten (stays from on_turn_started)
    assert turn.user_started_ms == 1200
    # bot_started must NOT be overwritten by the initial greeting latency
    assert turn.bot_started_ms == 2400
    # ttfb from the initial greeting breakdown IS captured (useful for "Welcome" segment)
    assert turn.ttfb_ms == 769
    # latency was consumed from the queue — verify it's empty now
    assert len(acc._pending_latency_ms_queue) == 0


def test_async_race_vad_frames_arrive_before_on_turn_started():
    """Regression: VAD frames arrive before on_turn_started (async callback fires late).

    Simulates the race where UserStartedSpeakingFrame and UserStoppedSpeakingFrame
    process_frame calls happen before the TurnTrackingObserver async callback fires.
    The pending cache must preserve the timestamps so on_turn_started can apply them.
    """
    base_ns = 1_000_000_000  # t0 = 1.0s

    acc = FlowsAccumulator()
    acc.on_start(base_ns)  # t0

    t1 = base_ns + 500_000_000    # user started speaking at +500ms
    t2 = base_ns + 2_000_000_000  # user stopped speaking at +2000ms
    t3 = base_ns + 34_000_000_000  # bot started speaking at +34000ms
    t4 = base_ns + 34_100_000_000  # on_turn_started fires late at +34100ms

    # VAD frames arrive BEFORE the async on_turn_started callback
    acc.on_user_started_speaking(t1)
    acc.on_user_stopped_speaking(t2)

    # Bot starts speaking (also before on_turn_started in this extreme race)
    acc.on_bot_started_speaking(t3)  # no open turn yet — silently no-ops

    # on_turn_started fires late
    acc.on_turn_started(0, t4)

    # Pending VAD timestamps must have been applied
    assert acc.latency_turns[0].user_stopped_ms == 2000  # from t2, not 0

    # Now latency breakdown arrives; bot_started was captured by on_bot_started_speaking
    # but since that fired before on_turn_started, _open_latency_idx was None and
    # bot_started_ms is still 0 here. The fallback formula runs.
    acc.on_latency_measured(1.164)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=1.5,
            user_turn_secs=1.5,
            ttfb=[SimpleNamespace(duration_secs=0.05)],
            function_calls=[],
        )
    )

    turn = acc.latency_turns[0]
    # user_stopped_ms must be non-zero (from cached pending frame)
    assert turn.user_stopped_ms > 0, "user_stopped_ms must not be 0 (async race bug)"
    # bot_started_ms must not equal 1164 (0 + 1164) since user_stopped_ms is 2000
    assert turn.bot_started_ms != 1164, "bot_started_ms must not be derived from stale 0 user_stopped_ms"


def test_breakdown_does_not_overwrite_bot_started_ms_when_already_set():
    """Regression: on_latency_breakdown must not overwrite bot_started_ms if already set.

    on_bot_started_speaking is authoritative; the breakdown formula is only a fallback.
    """
    base_ns = 1_000_000_000

    acc = FlowsAccumulator()
    acc.on_start(base_ns)
    acc.on_turn_started(0, base_ns + 500_000_000)  # user_started_ms = 500

    # Bot starts at +34000ms — direct from BotStartedSpeakingFrame
    acc.on_bot_started_speaking(base_ns + 34_000_000_000)
    assert acc.latency_turns[0].bot_started_ms == 34_000

    # Breakdown arrives; user_stopped computes to 2000ms, latency = 1164ms
    # Formula would yield 2000 + 1164 = 3164 — must NOT overwrite 34000
    acc.on_latency_measured(1.164)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=1.5,
            user_turn_secs=1.5,
            ttfb=[SimpleNamespace(duration_secs=0.05)],
            function_calls=[],
        )
    )

    assert acc.latency_turns[0].bot_started_ms == 34_000  # preserved from on_bot_started_speaking


def test_pending_vad_cache_not_leaked_into_second_turn():
    """Regression: pending VAD cache must be cleared after normal (non-race) application.

    In the normal flow (on_turn_started fires before VAD frames), the cache is set AND
    immediately applied. If not cleared, the next on_turn_started would re-apply stale
    first-turn timestamps to the second turn, corrupting user_started_ms / user_stopped_ms.
    """
    base_ns = 1_000_000_000

    acc = FlowsAccumulator()
    acc.on_start(base_ns)

    # --- Turn 0: normal order (on_turn_started first, then VAD frames) ---
    acc.on_turn_started(0, base_ns + 1_000_000_000)  # user_started from turn callback = 1000ms
    acc.on_user_started_speaking(base_ns + 900_000_000)   # VAD start = 900ms (earlier)
    acc.on_user_stopped_speaking(base_ns + 2_000_000_000)  # VAD stop = 2000ms

    assert acc.latency_turns[0].user_started_ms == 900   # min(1000, 900)
    assert acc.latency_turns[0].user_stopped_ms == 2000

    # Pending caches must be cleared now
    assert acc._pending_user_started_ns is None
    assert acc._pending_user_stopped_ns is None

    # Complete turn 0 with a bot response so turn 1 creates a new LatencyTurn
    acc.on_bot_started_speaking(base_ns + 3_000_000_000)  # closes _open_latency_idx

    # --- Turn 1: on_turn_started fires with its own fresh timestamp ---
    acc.on_turn_started(1, base_ns + 5_000_000_000)  # user_started = 5000ms

    assert len(acc.latency_turns) == 2
    # Must NOT be contaminated by turn 0's VAD timestamps
    assert acc.latency_turns[1].user_started_ms == 5000
    assert acc.latency_turns[1].user_stopped_ms == 0


def test_on_call_end_marks_done_without_creating_turns():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert not acc.latency_turns
    acc.on_call_end(1_000_000_000 + 500_000_000)
    assert acc.done
    assert acc.call_end_abs_ns == 1_000_000_000 + 500_000_000
    assert not acc.latency_turns  # no synthetic turns created

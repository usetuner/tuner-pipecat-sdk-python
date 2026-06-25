"""Accumulator timing helpers and lifecycle tests."""

import time

from tuner_pipecat_sdk.accumulator import CallAccumulator


def test_rel_ms_zero_when_no_start():
    acc = CallAccumulator()
    assert acc._rel_ms(1_000_000_000) == 0
    assert acc._rel_ms(0) == 0


def test_rel_ms_relative_to_call_start():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert acc._rel_ms(1_500_000_000) == 500
    assert acc._rel_ms(1_000_000_000) == 0


def test_abs_to_rel_ms_returns_zero_when_no_start():
    acc = CallAccumulator()
    assert acc._abs_to_rel_ms(1.5) == 0
    assert acc._abs_to_rel_ms(0) == 0


def test_abs_to_rel_ms_computes_milliseconds():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert acc._abs_to_rel_ms(2.0) == 1000
    assert acc._abs_to_rel_ms(1.5) == 500


def test_on_start_sets_call_start():
    acc = CallAccumulator()
    ns = time.time_ns()
    acc.on_start(ns)
    assert acc.call_start_abs_ns == ns


def test_on_call_end_sets_done_and_end_time():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 100
    acc.on_call_end(200)
    assert acc.done is True
    assert acc.call_end_abs_ns == 200


def test_on_call_end_idempotent_when_done():
    acc = CallAccumulator()
    acc.done = True
    acc.call_end_abs_ns = 100
    acc.on_call_end(999)
    assert acc.call_end_abs_ns == 100


def _user_segs(acc):
    return [s for s in acc.speech_segments if s.speaker == "user"]


def _ttfb_frame(processor, value):
    from types import SimpleNamespace

    data = type("TTFBMetricsData", (), {"processor": processor, "value": value})()
    return SimpleNamespace(data=[data])


def test_stt_ms_set_from_stt_service_ttfb():
    """stt_node_ttfb is the STT service's own TTFB (pure model latency), read from the STT
    TTFBMetricsData on the active user segment."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_001_000_000)

    acc.on_metrics_frame(_ttfb_frame("GoogleSTTService#0", 0.879))

    assert _user_segs(acc)[0].stt_ms == 879


def test_stt_ms_ignores_non_stt_ttfb():
    """LLM/TTS TTFB metrics must not be written to the user segment's stt_ms (note that
    'GoogleSTTService'.lower() contains the substring 'tts' — only genuine STT counts)."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_001_000_000)

    acc.on_metrics_frame(_ttfb_frame("GoogleLLMService#0", 0.658))
    acc.on_metrics_frame(_ttfb_frame("GoogleTTSService#0", 0.132))

    assert _user_segs(acc)[0].stt_ms is None


def test_stt_ms_first_ttfb_of_turn_wins():
    """A coalesced turn emits several STT TTFBs; the first utterance's latency is kept."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_001_000_000)

    acc.on_metrics_frame(_ttfb_frame("GoogleSTTService#0", 0.879))
    acc.on_metrics_frame(_ttfb_frame("GoogleSTTService#0", 1.250))

    assert _user_segs(acc)[0].stt_ms == 879


def test_stt_ms_none_when_no_active_user_segment():
    """An STT TTFB before any turn starts is ignored, not crashed on."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_metrics_frame(_ttfb_frame("GoogleSTTService#0", 0.879))
    assert _user_segs(acc) == []


def test_stt_ms_is_pure_ttfb_not_contaminated_by_turn_timing():
    """Drift guard: stt_node_ttfb must equal the STT service's TTFB ONLY — never the elapsed
    VAD/turn-finalization time or any other frame-timing overhead. This fires a long turn
    (~5s between turn-start and user-stop, which a VAD->turn gap would have measured) plus a
    small 300ms STT TTFB, and pins the result to the metric. If timestamp-gap logic is ever
    reintroduced — in either order relative to the metric — stt_ms becomes ~5000 and this
    fails. (on_metrics_frame only writes when stt_ms is still None, so a gap path running
    first would also be caught: the metric could no longer correct it.)"""
    base = 1_000_000_000
    long_turn_ms = 5_000_000_000  # 5s — clearly distinct from the 300ms TTFB

    # user-stop first, then the metric
    acc = CallAccumulator()
    acc.call_start_abs_ns = base
    acc.on_turn_started(1, base)
    acc.on_user_stopped_speaking(base + long_turn_ms)
    acc.on_metrics_frame(_ttfb_frame("DeepgramSTTService#0", 0.300))
    assert _user_segs(acc)[0].stt_ms == 300  # the metric, NOT ~5000ms of turn elapsed

    # metric first, then user-stop (order independence)
    acc2 = CallAccumulator()
    acc2.call_start_abs_ns = base
    acc2.on_turn_started(1, base)
    acc2.on_metrics_frame(_ttfb_frame("DeepgramSTTService#0", 0.300))
    acc2.on_user_stopped_speaking(base + long_turn_ms)
    assert _user_segs(acc2)[0].stt_ms == 300

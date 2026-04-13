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


def test_on_user_turn_stopped_no_vad_stop_is_safe():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000

    # Set up a turn without ever firing on_vad_stopped
    acc.on_turn_started(turn_number=1, timestamp_ns=1_001_000_000)
    acc._active_turn_number = 1

    # Should log a warning and return — not crash
    acc.on_user_turn_stopped(timestamp_ns=1_002_000_000)

    turn = acc.latency_turns[0]
    assert turn.stt_ms is None  # or 0, whatever your default is


def test_stt_ms_computed_from_vad_gap():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000_000
    acc.on_turn_started(1, 1_000_000_000_000 + 1_000_000_000)

    vad_ts = 1_000_000_000_000 + 1_500_000_000
    user_stopped_ts = 1_000_000_000_000 + 1_800_000_000  # 300ms after VAD

    acc.on_vad_stopped(vad_ts)
    acc.on_user_turn_stopped(user_stopped_ts)

    assert acc.latency_turns[0].stt_ms == 300


def test_vad_stopped_before_user_stopped_speaking_sets_stt():
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000_000
    acc.on_turn_started(1, 1_000_000_000_000 + 1_000_000_000)

    acc.on_vad_stopped(1_000_000_000_000 + 1_400_000_000)
    acc.on_user_stopped_speaking(1_000_000_000_000 + 1_600_000_000)
    acc.on_user_turn_stopped(1_000_000_000_000 + 1_700_000_000)  # 300ms after VAD

    assert acc.latency_turns[0].stt_ms == 300
    assert acc.latency_turns[0].user_stopped_ms > 0


def test_on_user_turn_stopped_before_vad_stopped_warns_and_is_safe():
    """on_user_turn_stopped with no prior VAD stop logs warning and skips."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.on_turn_started(1, 1_001_000_000)

    # Fire user turn stopped without vad stopped — should not crash
    acc.on_user_turn_stopped(1_001_500_000)

    assert acc.latency_turns[0].stt_ms is None

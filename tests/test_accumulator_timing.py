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

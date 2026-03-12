"""Accumulator timing helpers and lifecycle tests."""

import time

from pipecat_flows_tuner.accumulator import FlowsAccumulator


def test_rel_ms_zero_when_no_start():
    acc = FlowsAccumulator()
    assert acc._rel_ms(1_000_000_000) == 0
    assert acc._rel_ms(0) == 0


def test_rel_ms_relative_to_call_start():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    assert acc._rel_ms(1_500_000_000) == 500
    assert acc._rel_ms(1_000_000_000) == 0


def test_ns_to_ms_returns_none_when_zero():
    acc = FlowsAccumulator()
    assert acc._ns_to_ms(0, 1_000_000) is None
    assert acc._ns_to_ms(1_000_000, 0) is None


def test_ns_to_ms_computes_milliseconds():
    acc = FlowsAccumulator()
    assert acc._ns_to_ms(0, 1_000_000) is None
    assert acc._ns_to_ms(1_000_000_000, 2_000_000_000) == 1_000
    assert acc._ns_to_ms(1_000_000_000, 1_500_000_000) == 500


def test_on_start_sets_call_start():
    acc = FlowsAccumulator()
    ns = time.time_ns()
    acc.on_start(ns)
    assert acc.call_start_abs_ns == ns


def test_on_call_end_sets_done_and_end_time():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 100
    acc.on_call_end(200)
    assert acc.done is True
    assert acc.call_end_abs_ns == 200


def test_on_call_end_idempotent_when_done():
    acc = FlowsAccumulator()
    acc.done = True
    acc.call_end_abs_ns = 100
    acc.on_call_end(999)
    assert acc.call_end_abs_ns == 100

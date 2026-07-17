"""Tests for the LangChain/LangGraph observability bridge.

Requires the ``langchain`` extra (``tuner-langchain`` + ``langchain-core``) to be
installed; skipped entirely otherwise, matching the SDK's "optional" contract.
"""

import pytest

pytest.importorskip("tuner_langchain")

from tuner_langchain import TunerAccumulator
from tuner_langchain.segment_builder import segments_from_invocation

from tuner_pipecat_sdk.accumulator import CallAccumulator
from tuner_pipecat_sdk.langchain_bridge import (
    _AccumulatorUsageSink,
    _segment_dict_to_transcript_segment,
    merge_langchain_segments,
)
from tuner_pipecat_sdk.models import TranscriptSegment


def _acc(call_start_ns: int = 1_000_000_000) -> CallAccumulator:
    acc = CallAccumulator()
    acc.call_start_abs_ns = call_start_ns
    return acc


def _native_segment(start_ms: int, role: str = "user") -> TranscriptSegment:
    return TranscriptSegment(role=role, text=f"native @ {start_ms}", start_ms=start_ms, metadata={})


def _lg_accumulator_with_node_and_tool(call_start_ns: int) -> TunerAccumulator:
    """Build a TunerAccumulator with one node containing one tool call, driven
    directly through its on_* event methods (no real LangChain run needed)."""
    lg_acc = TunerAccumulator()
    root_id, node_id, tool_id = "root-1", "node-1", "tool-1"
    lg_acc.on_graph_start(root_id, call_start_ns + 100_000_000)  # +100ms
    lg_acc.on_node_start(
        "booking_node", node_id, root_id, call_start_ns + 150_000_000, {}
    )  # +150ms
    lg_acc.on_tool_start(
        "check_availability",
        tool_id,
        node_id,
        call_start_ns + 200_000_000,
        '{"date": "2024-06-15"}',
    )  # +200ms
    lg_acc.on_tool_end(tool_id, call_start_ns + 250_000_000, '{"available": true}')  # +250ms
    lg_acc.on_node_end(node_id, call_start_ns + 300_000_000, {})  # +300ms
    lg_acc.on_graph_end(root_id, call_start_ns + 350_000_000)  # +350ms
    return lg_acc


# ── merge_langchain_segments ────────────────────────────────────────────────


def test_merge_with_none_accumulator_returns_native_list_unchanged():
    native = [_native_segment(0), _native_segment(100)]
    merged = merge_langchain_segments(native, None, 1_000_000_000)
    assert merged == native


def test_merge_interleaves_node_and_tool_segments_by_start_ms():
    call_start_ns = 1_000_000_000
    lg_acc = _lg_accumulator_with_node_and_tool(call_start_ns)
    native = [_native_segment(0, "user"), _native_segment(400, "agent")]

    merged = merge_langchain_segments(native, lg_acc, call_start_ns)

    roles_in_order = [(s.role, s.start_ms) for s in merged]
    assert roles_in_order == [
        ("user", 0),
        ("node_transition", 150),
        ("agent_function", 200),
        ("agent_result", 250),
        ("agent", 400),
    ]


def test_merge_node_transition_populates_node_info():
    call_start_ns = 1_000_000_000
    lg_acc = _lg_accumulator_with_node_and_tool(call_start_ns)

    merged = merge_langchain_segments([], lg_acc, call_start_ns)

    node_seg = next(s for s in merged if s.role == "node_transition")
    assert node_seg.node is not None
    assert node_seg.node.to == "booking_node"
    assert node_seg.node.from_ is None  # root-level node, no parent node name


def test_merge_tool_segments_populate_tool_info():
    call_start_ns = 1_000_000_000
    lg_acc = _lg_accumulator_with_node_and_tool(call_start_ns)

    merged = merge_langchain_segments([], lg_acc, call_start_ns)

    call_seg = next(s for s in merged if s.role == "agent_function")
    assert call_seg.tool is not None
    assert call_seg.tool.name == "check_availability"
    assert call_seg.tool.params == {"date": "2024-06-15"}

    result_seg = next(s for s in merged if s.role == "agent_result")
    assert result_seg.tool is not None
    assert result_seg.tool.name == "check_availability"
    assert result_seg.tool.result == {"value": {"available": True}}
    # ToolInfo has no duration_ms field; it's lifted to the segment's own field instead.
    assert result_seg.duration_ms == 50  # 250ms end - 200ms start
    assert result_seg.tool.is_error is False


def test_merge_without_tuner_langchain_installed_logs_and_returns_native(monkeypatch):
    import tuner_pipecat_sdk.langchain_bridge as bridge

    monkeypatch.setattr(bridge, "_LANGCHAIN_AVAILABLE", False)
    native = [_native_segment(0)]
    merged = merge_langchain_segments(native, object(), 1_000_000_000)
    assert merged == native


def test_merge_degrades_to_native_only_when_lg_accumulator_raises():
    """A broken/incompatible lg_accumulator (e.g. a future tuner-langchain shape
    change) must not cost the caller the whole call's payload -- merge must
    swallow the failure and fall back to the native transcript."""

    class _BrokenAccumulator:
        def get_invocations(self):
            raise RuntimeError("simulated tuner-langchain shape change")

    native = [_native_segment(0), _native_segment(100)]

    merged = merge_langchain_segments(native, _BrokenAccumulator(), 1_000_000_000)

    assert merged == native


def test_merge_degrades_to_native_only_when_segment_dict_is_malformed():
    """Same guarantee, but the failure happens deeper -- inside conversion of an
    individual segment dict rather than at get_invocations() itself."""

    class _FakeInvocation:
        pass

    class _AccumulatorWithBadSegment:
        def get_invocations(self):
            return [_FakeInvocation()]

    native = [_native_segment(0)]

    # segments_from_invocation() will choke on a fake invocation object missing
    # the attributes it expects (e.g. .nodes/.tools) -- exactly the kind of
    # "unknown future shape" failure this guard exists for.
    merged = merge_langchain_segments(native, _AccumulatorWithBadSegment(), 1_000_000_000)

    assert merged == native


def test_merge_skips_only_the_malformed_invocation_not_the_whole_call():
    """A single malformed invocation must not cost the caller every other
    invocation's segments for the same call -- only the bad one is skipped."""
    call_start_ns = 1_000_000_000
    good_lg_acc = _lg_accumulator_with_node_and_tool(call_start_ns)
    good_invocation = good_lg_acc.get_invocations()[0]

    class _FakeBadInvocation:
        pass

    class _MixedAccumulator:
        def get_invocations(self):
            return [good_invocation, _FakeBadInvocation()]

    native = [_native_segment(0)]

    merged = merge_langchain_segments(native, _MixedAccumulator(), call_start_ns)

    # The good invocation's segments still made it in, even though the second
    # invocation in the same call was malformed.
    roles = {s.role for s in merged}
    assert {"node_transition", "agent_function", "agent_result"} <= roles


# ── _segment_dict_to_transcript_segment ─────────────────────────────────────


def test_segment_dict_to_transcript_segment_handles_missing_optional_fields():
    seg = _segment_dict_to_transcript_segment({"role": "node_transition", "start_ms": 10})
    assert seg.role == "node_transition"
    assert seg.start_ms == 10
    assert seg.text is None
    assert seg.node is None
    assert seg.tool is None
    assert seg.metadata == {}


# ── compat test: guard against tuner-langchain segment shape drift ─────────


def test_segments_from_invocation_shape_matches_expectations():
    """Pins the exact dict shape tuner-langchain's segments_from_invocation()
    returns. If a future tuner-langchain release changes this shape, this test
    fails loudly here instead of silently breaking merge_langchain_segments()
    at runtime in production."""
    call_start_ns = 1_000_000_000
    lg_acc = _lg_accumulator_with_node_and_tool(call_start_ns)
    invocation = lg_acc.get_invocations()[0]

    segs = segments_from_invocation(invocation, call_start_ns)
    by_role = {s["role"]: s for s in segs}

    assert set(by_role) == {"node_transition", "agent_function", "agent_result"}
    for role in ("role", "text", "start_ms", "end_ms", "metadata"):
        assert role in by_role["node_transition"]
    assert "node" in by_role["node_transition"]
    assert set(by_role["node_transition"]["node"]) == {"to", "from", "reason"}

    for role_name in ("agent_function", "agent_result"):
        seg = by_role[role_name]
        assert "tool" in seg
        assert "name" in seg["tool"]
        assert "request_id" in seg["tool"]


# ── _AccumulatorUsageSink ────────────────────────────────────────────────────


def test_usage_sink_records_llm_usage_into_accumulator():
    acc = _acc()
    sink = _AccumulatorUsageSink(acc)

    sink._record_llm_usage(120, 45)

    assert acc.get_llm_prompt_tokens() == 120
    assert acc.get_llm_completion_tokens() == 45
    assert acc.get_total_llm_tokens() == 165


def test_usage_sink_start_ns_proxies_accumulator_call_start():
    acc = _acc(call_start_ns=42)
    sink = _AccumulatorUsageSink(acc)
    assert sink._start_ns == 42


def test_usage_sink_last_llm_duration_ms_forwards_into_accumulator():
    acc = _acc()
    sink = _AccumulatorUsageSink(acc)

    sink._last_llm_duration_ms = 250  # tuner-langchain assigns this directly

    assert sink._last_llm_duration_ms == 250  # readback still works
    assert acc._pending_external_llm_duration_ms == 250


def test_usage_sink_last_llm_duration_ms_overwrites_on_repeated_calls():
    """A turn with multiple LLM calls (tool-calling round-trip) overwrites, not
    accumulates -- only the last call's duration survives, matching
    tuner-langchain's own on_llm_end semantics."""
    acc = _acc()
    sink = _AccumulatorUsageSink(acc)

    sink._last_llm_duration_ms = 100
    sink._last_llm_duration_ms = 250

    assert acc._pending_external_llm_duration_ms == 250


def test_usage_sink_attach_degrades_gracefully_when_hook_missing():
    """If tuner-langchain's private _attach_session shape has drifted, attach()
    must never raise -- usage capture is skipped, everything else keeps working."""
    sink = _AccumulatorUsageSink(_acc())

    class _NoAttachSession:
        pass

    sink.attach(_NoAttachSession())  # must not raise


def test_usage_sink_attach_degrades_gracefully_when_hook_raises_unexpected_exception():
    """attach()'s docstring promises "Never raises" unconditionally, not just for
    a missing/reshaped hook (AttributeError/TypeError) -- any other exception from
    the undocumented private hook must also be swallowed."""
    sink = _AccumulatorUsageSink(_acc())

    class _BrokenAttachSession:
        def _attach_session(self, session):
            raise ValueError("simulated tuner-langchain internal validation error")

    sink.attach(_BrokenAttachSession())  # must not raise


def test_usage_sink_records_llm_usage_treats_none_as_zero():
    """A provider/turn that doesn't report usage must not crash on_llm_end --
    None is a valid, expected input, not a malformed one."""
    acc = _acc()
    sink = _AccumulatorUsageSink(acc)

    sink._record_llm_usage(None, None)

    assert acc.get_llm_prompt_tokens() == 0
    assert acc.get_llm_completion_tokens() == 0


def test_usage_sink_last_llm_duration_ms_clamps_negative_to_zero():
    acc = _acc()
    sink = _AccumulatorUsageSink(acc)

    sink._last_llm_duration_ms = -50

    assert acc._pending_external_llm_duration_ms == 0


def test_usage_sink_attach_succeeds_against_real_wrapped_runnable():
    from tuner_langchain.graph_wrapper import wrap_chain

    acc = _acc()
    sink = _AccumulatorUsageSink(acc)
    wrapped = wrap_chain(object())

    sink.attach(wrapped)

    assert wrapped._handler._session is sink
    assert wrapped._handler._session_start_ns == acc.call_start_abs_ns


# ── regression test pinning the "usage tokens can't double-count today" claim ─


def test_pipecat_and_langchain_usage_sum_without_overlap_today():
    """Pins the current, safe assumption: pipecat-sourced and langchain-sourced
    LLM usage counters are summed unconditionally in CallAccumulator because
    pipecat's LangchainProcessor never emits MetricsFrame/LLMUsageMetricsData
    itself (verified by reading pipecat/processors/frameworks/langchain.py).

    If a future pipecat processor driving a LangChain runnable starts emitting
    its own usage MetricsFrame for the same LLM call, this summing would
    double-count cost. This test exists so that assumption is visible and
    revisited deliberately, not discovered as a silent billing bug.
    """
    acc = _acc()

    # Simulate a native pipecat MetricsFrame usage event (LLMUsageMetricsData).
    token_usage = type(
        "LLMTokenUsage", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )()
    usage_data = type("LLMUsageMetricsData", (), {"value": token_usage})()
    acc.on_metrics_frame(type("Frame", (), {"data": [usage_data]})())

    # Simulate a langchain-sourced usage event for a *different* LLM call.
    acc.record_external_llm_usage(20, 8)

    assert acc.get_llm_prompt_tokens() == 30
    assert acc.get_llm_completion_tokens() == 13
    assert acc.get_total_llm_tokens() == 15 + 20 + 8

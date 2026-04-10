"""Tests for _BaseObserver: shared frame routing and accumulator wiring."""

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from pipecat.frames.frames import (
    FunctionCallResultFrame,
    MetricsFrame,
    StartFrame,
)

from tuner_pipecat_sdk.observer import Observer

# ---------------------------------------------------------------------------
# Use Observer as the concrete vehicle for testing base behaviour.
# All assertions here apply equally to FlowsObserver.
# ---------------------------------------------------------------------------


@pytest.fixture
def observer():
    return Observer(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
        debug=False,
    )


def test_observer_init():
    o = Observer(api_key="k", workspace_id=2, agent_id="a", call_id="c")
    assert o._config.workspace_id == 2
    assert o._config.base_url == "http://localhost:8000"
    assert o._acc is not None
    assert o._context_provider is None
    assert o._flushed is False


def test_handle_start_frame_updates_accumulator(observer):
    frame = StartFrame()
    observer._handle(frame, 1_000_000_000)
    # call_start_abs_ns is pre-set to time.time_ns() in __init__; just verify non-zero.
    assert observer._acc.call_start_abs_ns > 0


def test_handle_start_frame_warns_when_metrics_disabled(observer):
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        observer._handle(
            StartFrame(enable_metrics=False, enable_usage_metrics=False), 1_000_000_000
        )
    finally:
        logger.remove(sink_id)
    combined = "".join(messages)
    assert "enable_metrics=False" in combined
    assert "enable_usage_metrics=False" in combined


def test_handle_start_frame_no_warning_when_metrics_enabled(observer):
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
    try:
        observer._handle(StartFrame(enable_metrics=True, enable_usage_metrics=True), 1_000_000_000)
    finally:
        logger.remove(sink_id)
    combined = "".join(messages)
    assert "enable_metrics=False" not in combined
    assert "enable_usage_metrics=False" not in combined


def test_handle_metrics_frame_routes_to_accumulator(observer):
    frame = MetricsFrame(data=[])
    with patch.object(observer._acc, "on_metrics_frame") as mock_on_metrics:
        observer._handle(frame, 500)
        mock_on_metrics.assert_called_once_with(frame)


def test_handle_function_call_result_records_completion(observer):
    observer._acc.call_start_abs_ns = 1_000_000_000
    frame = FunctionCallResultFrame(
        tool_call_id="tc-1",
        function_name="foo",
        arguments="{}",
        result="ok",
    )
    observer._handle(frame, 1_000_000_000 + 300_000_000)
    assert observer._acc.get_tool_completion_ms("tc-1") == 300


def test_observer_exposes_latency_observer(observer):
    assert observer.latency_observer is not None


@pytest.mark.asyncio
async def test_attach_turn_tracking_observer_wiring(observer):
    handlers: dict[str, Any] = {}

    class FakeTurnTracker:
        def event_handler(self, event_name: str):
            def decorator(func):
                handlers[event_name] = func
                return func

            return decorator

    observer.attach_turn_tracking_observer(FakeTurnTracker())

    assert "on_turn_started" in handlers
    assert "on_turn_ended" in handlers

    observer._acc.call_start_abs_ns = 1_000_000_000
    with patch("tuner_pipecat_sdk._base.time") as mock_time:
        mock_time.time_ns.return_value = 1_000_000_000 + 300_000_000
        await handlers["on_turn_started"](None, 1)

    assert len(observer._acc.latency_turns) == 1
    assert observer._acc.latency_turns[0].user_started_ms == 300
    assert observer._acc._active_turn_number == 1

    await handlers["on_turn_ended"](None, 1, 2.5, True)
    assert observer._acc.latency_turns[0].was_interrupted is True
    assert observer._acc._active_turn_number is None


def test_cancel_frame_with_resolver_sets_reason(observer):
    from pipecat.frames.frames import CancelFrame

    observer._disconnection_reason_resolver = lambda: "user_hangup"
    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(CancelFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason == "user_hangup"


def test_cancel_frame_without_resolver_leaves_reason_empty(observer):
    from pipecat.frames.frames import CancelFrame

    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(CancelFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason == ""


def test_end_frame_with_resolver_sets_reason(observer):
    from pipecat.frames.frames import EndFrame

    observer._disconnection_reason_resolver = lambda: "agent_ended"
    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(EndFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason == "agent_ended"


def test_end_frame_without_resolver_leaves_reason_empty(observer):
    from pipecat.frames.frames import EndFrame

    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(EndFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason == ""


def test_resolver_raising_exception_does_not_crash(observer):
    from pipecat.frames.frames import CancelFrame

    def bad_resolver():
        raise RuntimeError("oops")

    observer._disconnection_reason_resolver = bad_resolver
    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(CancelFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason == ""


def test_resolver_returning_none_leaves_reason_empty(observer):
    from pipecat.frames.frames import CancelFrame

    observer._disconnection_reason_resolver = lambda: None
    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(CancelFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason == ""

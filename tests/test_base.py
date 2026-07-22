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
    assert o._config.base_url == "https://api.usetuner.ai"
    assert o._acc is not None
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


def test_handle_tts_text_frame_routes_to_accumulator(observer):
    from pipecat.frames.frames import TTSTextFrame

    frame = TTSTextFrame(text="hello there", aggregated_by="sentence")
    with patch.object(observer._acc, "on_tts_text") as mock_on_tts:
        observer._handle(frame, 500)
        mock_on_tts.assert_called_once_with("hello there", 500)


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


def _pushed(frame, direction):
    """Build a FramePushed event for the pipeline-level on_push_frame hook."""
    from unittest.mock import Mock

    from pipecat.observers.base_observer import FramePushed

    return FramePushed(
        source=Mock(), destination=Mock(), frame=frame, direction=direction, timestamp=0
    )


@pytest.mark.asyncio
async def test_on_push_frame_routes_transcription_downstream(observer):
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    observer._acc.call_start_abs_ns = 1_000_000_000
    frame = TranscriptionFrame(text="i want pizza", user_id="u1", timestamp="2026-01-01T00:00:00Z")
    await observer.on_push_frame(_pushed(frame, FrameDirection.DOWNSTREAM))
    assert [t for t, _ in observer._acc.user_transcriptions] == ["i want pizza"]


@pytest.mark.asyncio
async def test_on_push_frame_ignores_upstream(observer):
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    frame = TranscriptionFrame(text="ignored", user_id="u1", timestamp="2026-01-01T00:00:00Z")
    await observer.on_push_frame(_pushed(frame, FrameDirection.UPSTREAM))
    assert observer._acc.user_transcriptions == []


@pytest.mark.asyncio
async def test_on_push_frame_dedups_by_frame_id(observer):
    """A pipeline-level observer sees the same frame once per processor hop — it must be
    handled exactly once."""
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    observer._acc.call_start_abs_ns = 1_000_000_000
    frame = TranscriptionFrame(text="only once", user_id="u1", timestamp="2026-01-01T00:00:00Z")
    for _ in range(5):  # simulate the frame traversing five processor boundaries
        await observer.on_push_frame(_pushed(frame, FrameDirection.DOWNSTREAM))
    assert len(observer._acc.user_transcriptions) == 1
    assert observer._acc.user_transcriptions[0][0] == "only once"


@pytest.mark.asyncio
async def test_on_push_frame_interim_transcription_not_recorded(observer):
    from pipecat.frames.frames import InterimTranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection

    frame = InterimTranscriptionFrame(text="par", user_id="u1", timestamp="2026-01-01T00:00:00Z")
    await observer.on_push_frame(_pushed(frame, FrameDirection.DOWNSTREAM))
    assert observer._acc.user_transcriptions == []


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

    user_segs = [s for s in observer._acc.speech_segments if s.speaker == "user"]
    assert len(user_segs) == 1
    assert user_segs[0].start_ms == 300
    assert observer._acc._active_turn_number == 1

    # Bot responds, then the turn ends as interrupted.
    observer._acc.on_bot_started_speaking(1_000_000_000 + 500_000_000)
    await handlers["on_turn_ended"](None, 1, 2.5, True)
    assert observer._acc.latency_measurements[0].was_interrupted is True
    assert observer._acc._active_turn_number is None

@pytest.mark.asyncio
async def test_attach_user_aggregator_wiring(observer):
    """attach_user_aggregator subscribes to the aggregator's on_user_turn_stopped event and,
    when it fires, drives set_turn_eou with the strategy class name — the single signal that
    decides EOU (no dependency on UserStoppedSpeaking frame ordering)."""
    handlers: dict[str, Any] = {}

    class FakeUserAggregator:
        def event_handler(self, event_name: str):
            def decorator(func):
                handlers[event_name] = func
                return func

            return decorator

    observer.attach_user_aggregator(FakeUserAggregator())
    assert "on_user_turn_stopped" in handlers

    # Fire the event with a fake strategy object; the handler passes its class name through.
    class ExternalUserTurnStopStrategy:  # name is what set_turn_eou keys on
        pass

    with patch.object(observer._acc, "set_turn_eou") as mock_set_eou:
        with patch("tuner_pipecat_sdk._base.time") as mock_time:
            mock_time.time_ns.return_value = 123_456_789
            await handlers["on_user_turn_stopped"](None, ExternalUserTurnStopStrategy(), None)
        mock_set_eou.assert_called_once_with("ExternalUserTurnStopStrategy", 123_456_789)

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
    assert observer._acc.disconnection_reason is None


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
    assert observer._acc.disconnection_reason is None


def test_resolver_raising_exception_does_not_crash(observer):
    from pipecat.frames.frames import CancelFrame

    def bad_resolver():
        raise RuntimeError("oops")

    observer._disconnection_reason_resolver = bad_resolver
    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(CancelFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason is None


def test_resolver_returning_none_leaves_reason_empty(observer):
    from pipecat.frames.frames import CancelFrame

    observer._disconnection_reason_resolver = lambda: None
    with patch("tuner_pipecat_sdk._base.asyncio.create_task"):
        observer._handle(CancelFrame(), 1_000_000_000)
    assert observer._acc.disconnection_reason is None

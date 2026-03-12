"""Tests for FlowsObserver: frame routing, attach_flow_manager, flush."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from pipecat.frames.frames import (
    EndFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from pipecat_flows_tuner.observer import FlowsObserver


@pytest.fixture
def observer():
    return FlowsObserver(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
        debug=False,
    )


def test_observer_init():
    o = FlowsObserver(
        api_key="k",
        workspace_id=2,
        agent_id="a",
        call_id="c",
    )
    assert o._config.workspace_id == 2
    assert o._config.base_url == "http://localhost:8000"
    assert o._acc is not None
    assert o._flow_manager is None
    assert o._flushed is False


@pytest.mark.asyncio
async def test_attach_flow_manager_patches_set_node(observer, mock_flow_manager):
    original_set_node = AsyncMock()
    mock_flow_manager._set_node = original_set_node
    mock_flow_manager._current_node = "greeting"
    mock_flow_manager.state = {"key": "value"}

    observer.attach_flow_manager(mock_flow_manager)

    assert mock_flow_manager._set_node is not original_set_node
    # Call the patched _set_node
    await mock_flow_manager._set_node("transfer", {"functions": [], "task_messages": []})
    assert original_set_node.called
    assert len(observer._acc.node_transitions) == 1
    assert observer._acc.node_transitions[0].from_node == "greeting"
    assert observer._acc.node_transitions[0].to_node == "transfer"


def test_handle_start_frame_updates_accumulator(observer):
    """_handle(StartFrame) updates accumulator; process_frame requires pipeline setup."""
    frame = StartFrame()
    ts = 1_000_000_000
    observer._handle(frame, ts)
    assert observer._acc.call_start_abs_ns == ts


@pytest.mark.asyncio
async def test_handle_vad_user_started(observer):
    observer._acc.call_start_abs_ns = 0
    observer._handle(VADUserStartedSpeakingFrame(), 100)
    assert observer._acc._user_started_ns == 100


@pytest.mark.asyncio
async def test_handle_vad_user_stopped(observer):
    frame = MagicMock(spec=VADUserStoppedSpeakingFrame)
    frame.stop_secs = 0
    observer._acc._current_node = "n1"
    observer._handle(frame, 200)
    assert observer._acc._user_stopped_ns == 200
    assert observer._acc._llm_started_ns == 0


@pytest.mark.asyncio
async def test_handle_end_frame_triggers_flush(observer, mock_flow_manager):
    observer.attach_flow_manager(mock_flow_manager)
    mock_flow_manager.get_current_context.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    observer._acc.call_start_abs_ns = 0
    observer._acc.call_end_abs_ns = 1_000_000_000
    observer._acc.done = True
    observer._acc.latency_turns = []

    with patch("pipecat_flows_tuner.observer.post_call", new_callable=AsyncMock) as post_mock:
        observer._handle(EndFrame(), 1_000_000_000)
        # _flush is scheduled with create_task; allow it to run
        await asyncio.sleep(0.05)
        post_mock.assert_called_once()
        payload = post_mock.call_args[0][1]
        assert payload.call_id == "call-1"


@pytest.mark.asyncio
async def test_flush_without_flow_manager_does_not_post(observer):
    assert observer._flow_manager is None
    with patch("pipecat_flows_tuner.observer.post_call", new_callable=AsyncMock) as post_mock:
        await observer._flush()
        post_mock.assert_not_called()


@pytest.mark.asyncio
async def test_flush_with_flow_manager_builds_and_posts(observer, mock_flow_manager):
    observer.attach_flow_manager(mock_flow_manager)
    mock_flow_manager.get_current_context.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Bye"},
    ]
    observer._acc.on_start(0)
    observer._acc.on_call_end(1_000_000_000)

    with patch("pipecat_flows_tuner.observer.post_call", new_callable=AsyncMock) as post_mock:
        await observer._flush()
        post_mock.assert_called_once()
        config, payload = post_mock.call_args[0]
        assert config.call_id == "call-1"
        assert payload.call_id == "call-1"
        assert payload.call_status == "call_ended"

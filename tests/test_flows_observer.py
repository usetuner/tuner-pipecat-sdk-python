"""Tests for FlowsObserver: pipecat-flows pipeline."""

import asyncio
import sys
from unittest.mock import AsyncMock, patch

import pytest

if sys.version_info < (3, 11):
    pytest.skip("pipecat-flows requires Python 3.11+", allow_module_level=True)

from pipecat.frames.frames import EndFrame

from tuner_pipecat_sdk.flows_observer import FlowsObserver


@pytest.fixture
def observer():
    return FlowsObserver(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
    )


def test_attach_flow_manager_sets_provider(observer, mock_flow_manager):
    assert observer._context_provider is None
    observer.attach_flow_manager(mock_flow_manager)
    assert observer._context_provider is not None


def test_context_provider_calls_get_current_context(observer, mock_flow_manager):
    mock_flow_manager.get_current_context.return_value = [{"role": "user", "content": "Hi"}]
    observer.attach_flow_manager(mock_flow_manager)
    result = observer._context_provider()
    mock_flow_manager.get_current_context.assert_called_once()
    assert result == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_flush_without_flow_manager_warns_and_does_not_post(observer):
    assert observer._context_provider is None
    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        await observer._flush()
        post_mock.assert_not_called()


@pytest.mark.asyncio
async def test_flush_with_flow_manager_builds_and_posts(observer, mock_flow_manager):
    mock_flow_manager.get_current_context.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Bye"},
    ]
    observer.attach_flow_manager(mock_flow_manager)
    observer._acc.on_start(0)
    observer._acc.on_call_end(1_000_000_000)

    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        await observer._flush()
        post_mock.assert_called_once()
        config, payload = post_mock.call_args[0]
        assert config.call_id == "call-1"
        assert payload.call_id == "call-1"
        assert payload.call_status == "call_ended"


@pytest.mark.asyncio
async def test_handle_end_frame_triggers_flush(observer, mock_flow_manager):
    mock_flow_manager.get_current_context.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    observer.attach_flow_manager(mock_flow_manager)
    observer._acc.call_start_abs_ns = 0
    observer._acc.call_end_abs_ns = 1_000_000_000
    observer._acc.done = True
    observer._acc.latency_turns = []

    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        observer._handle(EndFrame(), 1_000_000_000)
        await asyncio.sleep(0.05)
        post_mock.assert_called_once()
        payload = post_mock.call_args[0][1]
        assert payload.call_id == "call-1"

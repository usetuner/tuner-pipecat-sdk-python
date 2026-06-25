"""Tests for Observer: plain pipecat pipeline."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from pipecat.frames.frames import EndFrame

from tuner_pipecat_sdk.observer import Observer


@pytest.fixture
def observer():
    return Observer(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
    )


def test_attach_context_is_noop(observer):
    # attach_context is a deprecated no-op: the transcript is built live from frames now.
    # Calling it must not raise and must not wire any context source.
    observer.attach_context(MagicMock())
    assert not hasattr(observer, "_context_provider")


@pytest.mark.asyncio
async def test_flush_builds_and_posts_without_context(observer):
    # No context is attached; the transcript comes from acc.live_turns. Flush still posts.
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
async def test_handle_end_frame_triggers_flush(observer):
    observer._acc.call_start_abs_ns = 0
    observer._acc.call_end_abs_ns = 1_000_000_000
    observer._acc.done = True
    observer._acc.speech_segments = []

    with (
        patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock,
        patch("tuner_pipecat_sdk._base.asyncio.create_task", side_effect=asyncio.ensure_future),
    ):
        observer._handle(EndFrame(), 1_000_000_000)
        await asyncio.sleep(0)  # single event loop yield is enough
        post_mock.assert_called_once()
        payload = post_mock.call_args[0][1]
        assert payload.call_id == "call-1"

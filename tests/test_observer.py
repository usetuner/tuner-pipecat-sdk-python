"""Tests for Observer: plain pipecat pipeline."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from pipecat.frames.frames import EndFrame, FunctionCallInProgressFrame, FunctionCallResultFrame

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


def test_wrap_graph_raises_import_error_when_langchain_unavailable(observer, monkeypatch):
    monkeypatch.setattr("tuner_pipecat_sdk.langchain_bridge._LANGCHAIN_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install tuner-pipecat-sdk\[langchain\]"):
        observer.wrap_graph(object())


def test_wrap_chain_raises_import_error_when_langchain_unavailable(observer, monkeypatch):
    monkeypatch.setattr("tuner_pipecat_sdk.langchain_bridge._LANGCHAIN_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"pip install tuner-pipecat-sdk\[langchain\]"):
        observer.wrap_chain(object())


@pytest.mark.asyncio
async def test_wrap_chain_injects_tuner_callback_and_invokes_underlying_runnable(observer):
    pytest.importorskip("tuner_langchain")

    captured_config = {}

    class _FakeChain:
        async def ainvoke(self, input, config=None, **kwargs):
            captured_config.update(config or {})
            return "ok"

    wrapped = observer.wrap_chain(_FakeChain())
    result = await wrapped.ainvoke({"input": "hi"})

    assert result == "ok"
    assert len(captured_config.get("callbacks", [])) == 1
    assert observer._langchain.active


def test_wrap_graph_and_wrap_chain_share_one_lg_accumulator(observer):
    pytest.importorskip("tuner_langchain")

    class _FakeRunnable:
        pass

    observer.wrap_chain(_FakeRunnable())
    first_acc = observer._langchain._lg_accumulator
    observer.wrap_graph(_FakeRunnable())

    assert observer._langchain._lg_accumulator is first_acc


def test_wrap_graph_and_wrap_chain_register_segment_enricher_only_once(observer):
    """The segment-merging enricher must be registered exactly once, tied to
    the lazy creation of LangchainIntegration._lg_accumulator -- not once per
    wrap_* call. Otherwise calling both wrap_chain() then wrap_graph() would
    merge the same LangChain invocations into the transcript twice."""
    pytest.importorskip("tuner_langchain")

    class _FakeRunnable:
        pass

    observer.wrap_chain(_FakeRunnable())
    observer.wrap_graph(_FakeRunnable())
    observer.wrap_chain(_FakeRunnable())

    assert len(observer._acc._segment_enrichers) == 1


def test_tool_call_frames_skipped_when_lg_accumulator_attached(observer):
    # Sentinel: the dedup guard only checks "active", so any non-None value proves it.
    observer._langchain._lg_accumulator = object()
    now_ns = time.time_ns()

    observer._handle(
        FunctionCallInProgressFrame(function_name="lookup", tool_call_id="tc-1", arguments={}),
        now_ns,
    )
    observer._handle(
        FunctionCallResultFrame(
            function_name="lookup", tool_call_id="tc-1", arguments={}, result={"ok": True}
        ),
        now_ns,
    )

    assert observer._acc.live_turns == []


def test_tool_call_frames_captured_natively_when_no_lg_accumulator(observer):
    now_ns = time.time_ns()

    observer._handle(
        FunctionCallInProgressFrame(function_name="lookup", tool_call_id="tc-1", arguments={}),
        now_ns,
    )
    observer._handle(
        FunctionCallResultFrame(
            function_name="lookup", tool_call_id="tc-1", arguments={}, result={"ok": True}
        ),
        now_ns,
    )

    kinds = [t.kind for t in observer._acc.live_turns]
    assert kinds == ["agent_function", "agent_result"]


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

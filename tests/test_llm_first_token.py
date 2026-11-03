"""llm_node_ttft from pipecat's ServiceMetricsObserver (pipecat >= 1.9).

The LLM's time to first token (or first answer token) of the inference a reply speaks, matched
by time, instead of the processing time of every inference in the turn.
"""

import asyncio

import pytest


def _processing(r, processor, secs):
    data = type("ProcessingMetricsData", (), {})()
    data.processor = processor
    data.value = secs
    r.acc.on_metrics_frame(type("Frame", (), {"data": [data]})())


def _first_token(r, kind, ms, secs):
    r.acc.on_llm_first_token(kind, r.base / 1_000_000_000 + ms / 1000, secs)


def _reply(r, tuner_config, *, start_ms=2400):
    r.turn_start(1, 1000).user_start(1000).transcription("hi", 1100).user_stop(1500)
    r.generate("hello", 1800, 2000).tts("hello", 2300).bot_start(start_ms)
    r.latency(0.05, "TTSService", user_start_ms=1000).bot_stop(start_ms + 500).turn_end(1)
    return [row for row in r.rows(tuner_config) if row.role == "agent"][-1]


def test_llm_ttft_is_the_first_token_of_the_inference_the_reply_speaks(replay, tuner_config):
    r = replay()
    r.acc.service_metrics = True
    _first_token(r, "ttfb", 1700, 0.8)  # the inference that asked for a tool
    _processing(r, "OpenAILLMService#0", 2.5)  # both inferences, start to finish
    _first_token(r, "ttfb", 2100, 0.6)  # the inference whose answer is spoken
    assert _reply(r, tuner_config).metadata["llm_node_ttft"] == 600


def test_first_answer_token_wins_over_first_byte(replay, tuner_config):
    r = replay()
    r.acc.service_metrics = True
    _first_token(r, "ttfb", 2000, 0.3)
    _first_token(r, "ttfat", 2100, 0.9)  # a model that thinks before answering
    assert _reply(r, tuner_config).metadata["llm_node_ttft"] == 900


def test_without_service_metrics_llm_ttft_is_the_processing_time(replay, tuner_config):
    r = replay()
    _first_token(r, "ttfb", 2100, 0.6)
    _processing(r, "OpenAILLMService#0", 2.5)
    assert _reply(r, tuner_config).metadata["llm_node_ttft"] == 2500


def test_a_zero_measurement_is_not_a_first_token(replay, tuner_config):
    """A static greeting has no inference; a zero TTFB reported before it must not become 0."""
    r = replay()
    r.acc.service_metrics = True
    _first_token(r, "ttfb", 100, 0.0)
    r.bot_says("welcome", 500, 900)
    r.latency(0.05, "TTSService")
    greeting = next(row for row in r.rows(tuner_config) if row.role == "agent")
    assert "llm_node_ttft" not in greeting.metadata


def test_an_inference_before_the_previous_reply_is_not_used(replay, tuner_config):
    r = replay()
    r.acc.service_metrics = True
    _first_token(r, "ttfb", 300, 0.4)
    r.bot_says("welcome", 500, 900)  # the greeting, after that inference
    _processing(r, "OpenAILLMService#0", 2.5)
    assert _reply(r, tuner_config).metadata["llm_node_ttft"] == 2500


async def test_observer_records_llm_first_token_from_metrics_frames():
    pytest.importorskip("pipecat.observers.service_metrics_observer")
    from unittest.mock import Mock

    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import TTFBMetricsData
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from tuner_pipecat_sdk.observer import Observer

    observer = Observer(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
        forward_traces=False,
    )
    frame = MetricsFrame(
        data=[
            TTFBMetricsData(processor="OpenAILLMService#0", value=0.6),
            TTFBMetricsData(processor="OpenAITTSService#0", value=0.2),
        ]
    )
    await observer.on_push_frame(
        FramePushed(
            source=Mock(),
            destination=Mock(),
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )
    await asyncio.sleep(0)  # ServiceMetricsObserver runs its event handler as a task

    assert observer._acc.service_metrics is True
    assert [(ttft, kind) for _, ttft, kind in observer._acc.llm_first_token] == [(600, "ttfb")]

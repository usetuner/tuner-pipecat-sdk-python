"""Integration concern: Pipecat FrameProcessor that orchestrates collection + delivery.

Place once in the pipeline after TTS:
    transport.input() → stt → user_agg → llm → tts → FlowsObserver → transport.output()

Before running the pipeline:
    observer.attach_flow_manager(flow_manager)

Transcript is read from flow_manager.get_current_context() at call end.
"""

import asyncio
import time
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    FunctionCallInProgressFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TTSStartedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .accumulator import FlowsAccumulator
from .client import post_call
from .config import TunerConfig
from .flow_manager_integration import attach_flow_manager_patch


class FlowsObserver(FrameProcessor):
    """FrameProcessor that accumulates pipecat-flows call data and sends it to Tuner at call end."""

    def __init__(
        self,
        api_key: str,
        workspace_id: int,
        agent_id: str,
        call_id: str,
        call_type: str = "web_call",
        base_url: str = "http://localhost:8000",
        recording_url: str = "pipecat://no-recording",
        debug: bool = False,
        asr_model: str = "",
        llm_model: str = "",
        tts_model: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._config = TunerConfig(
            api_key=api_key,
            workspace_id=workspace_id,
            agent_id=agent_id,
            call_id=call_id,
            call_type=call_type,
            base_url=base_url,
            recording_url=recording_url,
            debug=debug,
            asr_model=asr_model,
            llm_model=llm_model,
            tts_model=tts_model,
        )
        self._acc = FlowsAccumulator()
        self._flushed = False
        self._flow_manager: Any | None = None

    def attach_flow_manager(self, flow_manager: Any) -> None:
        self._flow_manager = flow_manager
        attach_flow_manager_patch(flow_manager, self._acc, self._config)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        ts = time.time_ns()
        self._handle(frame, ts)
        await self.push_frame(frame, direction)

    def _handle(self, frame, timestamp_ns: int) -> None:
        if isinstance(frame, StartFrame):
            self._acc.on_start(timestamp_ns)

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._acc.on_user_started(timestamp_ns)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._acc.on_user_stopped(frame, timestamp_ns)

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._acc.on_llm_started(timestamp_ns)

        elif isinstance(frame, TTSStartedFrame):
            # Downstream frame from TTS service — reliable ttfb anchor (always
            # arrives before BotStartedSpeakingFrame which travels upstream).
            self._acc.on_tts_started(timestamp_ns)

        elif isinstance(frame, TTSTextFrame):
            # Count chars for call-level billing metadata.
            self._acc.on_tts_text_chars(frame)

        elif isinstance(frame, FunctionCallInProgressFrame):
            self._acc.on_function_call_in_progress(frame, timestamp_ns)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._acc.on_bot_started_speaking(timestamp_ns)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._acc.on_bot_stopped(timestamp_ns)

        elif isinstance(frame, CancelFrame | EndFrame):
            self._acc.on_call_end(timestamp_ns)
            if not self._flushed:
                self._flushed = True
                asyncio.create_task(self._flush())

    async def _flush(self) -> None:
        if self._flow_manager is None:
            logger.warning("[flows-tuner] no flow_manager attached — skipping flush")
            return
        transcript = self._flow_manager.get_current_context()
        if self._config.debug:
            logger.debug("[flows-tuner] transcript ({} messages): {}", len(transcript), transcript)
        payload = self._acc.build_payload(self._config, transcript)
        logger.info("[flows-tuner] payload: {}", payload)
        await post_call(self._config, payload)

"""Integration concern: Pipecat FrameProcessor that orchestrates collection + delivery.

Place once in the pipeline after TTS:
    transport.input() → stt → user_agg → llm → tts → FlowsObserver → transport.output()

Before running the pipeline:
    observer.attach_flow_manager(flow_manager)
    observer.attach_turn_tracking_observer(turn_tracking_observer)

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
    FunctionCallResultFrame,
    MetricsFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .accumulator import FlowsAccumulator
from .client import post_call
from .config import TunerConfig

from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver


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
        self._acc.call_start_abs_ns = time.time_ns()  # anchor timeline before pipeline starts
        self._latency_observer = UserBotLatencyObserver()
        self._flushed = False
        self._flow_manager: Any | None = None

        @self._latency_observer.event_handler("on_latency_measured")
        async def _on_latency_measured(_observer: Any, latency: float) -> None:
            self._acc.on_latency_measured(latency)

        @self._latency_observer.event_handler("on_latency_breakdown")
        async def _on_latency_breakdown(_observer: Any, breakdown: Any) -> None:
            self._acc.on_latency_breakdown(breakdown)

    def attach_flow_manager(self, flow_manager: Any) -> None:
        self._flow_manager = flow_manager

    def attach_turn_tracking_observer(self, turn_tracker: Any) -> None:
        """Wire TurnTrackingObserver turn lifecycle events to the accumulator."""

        @turn_tracker.event_handler("on_turn_started")
        async def _on_turn_started(_tracker: Any, turn_number: int) -> None:
            self._acc.on_turn_started(turn_number, time.time_ns())

        @turn_tracker.event_handler("on_turn_ended")
        async def _on_turn_ended(
            _tracker: Any, turn_number: int, _duration: float, was_interrupted: bool
        ) -> None:
            self._acc.on_turn_ended(turn_number, was_interrupted)

    @property
    def latency_observer(self) -> Any:
        return self._latency_observer

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        ts = time.time_ns()
        self._handle(frame, ts)
        await self.push_frame(frame, direction)

    def _handle(self, frame, timestamp_ns: int) -> None:
        if isinstance(frame, StartFrame):
            self._acc.on_start(timestamp_ns)
            if not getattr(frame, "enable_metrics", False):
                logger.warning(
                    "[flows-tuner] enable_metrics=False — latency breakdown from "
                    "UserBotLatencyObserver will be absent. Set "
                    "PipelineParams(enable_metrics=True)."
                )
            if not getattr(frame, "enable_usage_metrics", False):
                logger.warning(
                    "[flows-tuner] StartFrame.enable_usage_metrics=False — token and TTS "
                    "character metrics will be absent. Pass "
                    "enable_usage_metrics=True to StartFrame."
                )

        elif isinstance(frame, FunctionCallInProgressFrame):
            self._acc.on_function_call_in_progress(frame, timestamp_ns)

        elif isinstance(frame, FunctionCallResultFrame):
            self._acc.on_function_call_result(frame.tool_call_id, timestamp_ns)

        elif isinstance(frame, MetricsFrame):
            self._acc.on_metrics_frame(frame)

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._acc.on_user_started_speaking(timestamp_ns)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._acc.on_user_stopped_speaking(timestamp_ns)

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

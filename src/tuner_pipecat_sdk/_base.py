"""Internal base observer — not part of the public API."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
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
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .accumulator import CallAccumulator
from .client import post_call
from .config import TunerConfig


class _BaseObserver(FrameProcessor):
    """
    Shared frame-processing logic for all Tuner observers.

    Subclasses must call ``attach_context()`` (or an equivalent wrapper) with a
    callable that returns the transcript list before the pipeline starts.
    """

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
        disconnection_reason_resolver: Callable[[], str | None] | None = None,
        **kwargs: Any,
    ) -> None:
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
        self._acc = CallAccumulator()
        self._acc.call_start_abs_ns = time.time_ns()
        self._flushed = False
        self._context_provider: Callable[[], list] | None = None
        self._disconnection_reason_resolver = disconnection_reason_resolver

        self._latency_observer = UserBotLatencyObserver()

        @self._latency_observer.event_handler("on_latency_measured")
        async def _on_latency_measured(_observer: Any, latency: float) -> None:
            self._acc.on_latency_measured(latency)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def attach_turn_tracking_observer(self, turn_tracker: Any) -> None:
        """Wire a TurnTrackingObserver's lifecycle events into the accumulator."""

        @turn_tracker.event_handler("on_turn_started")
        async def _on_turn_started(_tracker: Any, turn_number: int) -> None:
            self._acc.on_turn_started(turn_number, time.time_ns())

        @turn_tracker.event_handler("on_turn_ended")
        async def _on_turn_ended(
            _tracker: Any, turn_number: int, _duration: float, was_interrupted: bool
        ) -> None:
            self._acc.on_turn_ended(turn_number, was_interrupted)

    @property
    def latency_observer(self) -> UserBotLatencyObserver:
        return self._latency_observer

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self._handle(frame, time.time_ns())
        await self.push_frame(frame, direction)

    def _handle(self, frame: Any, timestamp_ns: int) -> None:
        if isinstance(frame, StartFrame):
            self._acc.on_start(timestamp_ns)
            if self._context_provider is None and not self._flushed:
                logger.warning(
                    "[tuner] no context_provider attached at pipeline start — "
                    "call attach_context() or attach_flow_manager() before call end "
                    "or call data will be lost at flush"
                )
            if not getattr(frame, "enable_metrics", False):
                logger.warning(
                    "[tuner] enable_metrics=False — latency breakdown will be absent. "
                    "Set PipelineParams(enable_metrics=True)."
                )
            if not getattr(frame, "enable_usage_metrics", False):
                logger.warning(
                    "[tuner] enable_usage_metrics=False — token and TTS character metrics "
                    "will be absent. Pass enable_usage_metrics=True to StartFrame."
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
            self._acc.on_user_turn_stopped(timestamp_ns)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._acc.on_bot_started_speaking(timestamp_ns)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._acc.on_bot_stopped(timestamp_ns)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._acc.on_vad_stopped(timestamp_ns)

        elif isinstance(frame, CancelFrame | EndFrame):
            if self._disconnection_reason_resolver is not None:
                try:
                    reason = self._disconnection_reason_resolver()
                    if reason:
                        self._acc.set_disconnection_reason(reason)
                except Exception:
                    logger.exception(
                        "[tuner] disconnection_reason_resolver raised an exception — ignoring"
                    )

            self._acc.on_call_end(timestamp_ns)
            if not self._flushed:
                self._flushed = True
                asyncio.create_task(self._flush())

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    async def _flush(self) -> None:
        if self._context_provider is None:
            logger.warning("[tuner] no context_provider attached — skipping flush")
            return
        transcript = self._context_provider()
        if self._config.debug:
            logger.debug("[tuner] transcript ({} messages): {}", len(transcript), transcript)
        payload = self._acc.build_payload(self._config, transcript)
        logger.info("[tuner] payload: {}", payload)
        await post_call(self._config, payload)

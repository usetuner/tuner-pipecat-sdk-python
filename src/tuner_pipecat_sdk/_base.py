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


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from either a mapping or an object with attributes.

    Lets the SIP helpers accept ``DialinSettings`` instances, plain dicts,
    or any duck-typed equivalent without the caller pre-converting.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# Provider-specific keys that hold customParameters / extra_headers in the
# WebSocket "start" payload parsed by ``parse_telephony_websocket``:
#   Twilio  → ``body``   (TwiML ``<Parameter>`` tags)
#   Telnyx  → ``customParameters``
#   Exotel  → ``custom_parameters``
_CUSTOMPARAMS_KEYS: tuple[str, ...] = ("body", "customParameters", "custom_parameters")

# Case-insensitive aliases the SIP-layer Call-ID is commonly forwarded under.
_SIP_CALL_ID_ALIASES: tuple[str, ...] = (
    "sipcallid",
    "sip_call_id",
    "sip-call-id",
    "x-sip-call-id",
)


def _collect_customparams(call_data: dict[str, Any]) -> dict[str, Any]:
    """Merge every customParameters-like dict found on ``call_data``."""
    out: dict[str, Any] = {}
    if not isinstance(call_data, dict):
        return out
    for key in _CUSTOMPARAMS_KEYS:
        v = call_data.get(key)
        if isinstance(v, dict) and v:
            out.update(v)
    return out


def _find_sip_call_id(params: dict[str, Any]) -> str | None:
    """Case-insensitive lookup for the SIP-layer Call-ID across known aliases."""
    if not params:
        return None
    lowered = {str(k).lower(): v for k, v in params.items()}
    for alias in _SIP_CALL_ID_ALIASES:
        v = lowered.get(alias)
        if v:
            return str(v)
    return None


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
        sip_call_id: str | None = None,
        sip_headers: dict[str, str] | None = None,
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
            sip_call_id=sip_call_id,
            sip_headers=sip_headers,
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

        @self._latency_observer.event_handler("on_latency_breakdown")
        async def _on_latency_breakdown(_observer: Any, breakdown: Any) -> None:
            self._acc.on_latency_breakdown(breakdown)

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

    def attach_sip_info(
        self,
        sip_call_id: str | None = None,
        sip_headers: dict[str, str] | None = None,
    ) -> None:
        """Attach SIP metadata captured from any Pipecat SIP/telephony source.

        Provider-agnostic — pass whichever identifier your provider exposes:

        * Daily PSTN/SIP: ``DialinSettings.call_id`` + ``sip_headers``
        * Twilio Media Streams: ``callSid``
        * Telnyx: ``call_control_id``
        * Plivo: ``callId``
        * Exotel: ``call_sid``
        * Any other provider: the unique call identifier and (optionally) the
          SIP INVITE headers as a flat string-to-string dict.

        Call this once the SIP info is known. For Daily, info arrives in
        ``runner_args.body``; for Twilio/Telnyx/Plivo/Exotel it arrives in the
        first WebSocket "start" message — call after
        ``parse_telephony_websocket`` returns.
        """
        if sip_call_id is not None:
            self._config.sip_call_id = sip_call_id
        if sip_headers is not None:
            self._config.sip_headers = dict(sip_headers)

    def attach_sip_from_dialin(self, dialin_settings: Any) -> None:
        """Populate SIP metadata from Pipecat's ``DialinSettings`` (Daily PSTN).

        Accepts either a ``DialinSettings`` instance or the equivalent dict
        (e.g. ``runner_args.body["dialin_settings"]``).
        """
        call_id = _get(dialin_settings, "call_id")
        headers = _get(dialin_settings, "sip_headers")
        self.attach_sip_info(sip_call_id=call_id, sip_headers=headers)

    def attach_sip_from_telephony(
        self,
        call_data: dict[str, Any],
        sip_headers: dict[str, str] | None = None,
    ) -> None:
        """Populate SIP metadata from ``parse_telephony_websocket`` output.

        Resolution order for ``sip_call_id``:

        1. **SIP-layer Call-ID** found in any nested customParameters dict
           (``call_data["body"]`` for Twilio, ``customParameters`` /
           ``custom_parameters`` for other providers) under a case-insensitive
           alias: ``SipCallId``, ``sip_call_id``, ``sip-call-id``,
           ``X-Sip-Call-Id``. This is the actual SIP Call-ID set by the trunk.
        2. **Provider native call id**: ``call_id`` (Twilio/Plivo/Exotel) or
           ``call_control_id`` (Telnyx). Used as a fallback.

        When ``sip_headers`` is not passed explicitly and the call_data carries
        customParameters, those parameters are used as the headers dict.

        For SIP-layer fields to actually appear, your trunk/webhook must
        forward them. With Twilio: add ``<Parameter name="SipCallId" .../>``
        to your TwiML response. See README "SIP / Telephony Calls" section.
        """
        nested = _collect_customparams(call_data)
        sip_layer_call_id = _find_sip_call_id(nested)
        provider_call_id = call_data.get("call_id") or call_data.get("call_control_id")
        resolved = sip_layer_call_id or provider_call_id

        if sip_headers is None and nested:
            sip_headers = {str(k): str(v) for k, v in nested.items()}

        self.attach_sip_info(sip_call_id=resolved, sip_headers=sip_headers)

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

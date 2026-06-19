"""Internal base observer — not part of the public API."""

from __future__ import annotations

import asyncio
import os
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
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ._providers import PROVIDER_EXTRACTORS, ProviderExtractor
from .accumulator import CallAccumulator
from .client import post_call
from .config import TunerConfig
from .models import CallUsage

_CI_VERSION_VARS = ("GITHUB_RUN_NUMBER", "CIRCLE_BUILD_NUM", "BUILD_NUMBER")


def resolve_agent_version(explicit: int | None) -> int | None:
    """Return agent version in priority order: explicit param → APP_VERSION → CI env vars."""
    if explicit is not None:
        return explicit
    for var in ("APP_VERSION", *_CI_VERSION_VARS):
        raw = os.environ.get(var)
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
    return None


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
        agent_version: int | None = None,
        recipient: str | None = None,
        cost_calculator: Callable[[CallUsage], float] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            recipient: Phone number or SIP URL of the callee party
                (e.g. ``"+15551234567"`` or ``"sip:alice@example.com"``).
                Optional — not collected automatically; pass it when the callee
                identity is known to your application.
        """
        # enable_direct_mode=True: frames bypass the internal process queue and are
        # pushed through immediately (like old pipecat's simple pass-through). Without
        # this, pipecat 1.0's _start_interruption() clears the queue on every
        # InterruptionFrame, dropping buffered TTS audio and cutting speech short.
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
            agent_version=resolve_agent_version(agent_version),
            recipient=recipient,
        )
        self._acc = CallAccumulator()
        self._acc.call_start_abs_ns = time.time_ns()
        self._flushed = False
        self._context_provider: Callable[[], list] | None = None
        self._disconnection_reason_resolver = disconnection_reason_resolver
        self._cost_calculator = cost_calculator

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
            if self._config.debug:
                logger.debug("[tuner] on_turn_started turn={}", turn_number)
            self._acc.on_turn_started(turn_number, time.time_ns())

        @turn_tracker.event_handler("on_turn_ended")
        async def _on_turn_ended(
            _tracker: Any, turn_number: int, _duration: float, was_interrupted: bool
        ) -> None:
            if self._config.debug:
                logger.debug(
                    "[tuner] on_turn_ended turn={} interrupted={}", turn_number, was_interrupted
                )
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

    def attach_sip_from_context(self, ctx: Any) -> None:
        """Attach SIP metadata from a typed provider context.

        Duck-typed: accepts any object exposing the attributes
        ``sip_call_id: str | None`` and ``raw_headers: dict[str, str]``.
        The canonical implementation is
        :class:`tuner_pipecat_sdk.providers.jambonz.JambonzCallContext`,
        but any equivalent dataclass (your own custom provider context)
        works the same way.
        """
        headers = getattr(ctx, "raw_headers", None) or None
        self.attach_sip_info(
            sip_call_id=getattr(ctx, "sip_call_id", None),
            sip_headers=dict(headers) if headers else None,
        )

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
        payload: dict[str, Any],
        *,
        provider: str | ProviderExtractor,
        sip_headers: dict[str, str] | None = None,
    ) -> None:
        """Populate SIP metadata from a provider's raw call payload.

        ``provider`` is required. Pass one of the built-in names —
        ``"twilio"``, ``"telnyx"``, ``"plivo"``, ``"exotel"``,
        ``"jambonz"`` — or a callable
        ``(payload) -> (sip_call_id, sip_headers)`` for any other provider.

        The expected ``payload`` is whatever the provider's server-side
        integration naturally produces:

        * Twilio / Telnyx / Plivo / Exotel: the ``call_data`` dict returned
          by Pipecat's ``parse_telephony_websocket``.
        * Jambonz: the JSON body of the call-hook webhook
          (``POST /``) — cache it on the server side and forward it to the
          bot when the WebSocket opens.

        When ``sip_headers`` is passed explicitly it overrides whatever the
        extractor derived from ``payload``.

        For SIP-layer fields to actually appear in the final Tuner record,
        your trunk/webhook must forward them. See README
        "SIP / Telephony Calls" for per-provider configuration.
        """
        if callable(provider):
            extractor = provider
        else:
            extractor = PROVIDER_EXTRACTORS.get(provider.lower())
            if extractor is None:
                raise ValueError(
                    f"unknown provider {provider!r}. "
                    f"Built-in: {sorted(PROVIDER_EXTRACTORS)}. "
                    f"Pass a callable (payload -> (call_id, headers)) "
                    f"to use a custom extractor."
                )

        resolved, derived_headers = extractor(payload)
        self.attach_sip_info(
            sip_call_id=resolved,
            sip_headers=sip_headers if sip_headers is not None else derived_headers,
        )

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
        if self._config.debug and isinstance(
            frame,
            (
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
                TTSTextFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
            ),
        ):
            logger.debug(
                "[tuner] frame {} @ {}ms", type(frame).__name__, self._acc._rel_ms(timestamp_ns)
            )

        if isinstance(frame, StartFrame):
            self._acc.on_start(timestamp_ns)
            if self._context_provider is None and not self._flushed:
                logger.warning(
                    "[tuner] no context_provider attached at pipeline start — "
                    "call attach_context() before call end "
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

        elif isinstance(frame, TTSTextFrame):
            # Voiced TTS sentence, recorded on a timeline and assigned to a bot segment by
            # window at bot-stop (TTSTextFrames arrive before BotStartedSpeakingFrame).
            self._acc.on_tts_text(frame.text, timestamp_ns)

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
        # Capture diagnostics: what the accumulator actually recorded from the pipeline.
        # If a turn that clearly fired frames is missing a segment here, the bug is frame
        # delivery to this observer (not transcript reconstruction).
        logger.info(
            "[tuner] segments ({}): {}",
            len(self._acc.speech_segments),
            [
                (s.id, s.speaker, s.start_ms, s.stop_ms, s.is_proactive, bool(s.spoken_text))
                for s in self._acc.speech_segments
            ],
        )
        logger.info(
            "[tuner] measurements: {}",
            [
                (m.user_segment_id, m.bot_segment_id, m.ttfb_ms)
                for m in self._acc.latency_measurements
            ],
        )
        payload = self._acc.build_payload(self._config, transcript, self._cost_calculator)
        logger.info("[tuner] payload: {}", payload)
        await post_call(self._config, payload)

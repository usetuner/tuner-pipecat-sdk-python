"""Internal base observer — not part of the public API."""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
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
    LLMContextAssistantTimestampFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frame_processor import FrameDirection

try:  # pipecat >= 1.9
    from pipecat.observers.speaking_observer import SpeakingObserver, SpeechEventKind
except ImportError:  # pragma: no cover - older pipecat keeps the frame-arrival timing
    SpeakingObserver = None  # type: ignore[assignment,misc,unused-ignore]
    SpeechEventKind = None  # type: ignore[assignment,misc,unused-ignore]

try:  # pipecat >= 1.9
    from pipecat.observers.function_call_observer import FunctionCallObserver
except ImportError:  # pragma: no cover - older pipecat reads the tool-call frames directly
    FunctionCallObserver = None  # type: ignore[assignment,misc,unused-ignore]

try:  # pipecat >= 1.9
    from pipecat.observers.service_metrics_observer import ServiceMetricsObserver
except ImportError:  # pragma: no cover - older pipecat keeps the processing-time fallback
    ServiceMetricsObserver = None  # type: ignore[assignment,misc,unused-ignore]

from ._deprecation import warn_deprecated
from ._providers import CARRIER_CALL_ID_READERS, PROVIDER_EXTRACTORS, ProviderExtractor
from .accumulator import CallAccumulator
from .client import post_call
from .config import TunerConfig
from .correlation import is_correlation_header, valid_correlation_id
from .langchain_bridge import LangchainIntegration
from .models import CallUsage
from .telephony import SipInfo
from .tracing import setup_call_tracing

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



def _merge_headers(
    existing: dict[str, str] | None, incoming: dict[str, str]
) -> dict[str, str]:
    """Merge ``incoming`` into ``existing``; first non-empty value per name wins.

    A malformed or oversized correlation header is dropped.
    """
    merged = dict(existing or {})
    known = {k.lower(): k for k, v in merged.items() if v}
    for name, value in incoming.items():
        if value and is_correlation_header(name):
            value = valid_correlation_id(value)
        if not value or name.lower() in known:
            continue
        merged[name] = value
        known[name.lower()] = name
    return merged

class _BaseObserver(BaseObserver):
    """
    Shared frame-processing logic for all Tuner observers.

    This is a pipeline-level observer (``PipelineTask(observers=[...])``), not a
    processor in the pipeline. It therefore sees every frame at every processor
    boundary — including frames an intermediate processor consumes without
    forwarding (e.g. ``TranscriptionFrame``, swallowed by the user aggregator) —
    and stays entirely out of the audio data path.

    The transcript is built live from the frame stream (``acc.live_turns``); no LLM context
    needs to be attached.
    """

    def __init__(
        self,
        api_key: str,
        workspace_id: int,
        agent_id: str,
        call_id: str,
        call_type: str = "web_call",
        base_url: str = "https://api.usetuner.ai",
        recording_url: str = "pipecat://no-recording",
        forward_traces: bool = True,
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
        extra_metadata: dict | None = None,
        recording_file_extension: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            api_key: Tuner API key.
            workspace_id: Tuner workspace ID.
            agent_id: Agent identifier.
            call_id: Unique call ID (use ``uuid4()``).
            call_type: Call type label. Defaults to ``"web_call"``.
            base_url: Tuner API base URL. Defaults to ``"https://api.usetuner.ai"``.
            recording_url: Recording URL, if available.
            debug: Log the full transcript at flush. Defaults to ``False``.
            asr_model: ASR model name, e.g. ``"deepgram/nova-3"``.
            llm_model: LLM model name, e.g. ``"gpt-4o-mini"``.
            tts_model: TTS model name, e.g. ``"cartesia/sonic"``.
            sip_call_id: SIP provider call identifier. See the README's SIP section
                for provider-specific extraction helpers.
            sip_headers: SIP INVITE headers as a flat dict.
            disconnection_reason_resolver: Called at flush time; return a string
                (or ``None``) describing why the call ended.
            agent_version: Deployment version number. Overrides the ``APP_VERSION``
                env var when set.
            recipient: Phone number or SIP URL of the callee party
                (e.g. ``"+15551234567"`` or ``"sip:alice@example.com"``).
                Optional — not collected automatically; pass it when the callee
                identity is known to your application.
            cost_calculator: Called at flush time with a ``CallUsage``; return the
                call's cost in USD cents.
            extra_metadata: Arbitrary call-level key/value metadata, merged into
                ``general_meta_data_raw`` (e.g. ``{"env": "prod"}``).
            recording_file_extension: Extension without a dot (e.g. ``"wav"``).
                Setting it opts the call into a presigned upload target, which
                arrives on the ``on_recording_upload`` event. The SDK never
                uploads anything itself — see ``audio_buffer()``.
        """
        super().__init__(**kwargs)
        self._config = TunerConfig(
            api_key=api_key,
            workspace_id=workspace_id,
            agent_id=agent_id,
            call_id=call_id,
            call_type=call_type,
            base_url=base_url,
            recording_url=recording_url,
            forward_traces=forward_traces,
            debug=debug,
            asr_model=asr_model,
            llm_model=llm_model,
            tts_model=tts_model,
            sip_call_id=sip_call_id,
            sip_headers=sip_headers,
            agent_version=resolve_agent_version(agent_version),
            recipient=recipient,
            extra_metadata=extra_metadata,
            recording_file_extension=recording_file_extension,
        )
        self._acc = CallAccumulator()
        self._acc.call_start_abs_ns = time.time_ns()
        self._flushed = False
        self._sip_call_id_is_fallback = False
        self._sip_internal = False
        self._langchain: LangchainIntegration = LangchainIntegration(self._acc)
        # As a pipeline-level observer, on_push_frame fires once PER processor hop, so the
        # same frame is seen many times. Dedup by frame id (bounded deque + set, mirroring
        # pipecat's UserBotLatencyObserver). All hops of a frame occur back-to-back in one
        # downstream traversal, so a modest window is sufficient.
        self._processed_frames: set[int] = set()
        self._frame_history: deque[int] = deque(maxlen=256)
        self._disconnection_reason_resolver = disconnection_reason_resolver
        self._cost_calculator = cost_calculator

        # Fired at flush with the presigned POST target, when one was issued.
        #
        # sync=True so the handler is awaited inline rather than dispatched as
        # its own task. _flush() is already running in a task nobody awaits, and
        # a handler spawned from inside it could be cut short when the process
        # exits — the exact failure this design avoids for uploads.
        #
        # Exceptions need no handling here: pipecat's _run_handler already wraps
        # every handler in try/except and logs what it catches, so a customer
        # handler that raises cannot break the flush either way.
        self._register_event_handler("on_recording_upload", sync=True)

        self._latency_observer = UserBotLatencyObserver()

        @self._latency_observer.event_handler("on_latency_measured")
        async def _on_latency_measured(_observer: Any, latency: float) -> None:
            self._acc.on_latency_measured(latency)

        @self._latency_observer.event_handler("on_latency_breakdown")
        async def _on_latency_breakdown(_observer: Any, breakdown: Any) -> None:
            self._acc.on_latency_breakdown(breakdown)

        # Pipecat's SpeakingObserver times the caller's speech from when it really began and
        # ended. It is fed the frames this observer handles (see on_push_frame), so integrations
        # register nothing extra, and it reports each stretch as it ends.
        self._speaking_observer = SpeakingObserver() if SpeakingObserver is not None else None
        if self._speaking_observer is not None:
            self._acc.speech_timing = True

            @self._speaking_observer.event_handler("on_speech_event")
            async def _on_speech_event(_observer: Any, event: Any) -> None:
                if event.kind == SpeechEventKind.USER_SPEECH_STOPPED and event.started_at:
                    self._acc.on_user_speech(event.started_at, event.timestamp)

        # Pipecat's FunctionCallObserver reports each tool call as it goes in progress and once
        # as it settles: completed, failed, timed out or cancelled. Progress updates from a
        # call that doesn't block are not results. Fed the same way as SpeakingObserver.
        self._function_call_observer = (
            FunctionCallObserver(include_results=True) if FunctionCallObserver is not None else None
        )
        if self._function_call_observer is not None:

            @self._function_call_observer.event_handler("on_function_call_event")
            async def _on_function_call_event(_observer: Any, event: Any) -> None:
                # LangChain integrations record their tool calls themselves (see _handle).
                if not self._langchain.active:
                    self._acc.on_function_call_event(event)

        # Pipecat's ServiceMetricsObserver reports each service measurement as its own record.
        # The LLM's time to first token (or first answer token) becomes llm_node_ttft.
        self._service_metrics_observer = (
            ServiceMetricsObserver() if ServiceMetricsObserver is not None else None
        )
        if self._service_metrics_observer is not None:
            self._acc.service_metrics = True

            @self._service_metrics_observer.event_handler("on_service_latency")
            async def _on_service_latency(_observer: Any, record: Any) -> None:
                kind = str(record.kind)
                if kind == "ttfat" or (kind == "ttfb" and "llm" in record.processor.lower()):
                    self._acc.on_llm_first_token(kind, record.timestamp, record.seconds)

        self._setup_trace_forwarding()

    def _setup_trace_forwarding(self) -> None:
        """Send this call's OTel spans to Tuner, tagged with the call id (ENG-1233).

        Done in the constructor because it has to happen before any span is emitted, and
        the call id is already on the config. Setting it up later would miss the spans from
        the start of the call.
        """
        if not self._config.forward_traces:
            return

        setup_call_tracing(config=self._config)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def audio_buffer(self, **kwargs: Any) -> Any:
        """Build the pipecat recorder for this call, or ``None`` if not recording.

        Returns a configured ``AudioBufferProcessor`` when
        ``recording_file_extension`` was set, and ``None`` otherwise, so the
        pipeline can splat it in unconditionally::

            audio_buffer = observer.audio_buffer()
            pipeline = Pipeline([
                ...,
                transport.output(),
                *([audio_buffer] if audio_buffer else []),
                context_aggregator.assistant(),
            ])

        Place it **after** ``transport.output()``: frames travel downstream past
        it in both directions, so one processor sees the caller's input audio
        and the agent's output audio.

        Owning the construction is what lets the SDK describe the layout in the
        call payload — ``num_channels=2`` means caller and agent are on separate
        channels, and pipecat always puts the user on the left. Asking callers to
        assert that themselves is an invitation to assert it wrongly.

        ``buffer_size=0`` keeps everything until ``stop_recording()``, so the
        call produces one recording rather than a stream of fragments.

        The SDK does not read, convert or upload the audio. Register an
        ``on_audio_data`` handler for the bytes (``to_wav()`` turns them into a
        playable file) and ``on_recording_upload`` for the presigned target.

        Args:
            **kwargs: Passed through to ``AudioBufferProcessor``, overriding the
                defaults above. Changing ``num_channels`` also changes what the
                payload reports.
        """
        if not self._config.recording_file_extension:
            return None

        from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

        params: dict[str, Any] = {
            "num_channels": 2,
            "buffer_size": 0,
            "auto_start_recording": True,
        }
        params.update(kwargs)

        buffer = AudioBufferProcessor(**params)
        self._config.recording_s3_multi_channel = buffer.num_channels > 1
        self._config.recording_s3_caller_channel = 0 if buffer.num_channels > 1 else None
        return buffer

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

    def attach_user_aggregator(self, user_aggregator: Any) -> None:
        """Wire the user aggregator's turn-stop event into the accumulator.

        on_user_turn_stopped is the single authoritative end-of-turn signal that knows
        which strategy closed the turn, so EOU is decided entirely from here — no
        dependency on UserStoppedSpeakingFrame ordering.
        """

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def _on_user_turn_stopped(
            _aggregator: Any, strategy: Any, _message: Any
        ) -> None:
            strategy_name = type(strategy).__name__
            if self._config.debug:
                logger.debug("[tuner] user turn stopped via strategy: {}", strategy_name)
            self._acc.set_turn_eou(strategy_name, time.time_ns())

    def attach_sip(
        self,
        info: SipInfo | None = None,
        *,
        sip_call_id: str | None = None,
        sip_headers: dict[str, str] | None = None,
        recipient: str | None = None,
    ) -> None:
        """Attach SIP linking data to this call. Safe to call more than once.

        ``info`` comes from :func:`tuner_pipecat_sdk.extract_sip`. Headers merge
        (explicit ``sip_headers`` beat ``info``'s; across calls the first non-empty
        value per name wins, case-insensitive). A carrier ID only
        fills an empty ``sip_call_id``; a real SIP Call-ID replaces it but never
        another real one. An explicit ``sip_call_id`` always wins. ``recipient``
        (or ``info.recipient``) is what an outbound call dialed.
        """
        # Explicit sip_headers win over info's within this call.
        headers = _merge_headers(sip_headers, info.sip_headers) if info is not None else sip_headers
        if info is not None:
            self._attach_resolved_sip(
                info.sip_call_id, headers or None, is_fallback=info.is_fallback
            )
            headers = None
        self._set_sip(sip_call_id=sip_call_id or None, sip_headers=headers)
        recipient = recipient or (info.recipient if info is not None else None)
        if recipient:
            self._config.recipient = recipient

    def attach_sip_info(
        self,
        sip_call_id: str | None = None,
        sip_headers: dict[str, str] | None = None,
    ) -> None:
        """Deprecated: use :meth:`attach_sip` (``sip_call_id=``, ``sip_headers=``).

        Sets ``sip_call_id`` when given and merges ``sip_headers`` (first
        non-empty value per name wins, case-insensitive).
        """
        if not getattr(self, "_sip_internal", False):
            warn_deprecated("attach_sip_info", "attach_sip(sip_call_id=..., sip_headers=...)")
        if sip_call_id is not None:
            self._config.sip_call_id = sip_call_id
            self._sip_call_id_is_fallback = False
        if sip_headers is not None:
            self._config.sip_headers = _merge_headers(self._config.sip_headers, sip_headers)

    def _set_sip(
        self,
        sip_call_id: str | None = None,
        sip_headers: dict[str, str] | None = None,
    ) -> None:
        # Goes through attach_sip_info so subclasses overriding it still see every attach.
        self._sip_internal = True
        try:
            self.attach_sip_info(sip_call_id=sip_call_id, sip_headers=sip_headers)
        finally:
            self._sip_internal = False

    def attach_sip_from_context(self, ctx: Any) -> None:
        """Deprecated: use ``attach_sip(extract_sip(payload, provider=...))``.

        Attach SIP metadata from a typed provider context.

        Duck-typed: accepts any object exposing the attributes
        ``sip_call_id: str | None`` and ``raw_headers: dict[str, str]``.
        The canonical implementation is
        :class:`tuner_pipecat_sdk.providers.jambonz.JambonzCallContext`,
        but any equivalent dataclass (your own custom provider context)
        works the same way.
        """
        from .providers.twilio import TwilioCallContext

        warn_deprecated("attach_sip_from_context", "attach_sip(extract_sip(...))")
        headers = getattr(ctx, "raw_headers", None) or None
        sip_call_id = getattr(ctx, "sip_call_id", None)
        if isinstance(ctx, TwilioCallContext):
            self._attach_resolved_sip(
                sip_call_id,
                dict(headers) if headers else None,
                is_fallback=bool(sip_call_id) and sip_call_id == ctx.call_sid,
            )
            return
        self._set_sip(
            sip_call_id=sip_call_id,
            sip_headers=dict(headers) if headers else None,
        )

    def _attach_resolved_sip(
        self,
        sip_call_id: str | None,
        sip_headers: dict[str, str] | None,
        *,
        is_fallback: bool,
    ) -> None:
        # A carrier ID (CallSid, call_control_id) only fills an empty slot; a
        # real SIP Call-ID replaces a carrier fallback but never another real one.
        if (
            sip_call_id
            and self._config.sip_call_id
            and (is_fallback or not self._sip_call_id_is_fallback)
        ):
            sip_call_id = None
        was_set = bool(sip_call_id)
        self._set_sip(sip_call_id=sip_call_id, sip_headers=sip_headers)
        if was_set:
            self._sip_call_id_is_fallback = is_fallback

    def attach_sip_from_dialin(self, dialin_settings: Any) -> None:
        """Deprecated: use ``attach_sip(extract_sip(dialin_settings, provider="daily"))``.

        Populate SIP metadata from Pipecat's ``DialinSettings`` (Daily PSTN).
        Accepts either a ``DialinSettings`` instance or the equivalent dict
        (e.g. ``runner_args.body["dialin_settings"]``).
        """
        warn_deprecated("attach_sip_from_dialin", 'attach_sip(extract_sip(..., provider="daily"))')
        call_id = _get(dialin_settings, "call_id")
        headers = _get(dialin_settings, "sip_headers")
        self._set_sip(sip_call_id=call_id, sip_headers=headers)

    def attach_sip_from_telephony(
        self,
        payload: dict[str, Any],
        *,
        provider: str | ProviderExtractor,
        sip_headers: dict[str, str] | None = None,
    ) -> None:
        """Deprecated: use ``attach_sip(extract_sip(payload, provider=...))``.

        A callable ``provider`` is deprecated too: use ``register_provider``.

        Populate SIP metadata from a provider's raw call payload.

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

        When ``sip_headers`` is passed explicitly it is used instead of the
        headers the extractor derived from ``payload``.

        For SIP-layer fields to actually appear in the final Tuner record,
        your trunk/webhook must forward them. See README
        "SIP / Telephony Calls" for per-provider configuration.
        """
        warn_deprecated("attach_sip_from_telephony", "attach_sip(extract_sip(...))")
        if callable(provider):
            warn_deprecated("attach_sip_from_telephony(provider=<callable>)", "register_provider")
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
        headers = sip_headers if sip_headers is not None else derived_headers
        reader = None if callable(provider) else CARRIER_CALL_ID_READERS.get(provider.lower())
        if reader is None:
            self._set_sip(sip_call_id=resolved, sip_headers=headers)
            return
        is_fallback = bool(resolved) and isinstance(payload, dict) and resolved == reader(payload)
        self._attach_resolved_sip(resolved, headers, is_fallback=is_fallback)

    @property
    def latency_observer(self) -> UserBotLatencyObserver:
        return self._latency_observer

    def wrap_graph(self, graph: Any, capture: Any = None) -> Any:
        """Wrap a LangGraph graph for Tuner observability.

        Returns a drop-in replacement for ``graph`` — pass it straight to
        pipecat's ``LangchainProcessor`` (or your own processor that drives a
        LangChain runnable):

            observer = Observer(...)
            wrapped = observer.wrap_graph(my_compiled_graph)
            lc = LangchainProcessor(chain=wrapped)

        Captures tool calls, node transitions, and LLM token usage via
        tuner-langchain's callback handler — independent of whatever pipecat
        frames the driving processor does or doesn't emit.

        Requires the ``langchain`` extra: ``pip install tuner-pipecat-sdk[langchain]``.
        """
        return self._langchain.wrap_graph(graph, capture)

    def wrap_chain(self, chain: Any, capture: Any = None) -> Any:
        """Wrap a plain LangChain runnable for Tuner observability.

        Same as :meth:`wrap_graph`, for non-graph LangChain pipelines. See that
        method's docstring for usage and requirements.
        """
        return self._langchain.wrap_chain(chain, capture)

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def on_push_frame(self, data: FramePushed) -> None:
        # Pipeline-level observer: invoked for every frame transfer between processors.
        # Only the downstream pass matters, and each frame must be handled once (it fires
        # once per hop). Stamp with wall-clock time — the accumulator works in time.time_ns()
        # relative to call_start_abs_ns, not the pipeline clock.
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        first_push = getattr(data, "first_push", None)
        if first_push is not None:
            # Pipecat >= 1.12 marks a frame's first push, the moment its own observers are told
            # about it. A frame waiting in a processor's queue can come back after any number
            # of other frames, so a window of recent ids can't tell it has been seen.
            if not first_push:
                return
        else:
            if data.frame.id in self._processed_frames:
                return
            self._processed_frames.add(data.frame.id)
            self._frame_history.append(data.frame.id)
            if len(self._processed_frames) > len(self._frame_history):
                self._processed_frames = set(self._frame_history)
        for observer in self._pipecat_observers:
            await observer.on_push_frame(data)
        self._handle(data.frame, time.time_ns())

    @property
    def _pipecat_observers(self) -> list[BaseObserver]:
        """The pipecat observers this observer feeds, as pipecat would feed them."""
        return [
            observer
            for observer in (
                self._speaking_observer,
                self._function_call_observer,
                self._service_metrics_observer,
            )
            if observer is not None
        ]

    async def cleanup(self) -> None:
        await super().cleanup()
        for observer in self._pipecat_observers:
            await observer.cleanup()

    # Frame type-name substrings logged when debug=True. Covers the transcript-relevant frames
    # (speech boundaries, STT/LLM/TTS/aggregated text, the assistant-commit timestamp frame, tool
    # calls, interruptions) while skipping high-volume audio frames.
    _DEBUG_FRAME_NAME_PARTS = (
        "BotStarted",
        "BotStopped",
        "UserStarted",
        "UserStopped",
        "Transcription",
        "Text",  # TTSTextFrame, LLMTextFrame, AggregatedTextFrame, AggregatedTextProgressFrame
        "LLMFullResponse",
        "LLMContext",  # LLMContextAssistantTimestampFrame (the commit marker)
        "Interruption",
        "FunctionCall",
        "TTSStarted",
        "TTSStopped",
        "UtteranceEnd",
    )

    def _handle(self, frame: Any, timestamp_ns: int) -> None:
        if self._config.debug:
            name = type(frame).__name__
            if any(part in name for part in self._DEBUG_FRAME_NAME_PARTS):
                text = getattr(frame, "text", None)
                extra = f" text={text!r}" if text else ""
                logger.debug(
                    "[tuner] frame {} @ {}ms{}", name, self._acc._rel_ms(timestamp_ns), extra
                )

        if isinstance(frame, StartFrame):
            self._acc.on_start(timestamp_ns)
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
            # When a LangChain accumulator is attached (wrap_graph/wrap_chain), tool
            # calls are captured via tuner-langchain's callback handler instead —
            # skip native frame-based capture so a processor that emits both never
            # double-records the same call. This is a call-wide flag, not scoped to
            # the wrapped runnable: if a pipeline ever mixed a wrap_graph()/wrap_chain()
            # runnable with a separate native tool-calling LLM service in the same
            # call, that native service's tool calls would also be silently skipped
            # here. Not a supported configuration today (Observer expects at most one
            # tool-calling surface per call), but worth knowing if that ever changes.
            if not self._langchain.active and self._function_call_observer is None:
                self._acc.on_function_call_in_progress(frame, timestamp_ns)

        elif isinstance(frame, FunctionCallResultFrame):
            if not self._langchain.active and self._function_call_observer is None:
                self._acc.on_function_call_result(frame, timestamp_ns)

        elif isinstance(frame, MetricsFrame):
            self._acc.on_metrics_frame(frame)

        elif isinstance(frame, TranscriptionFrame):
            # Finalized STT text — the authoritative signal for which context user messages
            # were actually spoken (vs developer-injected {"role":"user"} messages). Only
            # visible now that this is a pipeline-level observer: the user aggregator consumes
            # TranscriptionFrame and never forwards it downstream. InterimTranscriptionFrame is
            # a sibling, not a subclass, so interim results are naturally excluded.
            self._acc.on_transcription(frame.text, timestamp_ns)

        elif isinstance(frame, TTSTextFrame):
            # Voiced TTS sentence, recorded on a timeline and assigned to a bot segment by
            # window at bot-stop (TTSTextFrames arrive before BotStartedSpeakingFrame).
            self._acc.on_tts_text(frame.text, timestamp_ns)

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._acc.on_llm_response_start(timestamp_ns)

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._acc.on_llm_response_end(timestamp_ns)

        elif isinstance(frame, LLMTextFrame):
            # Generated LLM text — accumulated into the in-progress assistant turn (committed only
            # when its LLMContextAssistantTimestampFrame arrives).
            self._acc.on_llm_text(frame.text)

        elif isinstance(frame, LLMContextAssistantTimestampFrame):
            # Pipecat pushes this exactly when it commits an assistant message to the LLM context.
            # It is the authoritative signal for which assistant turns are real — an interrupted
            # response that never commits never gets this frame, so it never appears.
            self._acc.on_assistant_commit(timestamp_ns)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._acc.on_vad_user_stopped_speaking(timestamp_ns)

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._acc.on_user_started_speaking(timestamp_ns)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._acc.on_user_stopped_speaking(timestamp_ns)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._acc.on_bot_started_speaking(timestamp_ns)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._acc.on_bot_stopped(timestamp_ns)

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
        # The transcript is built live from the frame stream (acc.live_turns); it no longer
        # depends on the LLM context. Capture diagnostics: what the accumulator recorded from
        # the pipeline. If a turn that clearly fired frames is missing here, the bug is frame
        # delivery to this observer.
        if self._config.debug:
            logger.debug(
                "[tuner] live_turns ({}): {}",
                len(self._acc.live_turns),
                [(t.kind, t.order, (t.text or "")[:40]) for t in self._acc.live_turns],
            )
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
        payload = self._acc.build_payload(self._config, self._cost_calculator)
        response = await post_call(self._config, payload)
        await self._emit_recording_upload(response)

    async def _emit_recording_upload(self, response: dict[str, Any] | None) -> None:
        """Hand the presigned POST target to the application, if one was issued.

        Deliberately the end of the SDK's involvement: uploading here would put
        a multi-megabyte POST on the call's teardown path, inside a task nobody
        awaits, where a process exiting would truncate it. Whoever owns the
        agent decides when to upload, whether to retry, and whether to upload at
        all.

        The target is a presigned S3 POST — ``{"url", "fields", "key",
        "content_type", "max_bytes"}`` — and it expires. Anything that queues it
        for later needs to upload inside that window.
        """
        # post_call already narrows to dict | None, but this is the boundary
        # between the HTTP layer and application code — a non-dict here should
        # be ignored rather than raise inside an unawaited flush task.
        if not isinstance(response, dict):
            return
        target = response.get("recording_upload")
        if not isinstance(target, dict) or not target:
            # No target: the call did not set recording_file_extension, or the
            # API declined to issue one. Not an error.
            return

        logger.info(
            "[tuner] recording upload target issued for {} (key={})",
            self._config.call_id,
            target.get("key", "?"),
        )
        await self._call_event_handler("on_recording_upload", target)

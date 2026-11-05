"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from loguru import logger

from ._pipecat_compat import LegacyTiming
from .models import (
    CallPayload,
    CallUsage,
    LatencyMeasurement,
    LiveTurn,
    SegmentEnricher,
    SpeechSegment,
)
from .payload_builder import build_payload
from .tool_timing_registry import ToolTimingRegistry

# TTSTextFrames are emitted slightly before BotStartedSpeakingFrame. A bot segment claims
# voiced TTS sentences from up to this many ms before its start. Wide enough to cover the
# TTS lead, far below the multi-second gap to an interrupted-before-voiced response.
_TTS_LEAD_SLACK_MS = 2000

# UserStartedSpeakingFrame trails the speech it reports by the VAD's start_secs (0.2 s by
# default). Speech that ended this long before a turn's start frame is still that turn's.
_SPEECH_LEAD_MS = 500


@dataclass
class CallAccumulator(LegacyTiming):
    """Collects runtime events and produces a final call payload. Timing used only without
    pipecat's newer observers comes from LegacyTiming (_pipecat_compat)."""

    # call-level timing
    call_start_abs_ns: int = 0
    call_end_abs_ns: int = 0

    # Speech + latency model (append-only segments; measurements linked by segment id).
    speech_segments: list[SpeechSegment] = field(default_factory=list)
    latency_measurements: list[LatencyMeasurement] = field(default_factory=list)
    _next_segment_id: int = field(default=0, repr=False)

    # Finalized STT transcriptions, in order: (text, rel_ms). Real user speech produces a
    # transcription; developer-injected {"role":"user"} context messages do not. The enricher
    # uses this as the authoritative signal for which context user messages were actually
    # spoken (mirrors spoken_text/TTS matching on the bot side).
    user_transcriptions: list[tuple[str, int]] = field(default_factory=list)

    # Finalized LLM generations, in order: (text, rel_ms). The assistant aggregator builds each
    # assistant context message from exactly these LLMTextFrames, so a genuinely-generated message
    # matches one of these; a developer-injected {"role":"assistant"} context message matches none.
    # The enricher uses this as the authoritative real-vs-injected signal for assistant messages —
    # the mirror of user_transcriptions on the bot side.
    assistant_generations: list[tuple[str, int]] = field(default_factory=list)

    # Active turn tracking
    _active_turn_number: int | None = field(default=None, repr=False)
    # turn_number → user segment id, so on_turn_ended can target the right response.
    _turn_to_user_segment_id: dict[int, int] = field(default_factory=dict, repr=False)
    # User segment that turn/VAD/STT events currently attach to.
    _active_user_segment_id: int | None = field(default=None, repr=False)
    # Bot segment currently speaking; cleared on on_bot_stopped.
    _current_bot_segment_id: int | None = field(default=None, repr=False)
    # User segment the in-progress bot response is answering. Stable across multiple
    # bot speech segments within one response (speak → tool → speak). Re-pointed on
    # each on_turn_started, which is what signals a fresh exchange.
    _response_user_segment_id: int | None = field(default=None, repr=False)
    # Latency measurement awaiting its on_latency_breakdown payload.
    _pending_measurement: LatencyMeasurement | None = field(default=None, repr=False)
    # Timeline of voiced TTS sentences: (text, rel_ms). TTSTextFrames arrive slightly
    # BEFORE BotStartedSpeakingFrame, so we can't tie them to a segment at arrival time.
    # At on_bot_stopped each bot segment claims the TTS in its [start - lead, stop] window.
    # Text generated for a response that was interrupted before it was voiced (no
    # BotStartedSpeaking) falls outside every window and is dropped — never marked spoken.
    _tts_timeline: list[tuple[str, int]] = field(default_factory=list, repr=False)

    # In-progress LLM response text, accumulated across LLMTextFrames between
    # LLMFullResponseStartFrame and LLMFullResponseEndFrame, then finalized into
    # assistant_generations at the End frame.
    _llm_response_buffer: list[str] = field(default_factory=list, repr=False)

    # tool call timing keyed by tool_call_id
    registry: ToolTimingRegistry = field(default_factory=ToolTimingRegistry)

    # call-level pipecat-sourced counters (summed across all MetricsFrames)
    _pipecat_llm_total_tokens: int = field(default=0, repr=False)
    _pipecat_llm_prompt_tokens: int = field(default=0, repr=False)
    _pipecat_llm_completion_tokens: int = field(default=0, repr=False)
    _pipecat_tts_chars: int = field(default=0, repr=False)

    # call-level externally-sourced counters (summed across all on_llm_end callbacks
    # reported via a hook like langchain_bridge._AccumulatorUsageSink). Safe to sum
    # unconditionally alongside the _pipecat_llm_* counters above ONLY because pipecat's
    # LangchainProcessor never emits MetricsFrame/LLMUsageMetricsData itself — the
    # externally-sourced number is the only number that ever exists for those turns. If
    # a future pipecat processor driving such a runnable ever starts emitting usage
    # MetricsFrames too, this summing would double-count; see test_langchain_bridge.py's
    # pinned regression test.
    _external_llm_prompt_tokens: int = field(default=0, repr=False)
    _external_llm_completion_tokens: int = field(default=0, repr=False)

    # per-turn pending pipecat metrics (reset on each latency breakdown)
    _pending_pipecat_llm_processing_s: float = field(default=0.0, repr=False)
    _pending_pipecat_tts_processing_s: float = field(default=0.0, repr=False)

    # per-turn pending externally-sourced LLM call duration (reset on each latency
    # breakdown), reported via a hook like langchain_bridge._AccumulatorUsageSink from
    # tuner-langchain's on_llm_end callback. This is total on_llm_start->on_llm_end
    # duration, not a true time-to-first-token -- see on_latency_breakdown()'s comment.
    # None (not 0) means "nothing reported this turn", distinct from a genuine 0ms call.
    _pending_external_llm_duration_ms: int | None = field(default=None, repr=False)

    # Transforms applied to the final transcript segment list at payload-build
    # time, in registration order -- see register_segment_enricher(). Keeps this
    # class (and payload_builder.py) ignorant of which optional integrations
    # exist; each integration (e.g. the LangChain bridge) registers its own
    # transform instead of being imported and called by name here.
    _segment_enrichers: list[SegmentEnricher] = field(default_factory=list, repr=False)

    # Turn-started calls that arrived before StartFrame (call_start_abs_ns not yet set).
    # Processed retroactively in on_start once the reference timestamp is known.
    _pending_turn_starts: list[tuple[int, int]] = field(default_factory=list, repr=False)

    # Set to True on the first UserStartedSpeakingFrame — used to distinguish
    # the proactive bot greeting from mid-conversation tool or node transitions.
    _user_has_spoken: bool = field(default=False, repr=False)

    # Event-sourced transcript: turns built live from frames, in arrival order. This is the
    # transcript substrate (text + order); the speech_segments/measurements above remain the
    # latency substrate. The two are joined by segment id snapshotted at frame time, never by text.
    live_turns: list[LiveTurn] = field(default_factory=list)
    _turn_order: int = field(default=0, repr=False)
    # The user turn currently accruing transcriptions (one per active user segment).
    _open_user_turn: LiveTurn | None = field(default=None, repr=False)
    # The user turn committed last, until the bot starts speaking. A final transcript that lands
    # after its turn closed, before the bot replies, belongs to it.
    _last_user_turn: LiveTurn | None = field(default=None, repr=False)
    # The agent turn currently accruing LLM text / awaiting voicing.
    _open_agent_turn: LiveTurn | None = field(default=None, repr=False)
     # Timestamp (ns) of the most recent VAD-detected speech-end within the currently open
    # user turn. Anchors the silence-timeout EOU fallback below when no turn-analyzer
    # verdict arrives before the turn commits. Reset after each turn closes.
    _last_vad_stop_ns: int | None = field(default=None, repr=False)
    # The caller's speech as pipecat's SpeakingObserver times it, as (start_ms, end_ms): when it
    # really began and ended, with the VAD's confirmation delay taken off. Matched to user turns
    # by time overlap when rows are built, never by order, so a missed stretch moves nothing else.
    user_speech: list[tuple[int, int]] = field(default_factory=list)
    # True while SpeakingObserver feeds user_speech (pipecat >= 1.9). Agent rows then keep the
    # start seen at BotStartedSpeakingFrame, and latency runs from the end of the caller's speech.
    speech_timing: bool = False
    # The LLM's time to its first token for each inference, as pipecat's ServiceMetricsObserver
    # reports it: (observed_ms, ttft_ms, kind), kind "ttfat" (first answer token, after any
    # thinking) or "ttfb". Matched to the reply that follows it, by time.
    llm_first_token: list[tuple[int, int, str]] = field(default_factory=list)
    # True while ServiceMetricsObserver feeds llm_first_token (pipecat >= 1.9).
    service_metrics: bool = False

    # misc
    done: bool = False

    # ended reason (write-once: first meaningful value wins)
    _disconnection_reason: str = field(default="", repr=False)

    def _rel_ms(self, abs_ns: int) -> int:
        if self.call_start_abs_ns == 0 or abs_ns == 0:
            return 0
        return (abs_ns - self.call_start_abs_ns) // 1_000_000

    def _abs_to_rel_ms(self, abs_unix_s: float | None) -> int:
        if self.call_start_abs_ns == 0 or not abs_unix_s:
            return 0
        call_start_s = self.call_start_abs_ns / 1_000_000_000
        return max(0, int((abs_unix_s - call_start_s) * 1000))

    def _append_segment(self, speaker: str, start_ms: int = 0, **kwargs: Any) -> SpeechSegment:
        seg = SpeechSegment(id=self._next_segment_id, speaker=speaker, start_ms=start_ms, **kwargs)
        self._next_segment_id += 1
        self.speech_segments.append(seg)
        return seg

    def _append_live(self, turn: LiveTurn) -> LiveTurn:
        """Append a LiveTurn in arrival/commit order, assigning its order index."""
        turn.order = self._turn_order
        self._turn_order += 1
        self.live_turns.append(turn)
        return turn

    def _new_turn(self, kind: str, **kwargs: Any) -> LiveTurn:
        """Create and immediately append a LiveTurn (agent/tool rows)."""
        return self._append_live(LiveTurn(kind=kind, order=0, **kwargs))

    def _segment_by_id(self, segment_id: int | None) -> SpeechSegment | None:
        if segment_id is None:
            return None
        for seg in self.speech_segments:
            if seg.id == segment_id:
                return seg
        return None

    def get_tool_invocation_ms(self, tool_call_id: str) -> int | None:
        abs_ns = self.registry.get_invocation_ns(tool_call_id)
        return self._rel_ms(abs_ns) if abs_ns else None

    def get_tool_completion_ms(self, tool_call_id: str) -> int | None:
        abs_ns = self.registry.get_completion_ns(tool_call_id)
        return self._rel_ms(abs_ns) if abs_ns else None

    def get_total_llm_tokens(self) -> int:
        return (
            self._pipecat_llm_total_tokens
            + self._external_llm_prompt_tokens
            + self._external_llm_completion_tokens
        )

    def get_llm_prompt_tokens(self) -> int:
        return self._pipecat_llm_prompt_tokens + self._external_llm_prompt_tokens

    def get_llm_completion_tokens(self) -> int:
        return self._pipecat_llm_completion_tokens + self._external_llm_completion_tokens

    def get_total_tts_characters(self) -> int:
        return self._pipecat_tts_chars

    def record_external_llm_usage(
        self, prompt_tokens: int | None, completion_tokens: int | None
    ) -> None:
        """Record LLM token usage reported by a turn driven outside pipecat's own
        frame stream (e.g. a LangChain/LangGraph runnable).

        Called by ``langchain_bridge._AccumulatorUsageSink`` from tuner-langchain's
        ``on_llm_end`` callback today; any other external integration can reuse the
        same hook. See the class-level comment on the ``_external_llm_*`` fields for
        why summing with pipecat-sourced counters is safe today. ``None`` (a provider
        or turn that didn't report usage) is treated as ``0``, not an error.
        """
        self._external_llm_prompt_tokens += max(0, prompt_tokens or 0)
        self._external_llm_completion_tokens += max(0, completion_tokens or 0)

    def record_external_llm_duration_ms(self, duration_ms: int | None) -> None:
        """Record the duration of the most recent LLM call in a turn driven outside
        pipecat's own frame stream (e.g. a LangChain/LangGraph runnable).

        Called by ``langchain_bridge._AccumulatorUsageSink`` from tuner-langchain's
        ``on_llm_end`` callback. Consumed (and reset) by ``on_latency_breakdown()``.
        If the turn made multiple LLM calls (e.g. a tool-calling round-trip), only
        the last one's duration survives -- this is overwritten, not accumulated.
        """
        if duration_ms is not None:
            duration_ms = max(0, duration_ms)
        self._pending_external_llm_duration_ms = duration_ms

    def register_segment_enricher(self, enricher: SegmentEnricher) -> None:
        """Register a transform applied to the transcript segment list at
        payload-build time.

        Called by ``Observer.wrap_graph()``/``wrap_chain()`` the first time
        either is used, to merge in LangChain-sourced node/tool segments. This
        accumulator has no idea what the registered callable does or where it
        came from -- it just applies every registered enricher, in order, to
        the native segment list before building the payload.
        """
        self._segment_enrichers.append(enricher)

    @property
    def disconnection_reason(self) -> str | None:
        return self._disconnection_reason or None

    def set_disconnection_reason(self, reason: str) -> None:
        """Write-once: first meaningful value wins, subsequent calls are no-ops.

        The disconnection_reason_resolver on the observer also calls this at flush time.
        """
        if not self._disconnection_reason and reason:
            self._disconnection_reason = reason

    def on_start(self, timestamp_ns: int) -> None:
        # call_start_abs_ns is pre-initialized in the observer __init__ to avoid the
        # StartFrame race condition (TTS queues greeting audio before StartFrame, causing
        # StartFrame to arrive late at the observer). Only fall back to StartFrame time
        # if somehow not yet set.
        if self.call_start_abs_ns == 0:
            self.call_start_abs_ns = timestamp_ns
        # Retroactively process any on_turn_started calls that arrived before on_start.
        for turn_number, ts in self._pending_turn_starts:
            self.on_turn_started(turn_number, ts)
        self._pending_turn_starts.clear()

    def on_turn_started(self, turn_number: int, timestamp_ns: int) -> None:
        """Append a new user SpeechSegment for an incoming user speech segment.

        Every turn-start opens its own segment — there is no collapse. Whether two
        consecutive user utterances are one thought (a mid-sentence VAD pause) or a
        silence gap (the user waiting on an unresponsive agent) is decided later, as
        a presentation choice in the transcript enricher gated by the gap duration.
        Keeping every segment here means the silence gap is never destroyed.

        Re-pointing ``_response_user_segment_id`` to the new segment is what marks a
        fresh exchange for bot-response attribution.
        """
        if self.call_start_abs_ns == 0:
            self._pending_turn_starts.append((turn_number, timestamp_ns))
            return

        seg = self._append_segment(
            "user", start_ms=self._rel_ms(timestamp_ns), turn_number=turn_number
        )
        self._active_turn_number = turn_number
        self._turn_to_user_segment_id[turn_number] = seg.id
        self._active_user_segment_id = seg.id
        self._response_user_segment_id = seg.id

    def on_turn_ended(self, turn_number: int, was_interrupted: bool) -> None:
        """Called by TurnTrackingObserver when a turn ends (bot finished or interrupted)."""
        # Target the measurement for this turn's user segment (the bot may have already
        # stopped, so _current_bot_segment_id can be cleared — find by linkage instead).
        user_seg_id = self._turn_to_user_segment_id.get(turn_number)
        measurement = next(
            (m for m in reversed(self.latency_measurements) if m.user_segment_id == user_seg_id),
            self._pending_measurement,
        )
        if measurement is not None:
            measurement.was_interrupted = was_interrupted
            bot_seg = self._segment_by_id(measurement.bot_segment_id)
            if bot_seg is not None:
                bot_seg.interrupted = was_interrupted
        self._active_turn_number = None

    def _ensure_open_user_turn(self, start_ms: int) -> LiveTurn:
        """Open the in-progress user turn if none is accruing (created but not yet appended —
        it commits to the transcript at UserStoppedSpeaking, like pipecat's aggregator)."""
        if self._open_user_turn is None:
            self._open_user_turn = LiveTurn(
                kind="user",
                order=0,
                start_ms=start_ms,
                user_segment_id=self._active_user_segment_id,
            )
        return self._open_user_turn

    def on_user_started_speaking(self, timestamp_ns: int) -> None:
        """Use frame timestamp as the authoritative user-start anchor for the active segment."""
        self._user_has_spoken = True
        # If bot is currently speaking, this is an interruption.
        # Record when the user started cutting in on the BOT's current segment.
        bot_seg = self._segment_by_id(self._current_bot_segment_id)
        if bot_seg is not None:
            bot_seg.interrupted_at_ms = self._rel_ms(timestamp_ns)

        started_ms = self._rel_ms(timestamp_ns)
        self._ensure_open_user_turn(started_ms)

        seg = self._segment_by_id(self._active_user_segment_id)
        if seg is None:
            return
        if seg.start_ms == 0:
            seg.start_ms = started_ms
        else:
            seg.start_ms = min(seg.start_ms, started_ms)
        seg.windows.append([started_ms, None])

    def on_transcription(self, text: str, timestamp_ns: int) -> None:
        """Record a finalized STT transcription and accrue it onto the in-progress user turn.

        A user row groups exactly the transcriptions between two UserStoppedSpeaking boundaries —
        the same unit pipecat's aggregator commits as one user message — committed (appended) at
        on_user_stopped_speaking. A turn exists iff a real transcription arrived, so
        developer-injected ``{"role":"user"}`` context messages (which never transcribe) never
        appear.
        """
        if not (text and text.strip()):
            return
        rel = self._rel_ms(timestamp_ns)
        self.user_transcriptions.append((text, rel))  # kept for diagnostics
        if self._open_user_turn is None and self._last_user_turn is not None:
            # STT can finalize an utterance after its turn closed. The caller hasn't spoken
            # since and the bot hasn't replied, so it was said in that turn.
            turn = self._last_user_turn
            if not turn.chunks:
                # None of the turn's transcript had arrived when it closed, so it wasn't kept.
                # It is now, with the times it was spoken.
                self._append_live(turn)
            turn.chunks.append((text, rel))
            return
        self._ensure_open_user_turn(rel).chunks.append((text, rel))

    def on_tts_text(self, text: str, timestamp_ns: int) -> None:
        """Record a voiced TTS sentence on the timeline. Assigned to a bot segment by time
        window at on_bot_stopped (TTSTextFrames arrive before BotStartedSpeakingFrame)."""
        if text and text.strip():
            self._tts_timeline.append((text, self._rel_ms(timestamp_ns)))

    def _finalize_llm_response(self, timestamp_ns: int) -> None:
        """Join the buffered LLM text into one generation and record it."""
        joined = "".join(self._llm_response_buffer).strip()
        if joined:
            self.assistant_generations.append((joined, self._rel_ms(timestamp_ns)))
        self._llm_response_buffer = []

    def on_llm_response_start(self, timestamp_ns: int = 0) -> None:
        """Begin a new LLM response — open a fresh (not-yet-committed) agent turn.

        The previous open turn, if it never received a commit (LLMContextAssistantTimestampFrame),
        is abandoned here: pipecat discards an assistant response that was interrupted before it
        committed, so we mirror that by simply not appending it. A turn that was voiced and not
        cut off is kept instead (see ``_keep_voiced_turn``).
        """
        if self._open_agent_turn is not None:
            self._keep_voiced_turn(self._open_agent_turn)
        if self._llm_response_buffer:
            self._finalize_llm_response(0)
        self._llm_response_buffer = []
        self._open_agent_turn = LiveTurn(
            kind="agent", order=0, generated_ms=self._rel_ms(timestamp_ns)
        )

    def _keep_voiced_turn(self, turn: LiveTurn) -> None:
        """Keep an uncommitted agent turn the caller actually heard.

        Two kinds of voiced speech never get a commit before the next LLM response starts:
        static speech (a TTSSpeakFrame with append_to_context=False, e.g. a greeting or a
        transition line, which has no LLM text) and an LLM reply whose commit lags behind the
        next response (e.g. a read-back followed by a tool call). Static speech is always kept.
        An LLM reply is kept only if no user speech started during it, so interrupted replies
        are still dropped like pipecat does. The turn goes back to where it was voiced.
        """
        seg = self._segment_by_id(turn.bot_segment_id)
        if seg is None or turn.voiced_order is None:
            return
        if turn.text.strip() and self._was_cut_off(seg):
            return
        index = next(
            (i for i, t in enumerate(self.live_turns) if t.order >= turn.voiced_order),
            len(self.live_turns),
        )
        self.live_turns.insert(index, turn)
        for order, t in enumerate(self.live_turns):
            t.order = order
        self._turn_order = len(self.live_turns)

    def _was_cut_off(self, seg: SpeechSegment) -> bool:
        """True when user speech started while this bot segment was playing. BotStoppedSpeaking
        can arrive just before UserStartedSpeaking on a barge-in, so check user start times too."""
        if seg.interrupted_at_ms is not None:
            return True
        end_ms = seg.stop_ms or None
        user_turns = [t for t in self.live_turns if t.kind == "user"]
        if self._open_user_turn is not None:
            user_turns.append(self._open_user_turn)
        return any(
            seg.start_ms <= t.start_ms and (end_ms is None or t.start_ms <= end_ms)
            for t in user_turns
        )

    def on_llm_text(self, text: str) -> None:
        """Accumulate a streamed LLM text chunk for the in-progress response."""
        if text:
            self._llm_response_buffer.append(text)
            if self._open_agent_turn is not None:
                self._open_agent_turn.text += text

    def on_llm_response_end(self, timestamp_ns: int) -> None:
        """Finalize the in-progress LLM response. The assistant aggregator builds the context
        message from these same chunks, so this also feeds the diagnostic generations list."""
        self._finalize_llm_response(timestamp_ns)
        if self._open_agent_turn is not None:
            self._open_agent_turn.generated_ms = self._rel_ms(timestamp_ns)

    def on_assistant_commit(self, timestamp_ns: int) -> None:
        """Commit the in-progress assistant turn to the transcript.

        Driven by ``LLMContextAssistantTimestampFrame``, which pipecat pushes exactly when it
        commits an assistant message to the LLM context. An assistant response that never reaches
        a commit (interrupted before it was committed) is therefore never appended — matching
        pipecat's committed transcript precisely.
        """
        turn = self._open_agent_turn
        if turn is not None and (turn.text.strip() or turn.bot_segment_id is not None):
            self._append_live(turn)
        self._open_agent_turn = None

    def on_vad_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Track the latest VAD-detected speech-end within the open user turn. A turn can
        reopen after a VAD stop (the user resumes speaking), so only the LAST one before
        the turn actually commits is the real silence anchor — this simply records the most
        recent one seen; set_turn_eou decides whether it's needed."""
        self._last_vad_stop_ns = timestamp_ns

    def on_user_speech(self, started_at_s: float, stopped_at_s: float) -> None:
        """Record one stretch of the caller's speech, in unix seconds, as SpeakingObserver
        reports it when the stretch ends."""
        if self.call_start_abs_ns == 0:
            return
        call_start_s = self.call_start_abs_ns / 1_000_000_000
        # Rounded, not truncated: float seconds would otherwise lose a millisecond.
        start_ms = max(0, round((started_at_s - call_start_s) * 1000))
        end_ms = max(0, round((stopped_at_s - call_start_s) * 1000))
        if end_ms > start_ms:
            self.user_speech.append((start_ms, end_ms))

    def speech_bounds(
        self, start_ms: int, end_ms: int, after_ms: int = 0
    ) -> tuple[int, int] | None:
        """The caller's real speech within a user turn seen from start_ms to end_ms: the first
        start and last end of the stretches that belong to it. None when none does.

        A stretch belongs to the turn when it ended by end_ms, since UserStoppedSpeakingFrame
        trails the speech it closes; one still going then is the next turn's. It must start at
        or after after_ms, the previous user row's end, so no two rows share a stretch.

        start_ms is when UserStartedSpeakingFrame arrived, which trails the speech by the VAD's
        start_secs. A stretch shorter than that ("Eh") ends before it, so one that started up to
        _SPEECH_LEAD_MS earlier still counts. One that ended before start_ms but started earlier
        than that was other speech."""
        lead_ms = start_ms - _SPEECH_LEAD_MS
        inside = [
            (s, e)
            for s, e in self.user_speech
            if after_ms <= s and e <= end_ms and (e > start_ms or s >= lead_ms)
        ]
        if not inside:
            return None
        return min(s for s, _ in inside), max(e for _, e in inside)

    def speech_end_before(self, bot_seg: SpeechSegment) -> int | None:
        """When the caller last stopped speaking before this bot segment started, counting only
        speech that ended after the previous bot segment started. That is the moment pipecat's
        UserBotLatencyObserver measures from: it keeps the latest VAD stop and consumes it each
        time the bot starts speaking."""
        since_ms = self._previous_bot_start_ms(bot_seg)
        ends = [e for _, e in self.user_speech if since_ms < e <= bot_seg.start_ms]
        return max(ends) if ends else None

    def _previous_bot_start_ms(self, bot_seg: SpeechSegment) -> int:
        return max(
            (
                s.start_ms
                for s in self.speech_segments
                if s.speaker == "bot" and s.id != bot_seg.id and s.start_ms < bot_seg.start_ms
            ),
            default=0,
        )

    def on_llm_first_token(self, kind: str, observed_at_s: float, seconds: float) -> None:
        """Record one LLM inference's time to first token, as ServiceMetricsObserver reports it.
        A zero measurement, which pipecat can report before any inference ran, is skipped."""
        if self.call_start_abs_ns == 0 or seconds <= 0:
            return
        observed_ms = round((observed_at_s - self.call_start_abs_ns / 1_000_000_000) * 1000)
        self.llm_first_token.append((observed_ms, round(seconds * 1000), kind))

    def llm_first_token_before(self, bot_seg: SpeechSegment) -> int | None:
        """Time to first token of the last inference before this bot segment started, since
        the previous one: the inference whose answer it speaks. The first answer token wins
        over the first byte where pipecat reports it, so a model's thinking isn't counted."""
        since_ms = self._previous_bot_start_ms(bot_seg)
        inside = [r for r in self.llm_first_token if since_ms < r[0] <= bot_seg.start_ms]
        answer = [r for r in inside if r[2] == "ttfat"]
        return (answer or inside)[-1][1] if inside else None

    def set_turn_eou(self, strategy_name: str, stop_timestamp_ns: int) -> None:
        """Decide EOU for the just-ended user turn, driven by on_user_turn_stopped.

        This event is the one signal that natively knows BOTH that the turn ended and
        which strategy closed it — so the whole decision happens here, in one place, with
        no frame/event ordering race.

        The event can fire slightly before or after the UserStoppedSpeakingFrame commits
        the turn, so target whichever user turn is current: the still-open one if the
        event beat the commit, else the most recently appended user turn.

        - ExternalUserTurnStopStrategy => server-side STT (Flux) decided on its own clock;
          local VAD anchor is meaningless -> eou_reason only, no ms.
        - any other (local) strategy => VAD-anchored silence: last VAD stop -> this stop.
        - model_verdict (set earlier from TurnMetricsData) is left untouched.
        """
        # Consumed unconditionally, before the early-return below: every on_user_turn_stopped
        # event closes out whatever VAD anchor is pending, even on a target that turns out to
        # be already-decided (e.g. model_verdict) or absent. Otherwise a stale anchor from a
        # turn that never used it survives and can misfire onto a later turn that hasn't had
        # its own VAD stop yet, mislabeling it "silence_timeout" against the wrong anchor.
        last_vad_stop_ns = self._last_vad_stop_ns
        self._last_vad_stop_ns = None

        target = self._open_user_turn
        if target is None:
            for turn in reversed(self.live_turns):
                if turn.kind == "user":
                    target = turn
                    break
        if target is None or target.eou_ms is not None or target.eou_reason is not None:
            return  # nothing to target, or already decided (e.g. model_verdict)

        if strategy_name == "ExternalUserTurnStopStrategy":
            target.eou_reason = "stt_endpoint"
        elif last_vad_stop_ns is not None:
            eou_ms = self._rel_ms(stop_timestamp_ns) - self._rel_ms(last_vad_stop_ns)
            if eou_ms > 0:
                target.eou_ms = eou_ms
                target.eou_reason = "silence_timeout"
        # else: local strategy but no VAD anchor -> leave absent (honest).

    def on_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Capture user stop_ms and COMMIT the in-progress user turn.

        UserStoppedSpeaking is pipecat's user-message boundary: the aggregator commits the
        accumulated transcriptions as one message here. We mirror that — append the open user
        turn (if it captured any transcription) to the transcript at this moment, so the row
        grouping and ordering match pipecat exactly. A turn that captured nothing real is dropped.

        EOU is NOT decided here — it's driven by on_user_turn_stopped (see set_turn_eou),
        which is the one signal that natively knows which strategy closed the turn.
        """
        stopped_ms = self._rel_ms(timestamp_ns)
        seg = self._segment_by_id(self._active_user_segment_id)
        if seg is not None:
            seg.stop_ms = max(seg.stop_ms or 0, stopped_ms)
            # Close the open window (or open+close one if no start was recorded).
            if seg.windows and seg.windows[-1][1] is None:
                seg.windows[-1][1] = stopped_ms
            else:
                seg.windows.append([seg.start_ms, stopped_ms])

        if self._open_user_turn is not None:
            self._open_user_turn.end_ms = stopped_ms
            if self._open_user_turn.chunks:
                self._append_live(self._open_user_turn)
            # Kept even without a transcript yet: a late one still belongs to this turn.
            self._last_user_turn = self._open_user_turn
            self._open_user_turn = None

    def on_function_call_result(self, frame: Any, timestamp_ns: int) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        if tool_call_id:
            self.registry.record_completion_ns(tool_call_id, timestamp_ns)
        self._new_turn(
            "agent_result",
            tool_call_id=tool_call_id,
            function_name=getattr(frame, "function_name", None),
            result=getattr(frame, "result", None),
            occurrence_ms=self._rel_ms(timestamp_ns),
        )

    # How a tool call that didn't return is recorded on its result row.
    _UNSETTLED_RESULTS = {
        "function_call_timed_out": {"timed_out": True},
        "function_call_cancelled": {"cancelled": True},
    }

    def on_function_call_event(self, event: Any) -> None:
        """One moment of a tool call, as pipecat's FunctionCallObserver reports it: a call row
        when it goes in progress, and one result row when it settles. Timed at the moment
        itself, which the event carries."""
        kind = str(event.kind)
        timestamp_ns = round(event.timestamp * 1_000_000_000)
        call = SimpleNamespace(
            function_name=event.function_name,
            tool_call_id=event.tool_call_id,
            arguments=event.arguments,
            result=None,
        )
        if kind == "function_call_in_progress":
            self.on_function_call_in_progress(call, timestamp_ns)
            return
        if kind == "function_call_completed":
            call.result = event.result
        elif kind == "function_call_failed":
            call.result = {"error": event.error}
        elif kind in self._UNSETTLED_RESULTS:
            call.result = dict(self._UNSETTLED_RESULTS[kind])
        else:
            return  # started: the call row comes when it goes in progress
        self.on_function_call_result(call, timestamp_ns)

    def on_bot_stopped(self, timestamp_ns: int) -> None:
        """Record the end of a bot speech segment.

        Clears ``_current_bot_segment_id`` to signal that the bot is no longer
        speaking, but deliberately preserves ``_response_user_segment_id`` so that
        any subsequent bot segment within the same response (speak → tool → speak)
        still links to the same user segment via ``on_bot_started_speaking``.
        """
        if self.done:
            return
        bot_seg = self._segment_by_id(self._current_bot_segment_id)
        if bot_seg is not None:
            stop_ms = self._rel_ms(timestamp_ns)
            bot_seg.stop_ms = stop_ms
            # Claim TTS sentences voiced in this segment's window. Drop everything up to
            # this stop (claimed + any older un-voiced/abandoned text), keep later TTS.
            window_start = bot_seg.start_ms - _TTS_LEAD_SLACK_MS
            claimed = [t for t, ms in self._tts_timeline if window_start <= ms <= stop_ms]
            if claimed:
                bot_seg.spoken_text = " ".join(claimed).strip() or None
            self._tts_timeline = [(t, ms) for t, ms in self._tts_timeline if ms > stop_ms]
            self._current_bot_segment_id = None
            # NOTE: do NOT clear _open_agent_turn here — the assistant message commits slightly
            # after the bot stops (LLMContextAssistantTimestampFrame), and that commit is what
            # appends the turn. Clearing here would drop it.
        else:
            logger.warning("[tuner] bot_stopped with no active bot segment; ignoring event")

    def on_function_call_in_progress(self, frame: Any, timestamp_ns: int) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        if tool_call_id:
            self.registry.record_invocation_ns(tool_call_id, timestamp_ns)
        self._new_turn(
            "agent_function",
            tool_call_id=tool_call_id,
            function_name=getattr(frame, "function_name", None),
            arguments=getattr(frame, "arguments", None),
            occurrence_ms=self._rel_ms(timestamp_ns),
        )

    def on_bot_started_speaking(self, timestamp_ns: int) -> None:
        """Append a bot SpeechSegment and link a LatencyMeasurement to the user segment
        this response answers.

        The first bot segment of a response opens a LatencyMeasurement against
        ``_response_user_segment_id`` (or the proactive greeting when no user has
        spoken). Subsequent bot segments of the same response (speak → tool → speak)
        just append a segment and update ``_current_bot_segment_id`` — they do not open
        a new measurement. A fresh exchange is signalled by the next on_turn_started
        re-pointing ``_response_user_segment_id``.
        """
        started_ms = self._rel_ms(timestamp_ns)
        # The caller's turn ends where the bot's reply starts: what they said so far comes before
        # the reply, and anything they say after it starts a new turn. Done first, so the user
        # row is placed ahead of this voicing.
        self._last_user_turn = None
        if self._open_user_turn is not None and self._open_user_turn.chunks:
            self._open_user_turn.end_ms = self._open_user_turn.chunks[-1][1]
            self._append_live(self._open_user_turn)
            self._open_user_turn = None
        is_proactive = self._response_user_segment_id is None  # bot speaks before any user turn
        response_key = -1 if is_proactive else self._response_user_segment_id

        # Each bot speech start is its own segment with its own measurement, linked to the
        # user segment it answers (-1 for the proactive greeting). A new user turn re-points
        # _response_user_segment_id, so the next response naturally links to the new user
        # segment rather than folding into the previous one.
        seg = self._append_segment(
            "bot",
            start_ms=started_ms,
            is_proactive=is_proactive,
            turn_number=self._active_turn_number,
        )
        self._current_bot_segment_id = seg.id
        self._pending_measurement = LatencyMeasurement(
            user_segment_id=response_key, bot_segment_id=seg.id, is_proactive=is_proactive
        )
        self.latency_measurements.append(self._pending_measurement)

        # Bind this voicing to the in-progress (not-yet-committed) agent turn so the committed
        # row picks up the segment's final timing/latency. If the bot spoke with no tracked LLM
        # response (e.g. a static/templated greeting), open a turn here; its text falls back to
        # the voiced TTS at assembly. The turn is appended only when its commit frame arrives.
        if self._open_agent_turn is None:
            self._open_agent_turn = LiveTurn(kind="agent", order=0)
        if self._open_agent_turn.bot_segment_id is None:
            self._open_agent_turn.bot_segment_id = seg.id
            self._open_agent_turn.voiced_order = self._turn_order

    def on_latency_breakdown(self, breakdown: Any) -> None:
        measurement = self._pending_measurement
        if measurement is None:
            logger.warning(
                "[tuner] on_latency_breakdown fired with no pending measurement — skipping"
            )
            return
        self._pending_measurement = None  # consume immediately

        user_seg = self._segment_by_id(measurement.user_segment_id)
        bot_seg = self._segment_by_id(measurement.bot_segment_id)

        user_start_abs = getattr(breakdown, "user_turn_start_time", None)

        is_real_user_turn = True
        if user_start_abs is None:
            is_real_user_turn = False
            if not self._user_has_spoken:
                measurement.is_proactive = True
                if bot_seg is not None:
                    bot_seg.is_proactive = True
                # A ghost user segment opened by a pipeline-internal turn-start before the
                # greeting is not a real utterance — flag it so the enricher doesn't render
                # it as a user row or align it to a user message.
                if user_seg is not None:
                    user_seg.is_proactive = True
            # _user_has_spoken=True means this is a mid-conversation tool or node transition,
            # not a new user utterance. Leave bot start_ms as captured by on_bot_started_speaking.
        elif user_seg is not None:
            computed_started_ms = self._abs_to_rel_ms(user_start_abs)
            if computed_started_ms > 0:
                # Only write if valid — frame events already captured timing
                # correctly via on_turn_started, so don't overwrite with 0.
                user_seg.start_ms = computed_started_ms
            # computed_started_ms == 0: user spoke within the first milliseconds of the call.
            # Frame events already captured the correct timing via on_turn_started — skip.

        self._apply_queued_latency(
            bot_seg, user_seg, is_real_user_turn=is_real_user_turn, speech_timing=self.speech_timing
        )

        # Per-processor TTFB from the breakdown (each entry carries its processor name) so each
        # node's latency is attributed to the right node. NOTE: "GoogleSTTService".lower()
        # contains the substring "tts", so STT must be matched FIRST or it would leak into the
        # TTS bucket. STT TTFB is intentionally not surfaced here — STT latency lives on the
        # user row as stt_ms; this breakdown describes the bot response (LLM + TTS).
        first_ttfb_ms: int | None = None
        llm_ttfb_ms: int | None = None
        tts_ttfb_ms: int | None = None
        for ttfb in getattr(breakdown, "ttfb", []) or []:
            candidate = int((getattr(ttfb, "duration_secs", 0) or 0) * 1000)
            if candidate <= 0:
                continue
            if first_ttfb_ms is None:
                first_ttfb_ms = candidate
            proc = str(getattr(ttfb, "processor", "")).lower()
            if "stt" in proc:
                continue
            if "llm" in proc and llm_ttfb_ms is None:
                llm_ttfb_ms = candidate
            elif "tts" in proc and tts_ttfb_ms is None:
                tts_ttfb_ms = candidate
        # tts_node_ttfb: the TTS service's own TTFB. Fall back to the first positive TTFB only
        # when no processor was identifiable (e.g. unlabeled metrics in tests) — real pipelines
        # always label TTS, so STT/LLM TTFBs no longer leak into this field.
        measurement.ttfb_ms = tts_ttfb_ms if tts_ttfb_ms is not None else first_ttfb_ms

        # LLM latency. Prefer an externally-sourced call duration when this turn went
        # through observer.wrap_graph()/wrap_chain() — pipecat's LangchainProcessor
        # is a bare FrameProcessor, not an LLMService, so it never emits the
        # MetricsFrame data the other sources depend on; this is the only
        # number that ever exists for those turns. Note this is total on_llm_start->
        # on_llm_end duration, not strictly time-to-first-token. Otherwise use
        # ServiceMetricsObserver's time to first token, which is what llm_node_ttft names,
        # and without it (pipecat < 1.9) the processing time of the turn's inferences.
        first_token_ms = (
            self.llm_first_token_before(bot_seg)
            if self.service_metrics and bot_seg is not None
            else None
        )
        if self._pending_external_llm_duration_ms is not None:
            measurement.llm_ms = self._pending_external_llm_duration_ms
        elif first_token_ms is not None:
            measurement.llm_ms = first_token_ms
        else:
            measurement.llm_ms = self._processing_llm_ms(
                self._pending_pipecat_llm_processing_s, llm_ttfb_ms
            )
        measurement.tts_ms = (
            round(self._pending_pipecat_tts_processing_s * 1000)
            if self._pending_pipecat_tts_processing_s
            else None
        )
        measurement.interrupted_at_ms = bot_seg.interrupted_at_ms if bot_seg else None
        if (
            not measurement.is_proactive
            and bot_seg is not None
            and user_seg is not None
            and bot_seg.start_ms > 0
            and (user_seg.stop_ms or 0) > 0
        ):
            # What the caller waited: from the end of their speech to the bot's first audio.
            # Without a speech stretch (pipecat < 1.9), the end is UserStoppedSpeakingFrame.
            speech_end_ms = self.speech_end_before(bot_seg) if self.speech_timing else None
            measurement.e2e_ms = bot_seg.start_ms - (
                speech_end_ms if speech_end_ms is not None else user_seg.stop_ms
            )

        self._pending_pipecat_llm_processing_s = 0.0
        self._pending_pipecat_tts_processing_s = 0.0
        self._pending_external_llm_duration_ms = None

    def on_call_end(self, timestamp_ns: int) -> None:
        if self.done:
            return
        self.done = True
        self.call_end_abs_ns = timestamp_ns

        # If the user was still speaking when the call ended, anchor their
        # stop time to the call end so the last segment gets a valid timestamp.
        seg = self._segment_by_id(self._active_user_segment_id)
        if seg is not None and seg.start_ms > 0 and not seg.stop_ms:
            seg.stop_ms = self._rel_ms(timestamp_ns)

        # Commit any user turn still in progress (call cut off before UserStoppedSpeaking).
        if self._open_user_turn is not None:
            if self._open_user_turn.chunks:
                self._open_user_turn.end_ms = self._rel_ms(timestamp_ns)
                self._append_live(self._open_user_turn)
            self._open_user_turn = None

        # Commit the final assistant turn if the bot was still speaking when the call ended
        # (pipecat commits it on the closing Cancel/EndFrame). A turn that produced neither text
        # nor a bot segment is dropped.
        turn = self._open_agent_turn
        if turn is not None and (turn.text.strip() or turn.bot_segment_id is not None):
            bot_seg = self._segment_by_id(turn.bot_segment_id)
            if bot_seg is not None and not bot_seg.stop_ms and bot_seg.start_ms > 0:
                bot_seg.stop_ms = self._rel_ms(timestamp_ns)
            self._append_live(turn)
        self._open_agent_turn = None

    def on_metrics_frame(self, frame: Any) -> None:
        for d in getattr(frame, "data", []):
            cls_name = type(d).__name__
            if cls_name == "LLMUsageMetricsData":
                token_usage = getattr(d, "value", None)
                total_tokens = getattr(token_usage, "total_tokens", 0) or 0
                prompt_tokens = getattr(token_usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(token_usage, "completion_tokens", 0) or 0
                self._pipecat_llm_total_tokens += total_tokens
                self._pipecat_llm_prompt_tokens += prompt_tokens
                self._pipecat_llm_completion_tokens += completion_tokens
            elif cls_name == "TTSUsageMetricsData":
                self._pipecat_tts_chars += getattr(d, "value", 0) or 0
            elif cls_name == "ProcessingMetricsData":
                processor = str(getattr(d, "processor", "")).lower()
                val = getattr(d, "value", 0) or 0
                if "tts" in processor:
                    # Assignment (not +=): only one TTS job runs per turn, so the latest
                    # value is always the correct one. Multiple LLM steps can fire in a
                    # single turn (e.g. parallel tool calls), so LLM uses accumulation.
                    self._pending_pipecat_tts_processing_s = val
                else:
                    self._pending_pipecat_llm_processing_s += val
            elif cls_name == "TTFBMetricsData":
                # The STT service's time-to-first-byte is the user's pure STT latency.
                # Record it on the active user segment (value is in seconds). First
                # transcription of the turn wins, so a coalesced multi-utterance turn keeps
                # the latency of its first utterance. (Only "stt" here — the LLM/TTS TTFBs are
                # attributed to the bot response in on_latency_breakdown.)
                processor = str(getattr(d, "processor", "")).lower()
                if "stt" in processor:
                    val = getattr(d, "value", 0) or 0
                    seg = self._segment_by_id(self._active_user_segment_id)
                    if seg is not None and val > 0 and seg.stt_ms is None:
                        seg.stt_ms = round(val * 1000)
                    # Also stamp the in-progress user turn (the first STT TTFB of the turn wins).
                    # Like a late transcript, the metric can land just after its turn closed,
                    # before the bot replies; it is still that turn's.
                    turn = self._open_user_turn or self._last_user_turn
                    if turn is not None and val > 0 and turn.stt_ms is None:
                        turn.stt_ms = round(val * 1000)

            elif cls_name == "TurnMetricsData":
                # Only an is_complete=True verdict is a real end-of-turn decision — the framework
                # fires several is_complete=False data points per coalesced turn (VAD paused, model
                # said "not done yet"); those are diagnostic-only for now and not attached anywhere.
                # e2e_processing_time_ms is already in ms (unlike the *_secs values above).
                if getattr(d, "is_complete", False):
                    eou_ms = round(getattr(d, "e2e_processing_time_ms", 0) or 0)
                    confidence = getattr(d, "probability", None)
                    # Guard on eou_reason too, not just eou_ms: set_turn_eou can decide a
                    # still-open turn (e.g. stt_endpoint, which leaves eou_ms unset) before this
                    # metric arrives. TurnMetricsData carries no turn id, so a verdict that's
                    # late enough to land after the open turn already moved on to the next one
                    # can't be routed to its real turn — checking eou_reason at least stops it
                    # from clobbering an already-decided turn's verdict.
                    if (
                        self._open_user_turn is not None
                        and self._open_user_turn.eou_ms is None
                        and self._open_user_turn.eou_reason is None
                    ):
                        self._open_user_turn.eou_ms = eou_ms
                        self._open_user_turn.eou_confidence = confidence
                        self._open_user_turn.eou_reason = "model_verdict"

    def build_payload(
        self,
        config: Any,
        cost_calculator: Callable[[CallUsage], float] | None = None,
    ) -> CallPayload:
        return build_payload(self, config, cost_calculator, self._segment_enrichers)

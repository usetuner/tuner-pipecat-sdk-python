"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .models import CallPayload, CallUsage, LatencyMeasurement, LiveTurn, SpeechSegment
from .payload_builder import build_payload
from .tool_timing_registry import ToolTimingRegistry

# TTSTextFrames are emitted slightly before BotStartedSpeakingFrame. A bot segment claims
# voiced TTS sentences from up to this many ms before its start. Wide enough to cover the
# TTS lead, far below the multi-second gap to an interrupted-before-voiced response.
_TTS_LEAD_SLACK_MS = 2000


@dataclass
class CallAccumulator:
    """Collects runtime events and produces a final call payload."""

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

    # Ordered pairing for latency observer callbacks (on_latency_measured then breakdown).
    _pending_latency_ms_queue: deque[int] = field(default_factory=deque, repr=False)

    # tool call timing keyed by tool_call_id
    registry: ToolTimingRegistry = field(default_factory=ToolTimingRegistry)

    # call-level pipecat-sourced counters (summed across all MetricsFrames)
    _pipecat_llm_total_tokens: int = field(default=0, repr=False)
    _pipecat_llm_prompt_tokens: int = field(default=0, repr=False)
    _pipecat_llm_completion_tokens: int = field(default=0, repr=False)
    _pipecat_tts_chars: int = field(default=0, repr=False)

    # per-turn pending pipecat metrics (reset on each latency breakdown)
    _pending_pipecat_llm_processing_s: float = field(default=0.0, repr=False)
    _pending_pipecat_tts_processing_s: float = field(default=0.0, repr=False)

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
    # The agent turn currently accruing LLM text / awaiting voicing.
    _open_agent_turn: LiveTurn | None = field(default=None, repr=False)

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
        return self._pipecat_llm_total_tokens

    def get_llm_prompt_tokens(self) -> int:
        return self._pipecat_llm_prompt_tokens

    def get_llm_completion_tokens(self) -> int:
        return self._pipecat_llm_completion_tokens

    def get_total_tts_characters(self) -> int:
        return self._pipecat_tts_chars

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
                kind="user", order=0, start_ms=start_ms,
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
        committed, so we mirror that by simply not appending it.
        """
        if self._llm_response_buffer:
            self._finalize_llm_response(0)
        self._llm_response_buffer = []
        self._open_agent_turn = LiveTurn(
            kind="agent", order=0, generated_ms=self._rel_ms(timestamp_ns)
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

    def on_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Capture user stop_ms and COMMIT the in-progress user turn.

        UserStoppedSpeaking is pipecat's user-message boundary: the aggregator commits the
        accumulated transcriptions as one message here. We mirror that — append the open user
        turn (if it captured any transcription) to the transcript at this moment, so the row
        grouping and ordering match pipecat exactly. A turn that captured nothing real is dropped.
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
            if self._open_user_turn.chunks:
                self._open_user_turn.end_ms = stopped_ms
                self._append_live(self._open_user_turn)
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

    def on_latency_measured(self, latency_secs: float) -> None:
        self._pending_latency_ms_queue.append(max(0, int(latency_secs * 1000)))

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

        if self._pending_latency_ms_queue:
            latency_ms = self._pending_latency_ms_queue.popleft()
            if (
                is_real_user_turn
                and bot_seg is not None
                and user_seg is not None
                and not bot_seg.interrupted_at_ms
                and (user_seg.stop_ms or 0) > 0
            ):
                bot_seg.start_ms = user_seg.stop_ms + latency_ms

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

        # LLM time-to-first-token. Prefer the processing-time metric when present, but fall
        # back to the LLM's own TTFB — some providers (e.g. Google/Gemini) emit a TTFB metric
        # but no ProcessingMetricsData, which would otherwise leave LLM latency blank.
        measurement.llm_ms = (
            round(self._pending_pipecat_llm_processing_s * 1000)
            if self._pending_pipecat_llm_processing_s
            else llm_ttfb_ms
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
            measurement.e2e_ms = bot_seg.start_ms - user_seg.stop_ms

        self._pending_pipecat_llm_processing_s = 0.0
        self._pending_pipecat_tts_processing_s = 0.0

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
                    if self._open_user_turn is not None and val > 0 and (
                        self._open_user_turn.stt_ms is None
                    ):
                        self._open_user_turn.stt_ms = round(val * 1000)

    def build_payload(
        self,
        config: Any,
        cost_calculator: Callable[[CallUsage], float] | None = None,
    ) -> CallPayload:
        return build_payload(self, config, cost_calculator)

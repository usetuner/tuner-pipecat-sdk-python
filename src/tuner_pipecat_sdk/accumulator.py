"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .models import CallPayload, CallUsage, LatencyMeasurement, SpeechSegment
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

    # vad_stopped_ns keyed by user segment id — internal timing state used to
    # compute stt_ms (turn-stop − vad-stop) in on_user_turn_stopped.
    _vad_stopped_ns_by_user_segment_id: dict[int, int] = field(default_factory=dict, repr=False)

    # Set to True on the first UserStartedSpeakingFrame — used to distinguish
    # the proactive bot greeting from mid-conversation tool or node transitions.
    _user_has_spoken: bool = field(default=False, repr=False)

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

    def on_user_started_speaking(self, timestamp_ns: int) -> None:
        """Use frame timestamp as the authoritative user-start anchor for the active segment."""
        self._user_has_spoken = True
        # If bot is currently speaking, this is an interruption.
        # Record when the user started cutting in on the BOT's current segment.
        bot_seg = self._segment_by_id(self._current_bot_segment_id)
        if bot_seg is not None:
            bot_seg.interrupted_at_ms = self._rel_ms(timestamp_ns)

        seg = self._segment_by_id(self._active_user_segment_id)
        if seg is None:
            return
        started_ms = self._rel_ms(timestamp_ns)
        if seg.start_ms == 0:
            seg.start_ms = started_ms
        else:
            seg.start_ms = min(seg.start_ms, started_ms)
        # Open a new per-utterance window. A coalesced turn (multiple utterances before any
        # bot reply) accrues several windows, preserving per-utterance timing.
        seg.windows.append([started_ms, None])

    def on_tts_text(self, text: str, timestamp_ns: int) -> None:
        """Record a voiced TTS sentence on the timeline. Assigned to a bot segment by time
        window at on_bot_stopped (TTSTextFrames arrive before BotStartedSpeakingFrame)."""
        if text and text.strip():
            self._tts_timeline.append((text, self._rel_ms(timestamp_ns)))

    def on_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Capture user stop_ms directly from VAD frame.

        Used as the primary source for interrupted turns where on_latency_breakdown
        receives user_turn_start_time=None and cannot compute the stop time.
        """
        seg = self._segment_by_id(self._active_user_segment_id)
        if seg is not None:
            stopped_ms = self._rel_ms(timestamp_ns)
            seg.stop_ms = max(seg.stop_ms or 0, stopped_ms)
            # Close the open window (or open+close one if no start was recorded).
            if seg.windows and seg.windows[-1][1] is None:
                seg.windows[-1][1] = stopped_ms
            else:
                seg.windows.append([seg.start_ms, stopped_ms])

    def on_function_call_result(self, tool_call_id: str, timestamp_ns: int) -> None:
        self.registry.record_completion_ns(tool_call_id, timestamp_ns)

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
        else:
            logger.warning("[tuner] bot_stopped with no active bot segment; ignoring event")

    def on_function_call_in_progress(self, frame: Any, timestamp_ns: int) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        if tool_call_id:
            self.registry.record_invocation_ns(tool_call_id, timestamp_ns)

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

    def on_vad_stopped(self, timestamp_ns: int) -> None:
        if self._active_user_segment_id is None:
            # VADUserStoppedSpeakingFrame can legitimately fire before any turn
            # starts (background noise, room ambience at call start) — debug only.
            logger.debug("[tuner] on_vad_stopped: no active user segment")
            return
        self._vad_stopped_ns_by_user_segment_id[self._active_user_segment_id] = timestamp_ns

    def on_user_turn_stopped(self, timestamp_ns: int) -> None:
        seg = self._segment_by_id(self._active_user_segment_id)
        if seg is None:
            logger.warning("[tuner] on_user_turn_stopped: no active user segment")
            return
        vad_stopped_ns = self._vad_stopped_ns_by_user_segment_id.get(seg.id)
        if vad_stopped_ns is None:
            logger.warning(
                "[tuner] on_user_turn_stopped: vad_stopped_ns not set on segment {}", seg.id
            )
            return
        # stt_ms is a property of the user's own utterance — store it on the user
        # SpeechSegment so it survives even when no bot response ever follows.
        gap_ms = (timestamp_ns - vad_stopped_ns) // 1_000_000
        seg.stt_ms = max(0, gap_ms)

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

        ttfb_ms: int | None = None
        for ttfb in getattr(breakdown, "ttfb", []) or []:
            candidate = int((getattr(ttfb, "duration_secs", 0) or 0) * 1000)
            if candidate > 0:
                ttfb_ms = candidate
                break
        measurement.ttfb_ms = ttfb_ms

        measurement.llm_ms = (
            round(self._pending_pipecat_llm_processing_s * 1000)
            if self._pending_pipecat_llm_processing_s
            else None
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

    def build_payload(
        self,
        config: Any,
        transcript: list[dict[str, Any]],
        cost_calculator: Callable[[CallUsage], float] | None = None,
    ) -> CallPayload:
        return build_payload(self, config, transcript, cost_calculator)

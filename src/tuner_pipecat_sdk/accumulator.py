"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .models import CallPayload, LatencyTurn
from .payload_builder import build_payload
from .tool_timing_registry import ToolTimingRegistry


@dataclass
class CallAccumulator:
    """Collects runtime events and produces a final call payload."""

    # call-level timing
    call_start_abs_ns: int = 0
    call_end_abs_ns: int = 0

    # latency tracking
    latency_turns: list[LatencyTurn] = field(default_factory=list)

    # Active turn tracking
    _active_turn_number: int | None = field(default=None, repr=False)
    _turn_to_latency_idx: dict[int, int] = field(default_factory=dict, repr=False)
    # Turn index waiting for first bot response anchor (bot-start).
    _open_latency_idx: int | None = field(default=None, repr=False)
    # Turn index currently speaking; consumed by on_bot_stopped.
    _bot_turn_idx: int | None = field(default=None, repr=False)
    # Stable anchor for the user turn the bot is currently responding to.
    # Persists across multiple bot speech segments within the same user turn —
    # for example when the bot speaks, executes a tool, then speaks again.
    # Distinct from _bot_turn_idx which tracks only whether the bot is currently
    # speaking and is cleared on BotStoppedSpeakingFrame.
    _current_user_turn_latency_idx: int | None = field(default=None, repr=False)

    # Ordered pairing for latency observer callbacks (on_latency_measured then breakdown).
    _pending_latency_ms_queue: deque[int] = field(default_factory=deque, repr=False)

    # tool call timing keyed by tool_call_id
    registry: ToolTimingRegistry = field(default_factory=ToolTimingRegistry)

    # call-level pipecat-sourced counters (summed across all MetricsFrames)
    _pipecat_llm_total_tokens: int = field(default=0, repr=False)
    _pipecat_tts_chars: int = field(default=0, repr=False)

    # per-turn pending pipecat metrics (reset on each latency breakdown)
    _pending_pipecat_llm_processing_s: float = field(default=0.0, repr=False)
    _pending_pipecat_tts_processing_s: float = field(default=0.0, repr=False)

    # Turn-started calls that arrived before StartFrame (call_start_abs_ns not yet set).
    # Processed retroactively in on_start once the reference timestamp is known.
    _pending_turn_starts: list[tuple[int, int]] = field(default_factory=list, repr=False)

    # Stable turn index for on_latency_breakdown, set when bot starts speaking.
    # Decoupled from _active_turn_number so interruptions don't corrupt the target.
    _pending_breakdown_latency_idx: int | None = field(default=None, repr=False)

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

    def get_tool_invocation_ms(self, tool_call_id: str) -> int | None:
        abs_ns = self.registry.get_invocation_ns(tool_call_id)
        return self._rel_ms(abs_ns) if abs_ns else None

    def get_tool_completion_ms(self, tool_call_id: str) -> int | None:
        abs_ns = self.registry.get_completion_ns(tool_call_id)
        return self._rel_ms(abs_ns) if abs_ns else None

    def get_total_llm_tokens(self) -> int:
        return self._pipecat_llm_total_tokens

    def get_total_tts_characters(self) -> int:
        return self._pipecat_tts_chars

    @property
    def disconnection_reason(self) -> str:
        return self._disconnection_reason

    def set_disconnection_reason(self, reason: str) -> None:
        """Write-once: first meaningful value wins, subsequent calls are no-ops.

        CancelFrame sets this to ERROR automatically via _base_observer._handle().
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
        """Open or extend a LatencyTurn for an incoming user speech segment.

        A new LatencyTurn is created only when the bot has already started
        responding to the previous turn (``bot_started_ms > 0``).  Until that
        point every new user speech segment — whether it is a mid-sentence
        pause continuation — is collapsed into the existing turn.

        Interruptions (user speaks while bot is speaking) correctly open a new
        turn because ``bot_started_ms > 0`` at that point.

        This ensures that a single logical user→bot exchange always maps to
        exactly one LatencyTurn regardless of how many VAD/turn-detection
        events fire during that exchange, which in turn keeps tool call
        timestamps, latency breakdowns, and bot speech attribution consistent.
        """
        if self.call_start_abs_ns == 0:
            self._pending_turn_starts.append((turn_number, timestamp_ns))
            return

        started_ms = self._rel_ms(timestamp_ns)

        # ── Collapse guard ────────────────────────────────────────────────────
        # Only collapse into the existing turn when the bot has not yet started
        # speaking (bot_started_ms == 0). This covers mid-sentence pauses where
        # the user briefly stops and then continues before the bot responds.
        # Interruptions are excluded because bot_started_ms > 0 by that point,
        # and they correctly open a new LatencyTurn.
        if self._current_user_turn_latency_idx is not None:
            idx = self._current_user_turn_latency_idx
            if idx < len(self.latency_turns):
                current = self.latency_turns[idx]
                if current.bot_started_ms == 0 and not current.llm_completed:
                    if current.user_started_ms == 0:
                        current.user_started_ms = started_ms
                    else:
                        current.user_started_ms = min(current.user_started_ms, started_ms)
                    self._turn_to_latency_idx[turn_number] = idx
                    self._active_turn_number = turn_number
                    # Keep _open_latency_idx pointed at this turn if it was
                    # already cleared — restoring it ensures on_bot_started_speaking
                    # can still find its primary anchor.
                    if self._open_latency_idx is None:
                        self._open_latency_idx = idx
                    return

        # ── New turn ──────────────────────────────────────────────────────────
        # The bot has already started or finished responding, or there is no
        # previous turn.  Open a fresh LatencyTurn.
        new_idx = len(self.latency_turns)
        self._open_latency_idx = new_idx
        self._current_user_turn_latency_idx = new_idx
        self._turn_to_latency_idx[turn_number] = new_idx
        self._active_turn_number = turn_number
        self.latency_turns.append(
            LatencyTurn(
                turn_index=new_idx,
                user_started_ms=started_ms,
            )
        )

    def on_turn_ended(self, turn_number: int, was_interrupted: bool) -> None:
        """Called by TurnTrackingObserver when a turn ends (bot finished or interrupted)."""
        idx = self._turn_to_latency_idx.get(turn_number)
        if idx is not None and idx < len(self.latency_turns):
            self.latency_turns[idx].was_interrupted = was_interrupted
        self._active_turn_number = None

    def on_user_started_speaking(self, timestamp_ns: int) -> None:
        """Use frame timestamp as the authoritative user-start anchor for the active turn."""
        self._user_has_spoken = True
        # If bot is currently speaking, this is an interruption.
        # Record when the user started cutting in on the BOT's current turn.
        if self._bot_turn_idx is not None and self._bot_turn_idx < len(self.latency_turns):
            self.latency_turns[self._bot_turn_idx].interrupted_at_ms = self._rel_ms(timestamp_ns)

        if self._active_turn_number is None:
            return
        idx = self._turn_to_latency_idx.get(self._active_turn_number)
        if idx is None or idx >= len(self.latency_turns):
            return
        started_ms = self._rel_ms(timestamp_ns)
        turn = self.latency_turns[idx]
        if turn.user_started_ms == 0:
            turn.user_started_ms = started_ms
        else:
            turn.user_started_ms = min(turn.user_started_ms, started_ms)

    def on_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Capture user_stopped_ms directly from VAD frame.

        Used as the primary source for interrupted turns where on_latency_breakdown
        receives user_turn_start_time=None and cannot compute user_stopped_ms.
        on_latency_breakdown overrides this with its computed value when available.
        """
        if self._active_turn_number is None:
            return
        idx = self._turn_to_latency_idx.get(self._active_turn_number)
        if idx is not None and idx < len(self.latency_turns):
            stopped_ms = self._rel_ms(timestamp_ns)
            turn = self.latency_turns[idx]
            turn.user_stopped_ms = max(turn.user_stopped_ms, stopped_ms)

    def on_function_call_result(self, tool_call_id: str, timestamp_ns: int) -> None:
        self.registry.record_completion_ns(tool_call_id, timestamp_ns)

    def on_bot_stopped(self, timestamp_ns: int) -> None:
        """Record the end of a bot speech segment.

        Clears ``_bot_turn_idx`` to signal that the bot is no longer speaking,
        but deliberately preserves ``_current_user_turn_latency_idx`` so that
        any subsequent speech segment within the same user turn can still locate
        its ``LatencyTurn`` via ``on_bot_started_speaking``.
        """
        if self.done:
            return
        bot_stopped_ms = self._rel_ms(timestamp_ns)
        if self._bot_turn_idx is not None and self._bot_turn_idx < len(self.latency_turns):
            self.latency_turns[self._bot_turn_idx].bot_stopped_ms = bot_stopped_ms
            self._bot_turn_idx = None
        else:
            logger.warning("[tuner] bot_stopped with no active bot turn index; ignoring event")

    def on_function_call_in_progress(self, frame: Any, timestamp_ns: int) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        if tool_call_id:
            self.registry.record_invocation_ns(tool_call_id, timestamp_ns)

    def on_bot_started_speaking(self, timestamp_ns: int) -> None:
        """Record the start of a bot speech segment and bind it to the active user turn.

        Resolution order for the target LatencyTurn:

        1. _open_latency_idx — set when a user turn opens, consumed by the first bot
        speech. Handles the common case and the initial speech of multi-speech turns.

        2. _current_user_turn_latency_idx — stable anchor for the lifetime of the user
        turn. Used for every subsequent speech segment after _open_latency_idx is
        consumed.

        If the resolved turn already has a complete bot response (bot_stopped_ms > 0),
        a new LatencyTurn is created for the fresh exchange.

        bot_started_ms is written only once so it always reflects when the bot first
        began speaking in response to the user, regardless of how many segments follow.
        """
        if self._open_latency_idx is not None and self._open_latency_idx < len(self.latency_turns):
            idx = self._open_latency_idx
            self._open_latency_idx = None
        elif (
            self._current_user_turn_latency_idx is not None
            and self._current_user_turn_latency_idx < len(self.latency_turns)
        ):
            idx = self._current_user_turn_latency_idx
        elif not self.latency_turns:
            # Safety net: if no user turn exists yet, the bot is greeting proactively.
            # In practice on_turn_started always fires first via pipeline internals,
            # but if that changes this ensures the greeting is still captured correctly.
            turn = LatencyTurn(turn_index=0, is_proactive=True)
            turn.bot_started_ms = self._rel_ms(timestamp_ns)
            self.latency_turns.append(turn)
            self._bot_turn_idx = 0
            self._pending_breakdown_latency_idx = 0
            return
        else:
            return

        # If the resolved turn already has a complete bot response, this
        # new speech is a fresh exchange — create a new LatencyTurn for it.
        if self.latency_turns[idx].bot_stopped_ms is not None:
            new_idx = len(self.latency_turns)
            self.latency_turns.append(LatencyTurn(turn_index=new_idx))
            self._current_user_turn_latency_idx = new_idx
            idx = new_idx

        turn = self.latency_turns[idx]
        if turn.bot_started_ms == 0:
            turn.bot_started_ms = self._rel_ms(timestamp_ns)
        self._bot_turn_idx = idx
        self._pending_breakdown_latency_idx = idx

    def on_vad_stopped(self, timestamp_ns: int) -> None:
        if self._active_turn_number is None:
            logger.warning("[tuner] on_vad_stopped: no active turn")
            return
        idx = self._turn_to_latency_idx.get(self._active_turn_number)
        if idx is None or idx >= len(self.latency_turns):
            logger.warning("[tuner] on_vad_stopped: active turn not in latency_turns")
            return
        self.latency_turns[idx].vad_stopped_ns = timestamp_ns

    def on_user_turn_stopped(self, timestamp_ns: int) -> None:
        if self._active_turn_number is None:
            logger.warning("[tuner] on_user_turn_stopped: no active turn")
            return
        idx = self._turn_to_latency_idx.get(self._active_turn_number)
        if idx is None or idx >= len(self.latency_turns):
            logger.warning("[tuner] on_user_turn_stopped: active turn not in latency_turns")
            return
        turn = self.latency_turns[idx]
        if turn.vad_stopped_ns is None:
            logger.warning("[tuner] on_user_turn_stopped: vad_stopped_ns not set on turn {}", idx)
            return
        gap_ms = (timestamp_ns - turn.vad_stopped_ns) // 1_000_000
        turn.stt_ms = max(0, gap_ms)

    def on_latency_measured(self, latency_secs: float) -> None:
        self._pending_latency_ms_queue.append(max(0, int(latency_secs * 1000)))

    def on_latency_breakdown(self, breakdown: Any) -> None:
        if self._pending_breakdown_latency_idx is None:
            logger.warning(
                "[tuner] on_latency_breakdown fired with no pending breakdown idx — skipping"
            )
            return
        idx = self._pending_breakdown_latency_idx

        self._pending_breakdown_latency_idx = None  # consume immediately

        if idx >= len(self.latency_turns):
            logger.warning("[tuner] on_latency_breakdown: idx out of range — skipping")
            return

        turn = self.latency_turns[idx]

        user_start_abs = getattr(breakdown, "user_turn_start_time", None)

        is_real_user_turn = True
        if user_start_abs is None:
            is_real_user_turn = False
            if not self._user_has_spoken:
                turn.is_proactive = True
            # _user_has_spoken=True means this is a mid-conversation tool or node transition,
            # not a new user utterance. Leave bot_started_ms as captured by on_bot_started_speaking.
        else:
            computed_started_ms = self._abs_to_rel_ms(user_start_abs)
            if computed_started_ms > 0:
                # Only write if valid — frame events already captured timing
                # correctly via on_turn_started, so don't overwrite with 0.
                turn.user_started_ms = computed_started_ms
            # computed_started_ms == 0: user spoke within the first milliseconds of the call.
            # Frame events already captured the correct timing via on_turn_started — skip.

        if self._pending_latency_ms_queue:
            latency_ms = self._pending_latency_ms_queue.popleft()
            if is_real_user_turn and not turn.interrupted_at_ms:
                turn.bot_started_ms = turn.user_stopped_ms + latency_ms

        ttfb_ms: int | None = None
        for ttfb in getattr(breakdown, "ttfb", []) or []:
            candidate = int((getattr(ttfb, "duration_secs", 0) or 0) * 1000)
            if candidate > 0:
                ttfb_ms = candidate
                break
        turn.ttfb_ms = ttfb_ms

        turn.llm_ms = (
            round(self._pending_pipecat_llm_processing_s * 1000)
            if self._pending_pipecat_llm_processing_s
            else None
        )
        turn.tts_ms = (
            round(self._pending_pipecat_tts_processing_s * 1000)
            if self._pending_pipecat_tts_processing_s
            else None
        )

        self._pending_pipecat_llm_processing_s = 0.0
        self._pending_pipecat_tts_processing_s = 0.0
        turn.llm_completed = True

    def on_call_end(self, timestamp_ns: int) -> None:
        if self.done:
            return
        self.done = True
        self.call_end_abs_ns = timestamp_ns

        # If the user was still speaking when the call ended, anchor their
        # stop time to the call end so the last segment gets a valid timestamp.
        if self._active_turn_number is not None:
            idx = self._turn_to_latency_idx.get(self._active_turn_number)
            if idx is not None and idx < len(self.latency_turns):
                turn = self.latency_turns[idx]
                if turn.user_started_ms > 0 and turn.user_stopped_ms == 0:
                    turn.user_stopped_ms = self._rel_ms(timestamp_ns)

        # BotStartedSpeakingFrame can arrive after BotStoppedSpeakingFrame during
        # pipeline shutdown, causing bot_started_ms > bot_stopped_ms on the final
        # bot turn. Fix the inversion on the last turn only.
        if self.latency_turns:
            last = self.latency_turns[-1]
            if (
                last.bot_stopped_ms is not None
                and last.bot_started_ms > 0
                and last.bot_started_ms > last.bot_stopped_ms
            ):
                last.bot_started_ms = last.bot_stopped_ms

    def on_metrics_frame(self, frame: Any) -> None:
        for d in getattr(frame, "data", []):
            cls_name = type(d).__name__
            if cls_name == "LLMUsageMetricsData":
                total_tokens = getattr(getattr(d, "value", None), "total_tokens", 0) or 0
                self._pipecat_llm_total_tokens += total_tokens
            elif cls_name == "TTSUsageMetricsData":
                self._pipecat_tts_chars += getattr(d, "value", 0) or 0
            elif cls_name == "ProcessingMetricsData":
                processor = str(getattr(d, "processor", "")).lower()
                val = getattr(d, "value", 0) or 0
                if "tts" in processor:
                    self._pending_pipecat_tts_processing_s = val
                else:
                    self._pending_pipecat_llm_processing_s += val
                    # LLM finished processing — mark current turn as llm_completed
                    # so the collapse guard knows TTS was supposed to fire,
                    # preventing user follow-up messages from collapsing into
                    # the failed turn when bot_started_ms stays 0.
                    if self._current_user_turn_latency_idx is not None:
                        idx = self._current_user_turn_latency_idx
                        if idx < len(self.latency_turns):
                            self.latency_turns[idx].llm_completed = True

    def build_payload(self, config: Any, transcript: list[dict[str, Any]]) -> CallPayload:
        return build_payload(self, config, transcript)

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

    # vad_stopped_ns keyed by turn index — internal timing state kept off the
    # public LatencyTurn model so it doesn't appear in model_dump() output.
    _vad_stopped_ns_by_turn: dict[int, int] = field(default_factory=dict, repr=False)

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
        """Register the TurnTrackingObserver turn_number for the active LatencyTurn.

        This method is called from an asyncio task scheduled by TurnTrackingObserver
        and therefore runs *after* the corresponding UserStartedSpeakingFrame has
        already been processed inline by on_user_started_speaking (which is the
        primary turn-creation path).  The job here is to register the
        turn_number → latency_idx mapping so that on_turn_ended can find the
        turn, and to set _active_turn_number for the stopped-speaking helpers.

        For the very first turn (proactive greeting) there is no preceding
        UserStartedSpeakingFrame, so this method also handles the fallback of
        creating the LatencyTurn when none exists yet.
        """
        if self.call_start_abs_ns == 0:
            self._pending_turn_starts.append((turn_number, timestamp_ns))
            return

        started_ms = self._rel_ms(timestamp_ns)

        if self._current_user_turn_latency_idx is not None:
            idx = self._current_user_turn_latency_idx
            if idx < len(self.latency_turns):
                current = self.latency_turns[idx]

                # ── Proactive turn (safety-net or normal path) ────────────────
                # The proactive greeting has is_proactive=True either from the
                # safety net or retroactively from on_latency_breakdown.  Either
                # way, don't create a new turn — just register the mapping.
                if current.is_proactive:
                    self._turn_to_latency_idx[turn_number] = idx
                    self._active_turn_number = turn_number
                    return

                # ── Open user turn ────────────────────────────────────────────
                # on_user_started_speaking already created this turn inline.
                # Register the mapping and optionally sharpen the timestamp.
                if current.bot_started_ms == 0 and not current.llm_completed:
                    if current.user_started_ms == 0:
                        current.user_started_ms = started_ms
                    else:
                        current.user_started_ms = min(current.user_started_ms, started_ms)
                    self._turn_to_latency_idx[turn_number] = idx
                    self._active_turn_number = turn_number
                    if self._open_latency_idx is None:
                        self._open_latency_idx = idx
                    return

        # ── Fallback: no turn exists yet ──────────────────────────────────────
        # Handles the initial proactive greeting where StartFrame fires
        # on_turn_started but no UserStartedSpeakingFrame precedes it.
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
        """Primary turn-creation path for user speech.

        Called synchronously (inline) from process_frame when
        UserStartedSpeakingFrame arrives at the SDK observer.  Because
        TurnTrackingObserver's on_turn_started fires as an asyncio task
        (scheduled, not awaited), it arrives *after* this call.  This method
        therefore creates the LatencyTurn immediately so that the subsequent
        stopped-speaking / VAD handlers can always find it.
        """
        self._user_has_spoken = True
        started_ms = self._rel_ms(timestamp_ns)

        # Interruption: record when the user cuts in on the bot.
        if self._bot_turn_idx is not None and self._bot_turn_idx < len(self.latency_turns):
            self.latency_turns[self._bot_turn_idx].interrupted_at_ms = started_ms

        # ── Collapse guard ────────────────────────────────────────────────────
        # Reuse the existing turn when the bot has not yet responded.
        # Both conditions are intentionally checked and are not redundant:
        # - bot_started_ms == 0: bot hasn't spoken yet for this turn
        # - not llm_completed: the LLM can finish (setting llm_completed=True)
        #   before TTS fires bot_started_ms; without this check a user follow-up
        #   in that gap would wrongly collapse into the stale turn.
        if self._current_user_turn_latency_idx is not None:
            idx = self._current_user_turn_latency_idx
            if idx < len(self.latency_turns):
                current = self.latency_turns[idx]
                if current.bot_started_ms == 0 and not current.llm_completed:
                    if current.user_started_ms == 0:
                        current.user_started_ms = started_ms
                    else:
                        current.user_started_ms = min(current.user_started_ms, started_ms)
                    if self._open_latency_idx is None:
                        self._open_latency_idx = idx
                    return

        # ── New turn ──────────────────────────────────────────────────────────
        # Bot has already responded to the previous turn (or no turn exists).
        # _turn_to_latency_idx / _active_turn_number are set later when
        # on_turn_started (async task) fires and finds this new turn.
        new_idx = len(self.latency_turns)
        self._open_latency_idx = new_idx
        self._current_user_turn_latency_idx = new_idx
        self.latency_turns.append(
            LatencyTurn(turn_index=new_idx, user_started_ms=started_ms)
        )

    def _get_active_latency_idx(self) -> int | None:
        """Return the current user turn's latency index.

        Uses _active_turn_number (set by on_turn_started async task) as the
        primary source, with _current_user_turn_latency_idx (set synchronously
        by on_user_started_speaking) as a fallback for the window between when
        the user starts speaking and when the async task fires.
        """
        if self._active_turn_number is not None:
            idx = self._turn_to_latency_idx.get(self._active_turn_number)
            if idx is not None and idx < len(self.latency_turns):
                return idx
        if (
            self._current_user_turn_latency_idx is not None
            and self._current_user_turn_latency_idx < len(self.latency_turns)
        ):
            return self._current_user_turn_latency_idx
        return None

    def on_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Capture user_stopped_ms directly from VAD frame.

        Used as the primary source for interrupted turns where on_latency_breakdown
        receives user_turn_start_time=None and cannot compute user_stopped_ms.
        on_latency_breakdown overrides this with its computed value when available.
        """
        idx = self._get_active_latency_idx()
        if idx is None:
            return
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
            # Safety net: bot is greeting proactively before on_turn_started fired.
            # Set _current_user_turn_latency_idx so that when on_turn_started(1)
            # fires asynchronously it can find this turn via the is_proactive check
            # instead of creating a spurious second turn.
            turn = LatencyTurn(turn_index=0, is_proactive=True)
            turn.bot_started_ms = self._rel_ms(timestamp_ns)
            self.latency_turns.append(turn)
            self._bot_turn_idx = 0
            self._current_user_turn_latency_idx = 0
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
        idx = self._get_active_latency_idx()
        if idx is None:
            # VADUserStoppedSpeakingFrame can legitimately fire before any turn
            # starts (background noise, room ambience at call start) — debug only.
            logger.debug("[tuner] on_vad_stopped: no active turn")
            return
        self._vad_stopped_ns_by_turn[idx] = timestamp_ns

    def on_user_turn_stopped(self, timestamp_ns: int) -> None:
        idx = self._get_active_latency_idx()
        if idx is None:
            logger.warning("[tuner] on_user_turn_stopped: no active turn")
            return
        turn = self.latency_turns[idx]
        vad_stopped_ns = self._vad_stopped_ns_by_turn.get(idx)
        if vad_stopped_ns is None:
            logger.warning("[tuner] on_user_turn_stopped: vad_stopped_ns not set on turn {}", idx)
            return
        gap_ms = (timestamp_ns - vad_stopped_ns) // 1_000_000
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
            if is_real_user_turn and not turn.interrupted_at_ms and turn.user_stopped_ms > 0:
                turn.bot_started_ms = turn.user_stopped_ms + latency_ms

        llm_ttfb_ms: int | None = None
        tts_ttfb_ms: int | None = None
        breakpoint()
        for ttfb in getattr(breakdown, "ttfb", []) or []:
            processor = str(getattr(ttfb, "processor", "")).lower()
            duration_ms = int((getattr(ttfb, "duration_secs", 0) or 0) * 1000)
            if duration_ms > 0:
                if "tts" in processor:
                    if tts_ttfb_ms is None:
                        tts_ttfb_ms = duration_ms
                else:
                    if llm_ttfb_ms is None:
                        llm_ttfb_ms = duration_ms
        turn.ttfb_ms = tts_ttfb_ms
        turn.llm_ms = llm_ttfb_ms

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
        idx = self._get_active_latency_idx()
        if idx is not None:
            turn = self.latency_turns[idx]
            if turn.user_started_ms > 0 and turn.user_stopped_ms == 0:
                turn.user_stopped_ms = self._rel_ms(timestamp_ns)

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
                    # Assignment (not +=): only one TTS job runs per turn, so the latest
                    # value is always the correct one. Multiple LLM steps can fire in a
                    # single turn (e.g. parallel tool calls), so LLM uses accumulation.
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

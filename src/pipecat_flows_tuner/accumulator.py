"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .models import CallPayload, LatencyTurn
from .payload_builder import build_payload
from .tool_timing_registry import ToolTimingRegistry


@dataclass
class FlowsAccumulator:
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
    # VAD frame timestamps that arrived before on_turn_started (async race guard).
    _pending_user_started_ns: int | None = field(default=None, repr=False)
    _pending_user_stopped_ns: int | None = field(default=None, repr=False)

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

    # misc
    done: bool = False

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

    def on_start(self, timestamp_ns: int) -> None:
        self.call_start_abs_ns = timestamp_ns

    def on_turn_started(self, turn_number: int, timestamp_ns: int) -> None:
        """Called by TurnTrackingObserver when the user starts speaking.

        If the previous turn has not yet received a bot response (breakdown not fired),
        we collapse this into the existing LatencyTurn rather than creating a new one.
        This handles users who pause briefly and speak again before the bot replies.
        The breakdown fires once per bot response, so one LatencyTurn must match it.
        """
        if self._open_latency_idx is not None and self._open_latency_idx < len(self.latency_turns):
            # Collapse consecutive user fragments before bot starts speaking.
            active_turn = self.latency_turns[self._open_latency_idx]
            if active_turn.bot_started_ms == 0:
                started_ms = self._rel_ms(timestamp_ns)
                if active_turn.user_started_ms == 0:
                    active_turn.user_started_ms = started_ms
                else:
                    active_turn.user_started_ms = min(
                        active_turn.user_started_ms, started_ms
                    )
                self._turn_to_latency_idx[turn_number] = self._open_latency_idx
                self._active_turn_number = turn_number
                return

        new_idx = len(self.latency_turns)
        self._open_latency_idx = new_idx
        self._turn_to_latency_idx[turn_number] = new_idx
        self._active_turn_number = turn_number
        self.latency_turns.append(
            LatencyTurn(
                turn_index=new_idx,
                user_started_ms=self._rel_ms(timestamp_ns),
            )
        )
        turn = self.latency_turns[new_idx]
        if self._pending_user_started_ns is not None:
            started_ms = self._rel_ms(self._pending_user_started_ns)
            if turn.user_started_ms == 0:
                turn.user_started_ms = started_ms
            else:
                turn.user_started_ms = min(turn.user_started_ms, started_ms)
            self._pending_user_started_ns = None
        if self._pending_user_stopped_ns is not None:
            turn.user_stopped_ms = max(
                turn.user_stopped_ms, self._rel_ms(self._pending_user_stopped_ns)
            )
            self._pending_user_stopped_ns = None

    def on_turn_ended(self, turn_number: int, was_interrupted: bool) -> None:
        """Called by TurnTrackingObserver when a turn ends (bot finished or interrupted)."""
        idx = self._turn_to_latency_idx.get(turn_number)
        if idx is not None and idx < len(self.latency_turns):
            self.latency_turns[idx].was_interrupted = was_interrupted
        self._active_turn_number = None

    def on_user_started_speaking(self, timestamp_ns: int) -> None:
        """Use frame timestamp as the authoritative user-start anchor for the active turn."""
        self._pending_user_started_ns = timestamp_ns  # cache in case on_turn_started fires late
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
        self._pending_user_started_ns = None  # consumed; don't leak into next on_turn_started

    def on_user_stopped_speaking(self, timestamp_ns: int) -> None:
        """Capture user_stopped_ms directly from VAD frame.

        Used as the primary source for interrupted turns where on_latency_breakdown
        receives user_turn_start_time=None and cannot compute user_stopped_ms.
        on_latency_breakdown overrides this with its computed value when available.
        """
        self._pending_user_stopped_ns = timestamp_ns  # cache in case on_turn_started fires late
        if self._active_turn_number is None:
            return
        idx = self._turn_to_latency_idx.get(self._active_turn_number)
        if idx is not None and idx < len(self.latency_turns):
            stopped_ms = self._rel_ms(timestamp_ns)
            turn = self.latency_turns[idx]
            turn.user_stopped_ms = max(turn.user_stopped_ms, stopped_ms)
            self._pending_user_stopped_ns = None  # consumed; don't leak into next on_turn_started

    def on_function_call_result(self, tool_call_id: str, timestamp_ns: int) -> None:
        self.registry.record_completion_ns(tool_call_id, timestamp_ns)

    def on_bot_stopped(self, timestamp_ns: int) -> None:
        if self.done:
            return
        bot_stopped_ms = self._rel_ms(timestamp_ns)
        if self._bot_turn_idx is not None and self._bot_turn_idx < len(self.latency_turns):
            # Keep bot stop anchored to the same turn chosen at breakdown time,
            # even if a new user turn started meanwhile (interruption case).
            self.latency_turns[self._bot_turn_idx].bot_stopped_ms = bot_stopped_ms
            self._bot_turn_idx = None
        else:
            logger.warning(
                "[flows-tuner] bot_stopped with no active bot turn index; ignoring event"
            )

    def on_function_call_in_progress(self, frame: Any, timestamp_ns: int) -> None:
        tool_call_id = getattr(frame, "tool_call_id", None)
        if tool_call_id:
            self.registry.record_invocation_ns(tool_call_id, timestamp_ns)

    def on_bot_started_speaking(self, timestamp_ns: int) -> None:
        if self._open_latency_idx is None or self._open_latency_idx >= len(self.latency_turns):
            return
        turn = self.latency_turns[self._open_latency_idx]
        turn.bot_started_ms = self._rel_ms(timestamp_ns)
        self._bot_turn_idx = self._open_latency_idx
        self._open_latency_idx = None

    def on_latency_measured(self, latency_secs: float) -> None:
        self._pending_latency_ms_queue.append(max(0, int(latency_secs * 1000)))

    def on_latency_breakdown(self, breakdown: Any) -> None:
        if self._active_turn_number is None:
            logger.warning(
                "[flows-tuner] on_latency_breakdown fired with no active turn — skipping"
            )
            return
        idx = self._turn_to_latency_idx.get(self._active_turn_number)
        if idx is None or idx >= len(self.latency_turns):
            logger.warning(
                "[flows-tuner] on_latency_breakdown: active turn mapping invalid — skipping"
            )
            return

        turn = self.latency_turns[idx]

        user_start_abs = getattr(breakdown, "user_turn_start_time", None)
        user_turn_secs = getattr(breakdown, "user_turn_secs", None)
        user_stop_abs = (
            (user_start_abs + user_turn_secs)
            if user_start_abs is not None and user_turn_secs is not None
            else None
        )
        is_real_user_turn = True
        if user_start_abs is not None:
            computed_started_ms = self._abs_to_rel_ms(user_start_abs)
            if computed_started_ms > 0:
                # Prefer breakdown timestamps when available, but only when they represent
                # a real user utterance (> 0ms from call start). For the initial proactive
                # bot greeting the breakdown fires with user_turn_start_time ≈ call start,
                # which would wrongly overwrite the user-turn timing captured by
                # on_turn_started with 0.
                turn.user_started_ms = computed_started_ms
                turn.user_stopped_ms = self._abs_to_rel_ms(user_stop_abs)
            else:
                # user_turn_start_time ≈ call start → initial proactive greeting breakdown.
                # Do not derive bot_started_ms from this latency; on_bot_started_speaking
                # already captured the correct bot start from the BotStartedSpeakingFrame.
                is_real_user_turn = False

        if self._pending_latency_ms_queue:
            latency_ms = self._pending_latency_ms_queue.popleft()
            if is_real_user_turn and turn.bot_started_ms == 0:
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

        # Preserve the bot-side turn mapping across interruptions.
        self._bot_turn_idx = idx
        if self._open_latency_idx == idx:
            self._open_latency_idx = None

        self._pending_pipecat_llm_processing_s = 0.0
        self._pending_pipecat_tts_processing_s = 0.0

    def on_call_end(self, timestamp_ns: int) -> None:
        if self.done:
            return
        self.done = True
        self.call_end_abs_ns = timestamp_ns

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
                    self._pending_pipecat_llm_processing_s = val

    def build_payload(self, config: Any, transcript: list[dict[str, Any]]) -> CallPayload:
        return build_payload(self, config, transcript)

"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .models import CallPayload, LatencyTurn, NodeTransitionRecord, PendingTransition
from .payload_builder import build_payload
from .tool_timing_registry import ToolTimingRegistry


@dataclass
class FlowsAccumulator:
    """Collects runtime events and produces a final call payload."""

    # call-level timing
    call_start_abs_ns: int = 0
    call_end_abs_ns: int = 0

    # flow state
    _current_node: str | None = field(default=None, repr=False)

    # node transitions
    node_transitions: list[NodeTransitionRecord] = field(default_factory=list)
    _pending_transition: PendingTransition | None = field(default=None, repr=False)

    # latency tracking
    latency_turns: list[LatencyTurn] = field(default_factory=list)

    # Active turn tracking
    _active_turn_number: int | None = field(default=None, repr=False)
    _turn_to_latency_idx: dict[int, int] = field(default_factory=dict, repr=False)
    # Turn index waiting for first bot response anchor (bot-start).
    _open_latency_idx: int | None = field(default=None, repr=False)
    # Turn index currently speaking; consumed by on_bot_stopped.
    _bot_turn_idx: int | None = field(default=None, repr=False)

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

    def _backfill_initial_transition_timestamp(self, timestamp_ns: int) -> None:
        if self.call_start_abs_ns == 0:
            return
        rel_ms = self._rel_ms(timestamp_ns)
        if rel_ms <= 0:
            return
        for transition in self.node_transitions:
            if transition.trigger_function is None and transition.timestamp_ms == 0:
                transition.timestamp_ms = rel_ms
                return

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

    def get_pending_transition(self) -> PendingTransition | None:
        return self._pending_transition

    def on_node_entered(
        self,
        from_node: str | None,
        to_node: str,
        node_config: dict[str, Any],
        trigger: PendingTransition | None,
        state_snapshot: dict[str, Any],
        timestamp_ns: int,
    ) -> None:
        functions_available: list[str] = []
        for function_ref in node_config.get("functions", []):
            if hasattr(function_ref, "name"):
                name = function_ref.name
            elif isinstance(function_ref, dict):
                name = function_ref.get("name")
            else:
                name = str(function_ref)
            if name:
                functions_available.append(str(name))
        self.node_transitions.append(NodeTransitionRecord(
            from_node=from_node,
            to_node=to_node,
            trigger_function=trigger.function_name if trigger else None,
            trigger_args=trigger.arguments if trigger else None,
            state_snapshot=state_snapshot,
            task_messages=node_config.get("task_messages", []),
            functions_available=functions_available,
            timestamp_ms=self._rel_ms(timestamp_ns),
            trigger_timestamp_ms=trigger.timestamp_ms if trigger else None,
        ))
        self._current_node = to_node
        self._pending_transition = None

    def on_start(self, timestamp_ns: int) -> None:
        self.call_start_abs_ns = timestamp_ns

    def on_turn_started(self, turn_number: int, timestamp_ns: int) -> None:
        """Called by TurnTrackingObserver when the user starts speaking.

        If the previous turn has not yet received a bot response (breakdown not fired),
        we collapse this into the existing LatencyTurn rather than creating a new one.
        This handles users who pause briefly and speak again before the bot replies.
        The breakdown fires once per bot response, so one LatencyTurn must match it.
        """
        self._backfill_initial_transition_timestamp(timestamp_ns)

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
                node=self._current_node,
                user_started_ms=self._rel_ms(timestamp_ns),
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
        self._backfill_initial_transition_timestamp(timestamp_ns)
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
        self._backfill_initial_transition_timestamp(timestamp_ns)
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
        if self.done:
            return
        bot_stopped_ms = self._rel_ms(timestamp_ns)
        if self._bot_turn_idx is not None and self._bot_turn_idx < len(self.latency_turns):
            # Use the turn index captured at on_latency_breakdown time.
            # This is correct even for interrupted turns where on_turn_started(N+1)
            # has already fired and appended a new entry to latency_turns.
            self.latency_turns[self._bot_turn_idx].bot_stopped_ms = bot_stopped_ms
            self._bot_turn_idx = None
        else:
            logger.warning(
                "[flows-tuner] bot_stopped with no active bot turn index; ignoring event"
            )

    def on_function_call_in_progress(self, frame: Any, timestamp_ns: int) -> None:
        name = getattr(frame, "function_name", "") or "function"
        arguments = getattr(frame, "arguments", None)
        ms = self._rel_ms(timestamp_ns)
        self._backfill_initial_transition_timestamp(timestamp_ns)
        tool_call_id = getattr(frame, "tool_call_id", None)
        if tool_call_id:
            self.registry.record_invocation_ns(tool_call_id, timestamp_ns)
        self._pending_transition = PendingTransition(
            function_name=name,
            arguments=arguments,
            timestamp_ms=ms,
        )

    def on_bot_started_speaking(self, timestamp_ns: int) -> None:
        self._backfill_initial_transition_timestamp(timestamp_ns)
        if self._open_latency_idx is None or self._open_latency_idx >= len(self.latency_turns):
            return
        turn = self.latency_turns[self._open_latency_idx]
        turn.bot_started_ms = self._rel_ms(timestamp_ns)
        turn.bot_node = self._current_node
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
        if user_start_abs is not None:
            # Override with breakdown's authoritative timestamps.
            # When user_turn_start_time is None (interrupted turns), the VAD-captured
            # values from on_user_stopped_speaking / on_turn_started are preserved.
            turn.user_started_ms = self._abs_to_rel_ms(user_start_abs)
            turn.user_stopped_ms = self._abs_to_rel_ms(user_stop_abs)

        if self._pending_latency_ms_queue:
            latency_ms = self._pending_latency_ms_queue.popleft()
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

        # Remember which latency_turn index the bot is currently speaking for,
        # so on_bot_stopped writes to the right turn even if a new user turn
        # has already started (interrupted scenario).
        self._bot_turn_idx = idx
        if not turn.bot_node:
            turn.bot_node = self._current_node
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

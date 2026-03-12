"""Collector concern: ingest Pipecat events and maintain call runtime state."""

from dataclasses import dataclass, field
from typing import Any

from .latency_metrics import flush_latency_turn
from .models import CallPayload, LatencyTurn, NodeTransitionRecord, PendingTransition
from .payload_builder import build_payload
from .transcript_enricher import enrich_transcript


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
    _turn_index: int = field(default=0, repr=False)
    _user_stopped_ns: int = field(default=0, repr=False)
    _llm_started_ns: int = field(default=0, repr=False)
    _tts_started_ns: int = field(default=0, repr=False)
    _bot_started_ns: int = field(default=0, repr=False)
    _bot_stopped_ns: int = field(default=0, repr=False)
    _user_started_ns: int = field(default=0, repr=False)
    _latency_node: str | None = field(default=None, repr=False)

    # call-level pipecat-sourced counters (summed across all MetricsFrames)
    _pipecat_llm_total_tokens: int = field(default=0, repr=False)
    _pipecat_tts_chars: int = field(default=0, repr=False)

    # per-turn pending pipecat metrics (reset on each on_user_stopped)
    _pending_pipecat_llm_ttfb_s: float = field(default=0.0, repr=False)
    _pending_pipecat_tts_ttfb_s: float = field(default=0.0, repr=False)
    _pending_pipecat_llm_processing_s: float = field(default=0.0, repr=False)
    _pending_pipecat_tts_processing_s: float = field(default=0.0, repr=False)

    # misc
    done: bool = False

    def _rel_ms(self, abs_ns: int) -> int:
        if self.call_start_abs_ns == 0 or abs_ns == 0:
            return 0
        return (abs_ns - self.call_start_abs_ns) // 1_000_000

    def _ns_to_ms(self, a_ns: int, b_ns: int) -> int | None:
        if a_ns == 0 or b_ns == 0:
            return None
        return max(0, (b_ns - a_ns) // 1_000_000)

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
        ))
        self._current_node = to_node
        self._pending_transition = None

    def on_start(self, timestamp_ns: int) -> None:
        self.call_start_abs_ns = timestamp_ns

    def on_user_started(self, timestamp_ns: int) -> None:
        if not self._user_started_ns:
            self._user_started_ns = timestamp_ns

    def on_user_stopped(self, frame: Any, timestamp_ns: int) -> None:
        stop_correction_ns = int(getattr(frame, "stop_secs", 0) * 1_000_000_000)
        self._user_stopped_ns = timestamp_ns - stop_correction_ns
        self._llm_started_ns = 0
        self._tts_started_ns = 0
        self._bot_started_ns = 0
        self._latency_node = self._current_node
        self._pending_pipecat_llm_ttfb_s = 0.0
        self._pending_pipecat_tts_ttfb_s = 0.0
        self._pending_pipecat_llm_processing_s = 0.0
        self._pending_pipecat_tts_processing_s = 0.0

    def on_llm_started(self, timestamp_ns: int) -> None:
        if self._user_stopped_ns and self._llm_started_ns == 0:
            self._llm_started_ns = timestamp_ns

    def on_tts_started(self, timestamp_ns: int) -> None:
        if self._user_stopped_ns and self._tts_started_ns == 0:
            self._tts_started_ns = timestamp_ns

    def on_bot_started_speaking(self, timestamp_ns: int) -> None:
        if self._user_stopped_ns and self._bot_started_ns == 0:
            self._bot_started_ns = timestamp_ns
            self._flush_latency_turn()

    def on_bot_stopped(self, timestamp_ns: int) -> None:
        if self.done:
            return
        self._bot_stopped_ns = timestamp_ns
        if self.latency_turns:
            self.latency_turns[-1].bot_stopped_ms = self._rel_ms(timestamp_ns)

    def on_function_call_in_progress(self, frame: Any, timestamp_ns: int) -> None:
        name = getattr(frame, "function_name", "") or "function"
        arguments = getattr(frame, "arguments", None)
        ms = self._rel_ms(timestamp_ns)
        self._pending_transition = PendingTransition(
            function_name=name,
            arguments=arguments,
            timestamp_ms=ms,
        )

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
            elif cls_name == "TTFBMetricsData":
                processor = str(getattr(d, "processor", "")).lower()
                val = getattr(d, "value", 0) or 0
                if "ttsservice" in processor:
                    self._pending_pipecat_tts_ttfb_s = val
                else:
                    self._pending_pipecat_llm_ttfb_s = val
            elif cls_name == "ProcessingMetricsData":
                processor = str(getattr(d, "processor", "")).lower()
                val = getattr(d, "value", 0) or 0
                if "tts" in processor:
                    self._pending_pipecat_tts_processing_s = val
                else:
                    self._pending_pipecat_llm_processing_s = val

    def _flush_latency_turn(self) -> None:
        flush_latency_turn(self)

    def build_payload(self, config: Any, transcript: list[dict[str, Any]]) -> CallPayload:
        return build_payload(self, config, transcript)

    def _enrich_transcript(self, messages: list[dict[str, Any]]) -> list[Any]:
        return enrich_transcript(self, messages)

"""What the observer does on Pipecat releases without what it now relies on.

The SDK supports pipecat >= 1.0. Pipecat 1.9 added SpeakingObserver, FunctionCallObserver and
ServiceMetricsObserver, and 1.12 marks each frame's first push (FramePushed.first_push). The
observer uses them when they're installed. Everything it does without them is here, so it can
be deleted in one place once the SDK's pipecat floor reaches 1.12.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

try:  # pipecat >= 1.9
    from pipecat.observers.function_call_observer import FunctionCallObserver
    from pipecat.observers.service_metrics_observer import ServiceMetricsObserver
    from pipecat.observers.speaking_observer import SpeakingObserver, SpeechEventKind
except ImportError:  # pragma: no cover - the locked pipecat 1.5 test run takes this path
    FunctionCallObserver = None  # type: ignore[assignment,misc,unused-ignore]
    ServiceMetricsObserver = None  # type: ignore[assignment,misc,unused-ignore]
    SpeakingObserver = None  # type: ignore[assignment,misc,unused-ignore]
    SpeechEventKind = None  # type: ignore[assignment,misc,unused-ignore]


class FrameDedupe:
    """Tells the observer whether it is seeing a frame for the first time.

    A pipeline-level observer hears about every hop of a frame. Pipecat >= 1.12 marks the first
    one (FramePushed.first_push), the moment its own observers are told. Older releases don't, so
    a window of the last 256 frame ids stands in for it. A frame that waits in a processor's
    queue behind more than 256 others falls out of that window and is handled again (#25 item 7).
    """

    def __init__(self, window: int = 256) -> None:
        self._seen: set[int] = set()
        self._history: deque[int] = deque(maxlen=window)

    def first_time(self, data: Any) -> bool:
        first_push = getattr(data, "first_push", None)
        if first_push is not None:
            return bool(first_push)
        frame_id = data.frame.id
        if frame_id in self._seen:
            return False
        self._seen.add(frame_id)
        self._history.append(frame_id)
        if len(self._seen) > len(self._history):
            self._seen = set(self._history)
        return True


@dataclass
class LegacyTiming:
    """CallAccumulator's timing without SpeakingObserver and ServiceMetricsObserver.

    - Latency: UserBotLatencyObserver's values, queued in arrival order and paired with the next
      breakdown; the agent row's start is moved to "user stop + latency".
    - llm_node_ttft: the processing time of every LLM inference in the turn.
    """

    # Ordered pairing for latency observer callbacks (on_latency_measured then breakdown).
    _pending_latency_ms_queue: deque[int] = field(default_factory=deque, repr=False)

    def on_latency_measured(self, latency_secs: float) -> None:
        self._pending_latency_ms_queue.append(max(0, int(latency_secs * 1000)))

    def _apply_queued_latency(
        self, bot_seg: Any, user_seg: Any, *, is_real_user_turn: bool, speech_timing: bool
    ) -> None:
        """Take this reply's queued latency. Without speech timing, move the bot start by it;
        with it, the bot start stays as observed and e2e comes from the caller's speech."""
        if not self._pending_latency_ms_queue:
            return
        latency_ms = self._pending_latency_ms_queue.popleft()
        if (
            not speech_timing
            and is_real_user_turn
            and bot_seg is not None
            and user_seg is not None
            and not bot_seg.interrupted_at_ms
            and (user_seg.stop_ms or 0) > 0
        ):
            bot_seg.start_ms = user_seg.stop_ms + latency_ms

    @staticmethod
    def _processing_llm_ms(processing_s: float, llm_ttfb_ms: int | None) -> int | None:
        """The processing-time metric, falling back to the LLM's own TTFB: some providers (e.g.
        Google/Gemini) emit a TTFB metric but no ProcessingMetricsData, which would otherwise
        leave LLM latency blank."""
        return round(processing_s * 1000) if processing_s else llm_ttfb_ms

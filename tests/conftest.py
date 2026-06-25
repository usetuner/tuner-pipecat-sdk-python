"""Shared pytest fixtures for tuner_pipecat_sdk tests."""

from types import SimpleNamespace

import pytest

from tuner_pipecat_sdk.accumulator import CallAccumulator
from tuner_pipecat_sdk.config import TunerConfig


class Replay:
    """Drives a CallAccumulator through its on_* event methods exactly as the pipeline-level
    observer would, scripting a real call with explicit ms timestamps (relative to call start).

    The transcript is built live from these events (event-sourced), so tests script the call
    and assert the rendered rows — no hand-built segment/transcript lists.
    """

    def __init__(self, base_ms: int = 0):
        self.base = 1_000_000_000  # arbitrary non-zero call-start anchor
        self.acc = CallAccumulator()
        self.acc.call_start_abs_ns = self.base
        self.acc.on_start(self.base)

    def _ns(self, ms: int) -> int:
        return self.base + ms * 1_000_000

    # turn lifecycle (TurnTrackingObserver)
    def turn_start(self, n: int, ms: int) -> "Replay":
        self.acc.on_turn_started(n, self._ns(ms))
        return self

    def turn_end(self, n: int, interrupted: bool = False) -> "Replay":
        self.acc.on_turn_ended(n, interrupted)
        return self

    # user speech
    def user_start(self, ms: int) -> "Replay":
        self.acc.on_user_started_speaking(self._ns(ms))
        return self

    def transcription(self, text: str, ms: int) -> "Replay":
        self.acc.on_transcription(text, self._ns(ms))
        return self

    def user_stop(self, ms: int) -> "Replay":
        self.acc.on_user_stopped_speaking(self._ns(ms))
        return self

    # bot generation + voicing
    def llm_start(self, ms: int) -> "Replay":
        self.acc.on_llm_response_start(self._ns(ms))
        return self

    def llm_text(self, text: str) -> "Replay":
        self.acc.on_llm_text(text)
        return self

    def llm_end(self, ms: int) -> "Replay":
        self.acc.on_llm_response_end(self._ns(ms))
        return self

    def generate(self, text: str, start_ms: int, end_ms: int) -> "Replay":
        """Shorthand: a full LLM response (start → text → end)."""
        return self.llm_start(start_ms).llm_text(text).llm_end(end_ms)

    def tts(self, text: str, ms: int) -> "Replay":
        self.acc.on_tts_text(text, self._ns(ms))
        return self

    def bot_start(self, ms: int) -> "Replay":
        self.acc.on_bot_started_speaking(self._ns(ms))
        return self

    def bot_stop(self, ms: int) -> "Replay":
        self.acc.on_bot_stopped(self._ns(ms))
        return self

    # tools
    def tool_call(self, name: str, tool_call_id: str, arguments, ms: int) -> "Replay":
        frame = SimpleNamespace(
            function_name=name, tool_call_id=tool_call_id, arguments=arguments
        )
        self.acc.on_function_call_in_progress(frame, self._ns(ms))
        return self

    def tool_result(self, name: str, tool_call_id: str, result, ms: int) -> "Replay":
        frame = SimpleNamespace(function_name=name, tool_call_id=tool_call_id, result=result)
        self.acc.on_function_call_result(frame, self._ns(ms))
        return self

    # latency (drives the same path UserBotLatencyObserver would)
    def latency(
        self, ttfb_secs: float, processor: str, user_start_ms: int | None = None
    ) -> "Replay":
        """Feed one latency breakdown for the most recent bot response."""
        ttfb = [SimpleNamespace(duration_secs=ttfb_secs, processor=processor)]
        user_turn_start_time = (
            (self.base / 1_000_000_000) + user_start_ms / 1000
            if user_start_ms is not None
            else None
        )
        self.acc.on_latency_breakdown(
            SimpleNamespace(user_turn_start_time=user_turn_start_time, ttfb=ttfb)
        )
        return self

    def stt_ttfb(self, secs: float) -> "Replay":
        """Record an STT TTFB on the active user segment (becomes the user row's stt_node_ttfb)."""
        data = type("TTFBMetricsData", (), {})()
        data.processor = "GoogleSTTService"
        data.value = secs
        self.acc.on_metrics_frame(SimpleNamespace(data=[data]))
        return self

    def end(self, ms: int) -> "Replay":
        self.acc.on_call_end(self._ns(ms))
        return self

    def rows(self, config):
        """Build and return the rendered transcript rows."""
        if not self.acc.done:
            self.acc.on_call_end(self._ns(10_000))
        return self.acc.build_payload(config).transcript_with_tool_calls


@pytest.fixture
def replay():
    return Replay


@pytest.fixture
def tuner_config():
    return TunerConfig(
        api_key="test-api-key",
        workspace_id=42,
        agent_id="test-agent",
        call_id="call-123",
        call_type="web_call",
        base_url="https://tuner.example.com",
        recording_url="https://example.com/recording.mp3",
        debug=False,
        asr_model="deepgram",
        llm_model="gpt-4",
        tts_model="eleven",
    )

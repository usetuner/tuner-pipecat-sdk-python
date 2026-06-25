"""End-to-end flow tests: runtime observer flow with accumulator and payload build."""

from types import SimpleNamespace

import pytest

from tuner_pipecat_sdk.accumulator import CallAccumulator
from tuner_pipecat_sdk.config import TunerConfig


def _metric(cls_name: str, **kwargs):
    return type(cls_name, (), kwargs)()


@pytest.fixture
def config():
    return TunerConfig(
        api_key="key",
        workspace_id=1,
        agent_id="agent",
        call_id="call-e2e",
        base_url="https://tuner.test",
        asr_model="asr",
        llm_model="llm",
        tts_model="tts",
    )


def test_full_call_flow_single_turn(config):
    """Simulate one user turn and one bot response; verify payload and transcript."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000

    # Start
    acc.on_start(base_ns)
    assert acc.call_start_abs_ns == base_ns

    # TurnTrackingObserver fires on_turn_started when user begins speaking
    acc.on_turn_started(1, base_ns + 50_000_000)  # user started at +50ms
    acc.on_user_started_speaking(base_ns + 50_000_000)
    acc.on_transcription("Hi", base_ns + 80_000_000)  # real user speech
    acc.on_user_stopped_speaking(base_ns + 100_000_000)  # user stopped at +100ms

    # Runtime observer data
    acc.on_latency_measured(0.15)
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=11)]))
    acc._pending_pipecat_llm_processing_s = 0.05
    acc._pending_pipecat_tts_processing_s = 0.05
    acc.on_llm_response_start(base_ns + 200_000_000)
    acc.on_llm_text("Hello there!")
    acc.on_llm_response_end(base_ns + 220_000_000)
    acc.on_tts_text("Hello there!", base_ns + 210_000_000)
    acc.on_bot_started_speaking(base_ns + 250_000_000)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=1.05,
            user_turn_secs=0.05,
            ttfb=[SimpleNamespace(duration_secs=0.1)],
            function_calls=[],
        )
    )
    acc.on_bot_stopped(base_ns + 400_000_000)

    user_segs = [s for s in acc.speech_segments if s.speaker == "user"]
    bot_segs = [s for s in acc.speech_segments if s.speaker == "bot"]
    assert len(user_segs) == 1
    assert len(bot_segs) == 1
    assert user_segs[0].stop_ms == 100
    assert bot_segs[0].start_ms == 250
    assert bot_segs[0].stop_ms == 400
    assert len(acc.latency_measurements) == 1
    assert acc.latency_measurements[0].ttfb_ms == 100

    # End
    acc.on_call_end(base_ns + 500_000_000)
    assert acc.done
    assert acc.call_end_abs_ns == base_ns + 500_000_000

    payload = acc.build_payload(config)

    assert payload.call_id == config.call_id
    assert payload.duration_ms == 500
    assert payload.call_status == "call_ended"
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 11
    assert len(payload.transcript_with_tool_calls) >= 2
    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "user" in roles
    assert "agent" in roles


def test_full_call_flow_with_tool_call(config):
    """Simulate user → tool call → assistant response."""
    acc = CallAccumulator()
    base_ns = 2_000_000_000

    acc.on_start(base_ns)

    # First turn: user asks for transfer
    acc.on_function_call_in_progress(
        SimpleNamespace(function_name="transfer", arguments={"to": "sales"}, tool_call_id="tc-1"),
        base_ns + 100_000_000,
    )

    acc.on_turn_started(1, base_ns + 10_000_000)  # user started speaking at +10ms
    acc.on_user_started_speaking(base_ns + 10_000_000)
    acc.on_transcription("Transfer me to sales", base_ns + 40_000_000)
    acc.on_user_stopped_speaking(base_ns + 60_000_000)
    acc.on_function_call_result(
        SimpleNamespace(function_name="transfer", tool_call_id="tc-1", result={"ok": True}),
        base_ns + 120_000_000,
    )
    acc.on_latency_measured(0.15)
    acc.on_llm_response_start(base_ns + 130_000_000)
    acc.on_llm_text("Transferring you now.")
    acc.on_llm_response_end(base_ns + 140_000_000)
    acc.on_tts_text("Transferring you now.", base_ns + 145_000_000)
    acc.on_bot_started_speaking(base_ns + 150_000_000)
    acc.on_latency_breakdown(
        SimpleNamespace(
            user_turn_start_time=2.01,
            user_turn_secs=0.04,
            ttfb=[SimpleNamespace(duration_secs=0.07)],
            function_calls=[
                SimpleNamespace(function_name="transfer", start_time=2.1, duration_secs=0.02)
            ],
        )
    )
    acc.on_bot_stopped(base_ns + 350_000_000)
    acc.on_call_end(base_ns + 400_000_000)

    payload = acc.build_payload(config)

    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "agent_function" in roles
    assert "agent_result" in roles
    assert "node_transition" not in roles

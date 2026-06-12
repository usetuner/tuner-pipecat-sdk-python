"""Accumulator payload-shape and metadata tests."""

from types import SimpleNamespace

import pytest

from tuner_pipecat_sdk.accumulator import CallAccumulator
from tuner_pipecat_sdk.models import LatencyTurn


def _metric(cls_name: str, **kwargs):
    return type(cls_name, (), kwargs)()


def test_build_payload_basic(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=50)]))
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    assert payload.call_id == tuner_config.call_id
    assert payload.call_type == tuner_config.call_type
    assert payload.start_timestamp == 1
    assert payload.end_timestamp == 2
    assert payload.duration_ms == 1_000
    assert payload.call_status == "call_ended"
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 50
    assert payload.general_meta_data_raw.ai_models.asr_model == tuner_config.asr_model
    assert len(payload.transcript_with_tool_calls) >= 2


def test_llm_token_uses_pipecat_value(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.on_metrics_frame(
        SimpleNamespace(
            data=[_metric("LLMUsageMetricsData", value=SimpleNamespace(total_tokens=500))]
        )
    )
    payload = acc.build_payload(tuner_config, [])
    assert payload.general_meta_data_raw.usage_token.llm_token == 500


def test_llm_token_is_none_when_pipecat_zero(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [{"role": "user", "content": "A" * 400}])
    assert payload.general_meta_data_raw.usage_token.llm_token is None


def test_tts_char_count_uses_pipecat_value(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=999)]))
    payload = acc.build_payload(tuner_config, [])
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 999


def test_tts_char_count_is_none_when_pipecat_zero(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [])
    assert payload.general_meta_data_raw.usage_token.tts_character_count is None


def test_enrich_transcript_user_and_assistant(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="n",
            ttfb_ms=100,
            llm_ms=50,
            tts_ms=50,
            bot_started_ms=200,
            user_stopped_ms=100,
            user_started_ms=50,
            bot_stopped_ms=300,
        )
    ]
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert "user" in roles
    assert "agent" in roles


def test_enrich_transcript_skips_system(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    transcript = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi!"},
    ]
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node=None,
            ttfb_ms=0,
            llm_ms=0,
            tts_ms=0,
            bot_started_ms=0,
            user_stopped_ms=0,
            user_started_ms=0,
            bot_stopped_ms=100,
        )
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert "system" not in roles


def test_payload_transcript_preserves_conversation_order(tuner_config):
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 10_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-1", base_ns + 80_000_000)
    acc.registry.record_completion_ns("tc-1", base_ns + 160_000_000)
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            node="greeting",
            ttfb_ms=10,
            llm_ms=20,
            tts_ms=30,
            bot_started_ms=6000,
            user_stopped_ms=1000,
            user_started_ms=500,
            bot_stopped_ms=7000,
        )
    ]
    transcript = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc-1", "function": {"name": "transfer", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert roles == ["user", "agent_function", "agent_result", "agent"]


def test_payload_keeps_initial_greeting_before_first_user(tuner_config):
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 5_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0,
            ttfb_ms=100,
            llm_ms=50,
            tts_ms=40,
            bot_started_ms=2000,
            user_stopped_ms=1300,
            user_started_ms=1000,
            bot_stopped_ms=2800,
        )
    ]
    transcript = [
        {"role": "assistant", "content": "Welcome to Pipecat Pizza!"},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hi there! Which pizza would you like?"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert roles == ["agent", "user", "agent"]

    greeting = payload.transcript_with_tool_calls[0]
    assert greeting.text == "Welcome to Pipecat Pizza!"
    assert greeting.start_ms == 0
    assert greeting.end_ms == 0


def test_build_payload_includes_disconnection_reason(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.set_disconnection_reason("user_hangup")
    payload = acc.build_payload(tuner_config, [])
    assert payload.disconnection_reason == "user_hangup"


def test_build_payload_disconnection_reason_none_when_unset(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [])
    assert payload.disconnection_reason is None


def test_build_payload_disconnection_reason_omitted_from_dict_when_none(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [])
    assert "disconnection_reason" not in payload.to_dict()


def test_cost_calculator_invoked_and_stored_in_payload(tuner_config):
    base_ns = 1_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 60_000_000_000  # 60s call
    acc.done = True
    # Simulate LLM metrics: 100 prompt + 50 completion = 150 total
    token_value = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("LLMUsageMetricsData", value=token_value)]))
    # Simulate TTS metrics: 200 characters
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=200)]))

    def calculate_cost(usage):
        llm_cost = (usage.llm_prompt_tokens or 0) * 0.000_003
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_015
        tts_cost = (usage.tts_characters or 0) * 0.000_030
        stt_cost = usage.stt_audio_seconds * 0.000_006
        return llm_cost + tts_cost + stt_cost

    payload = acc.build_payload(tuner_config, [], calculate_cost)

    expected = 100 * 0.000_003 + 50 * 0.000_015 + 200 * 0.000_030 + 60 * 0.000_006
    assert payload.cost == pytest.approx(expected)
    assert "call_cost" in payload.to_dict()
    assert payload.to_dict()["call_cost"] == pytest.approx(expected)


def test_cost_calculator_none_omits_cost_from_payload(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [])
    assert payload.cost is None
    assert "call_cost" not in payload.to_dict()


def test_cost_calculator_receives_correct_usage_fields(tuner_config):
    base_ns = 1_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 30_000_000_000  # 30s call
    acc.done = True
    token_value = SimpleNamespace(prompt_tokens=80, completion_tokens=20, total_tokens=100)
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("LLMUsageMetricsData", value=token_value)]))
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=500)]))

    captured = {}

    def capture_usage(usage):
        captured.update({
            "prompt": usage.llm_prompt_tokens,
            "completion": usage.llm_completion_tokens,
            "total": usage.llm_total_tokens,
            "tts": usage.tts_characters,
            "stt_secs": usage.stt_audio_seconds,
        })
        return 0.0

    acc.build_payload(tuner_config, [], capture_usage)

    assert captured["prompt"] == 80
    assert captured["completion"] == 20
    assert captured["total"] == 100
    assert captured["tts"] == 500
    assert captured["stt_secs"] == 30

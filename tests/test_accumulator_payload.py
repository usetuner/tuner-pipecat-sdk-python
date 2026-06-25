"""Accumulator payload metadata, usage, and cost tests.

Transcript-row shape/order is covered by test_accumulator_transcript.py (event-sourced). These
tests focus on the non-transcript payload: usage counters, timestamps, disconnection, cost.
"""

from types import SimpleNamespace

import pytest

from tuner_pipecat_sdk.accumulator import CallAccumulator


def _metric(cls_name: str, **kwargs):
    return type(cls_name, (), kwargs)()


def test_build_payload_basic(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=50)]))
    payload = acc.build_payload(tuner_config)
    assert payload.call_id == tuner_config.call_id
    assert payload.call_type == tuner_config.call_type
    assert payload.start_timestamp == 1
    assert payload.end_timestamp == 2
    assert payload.duration_ms == 1_000
    assert payload.call_status == "call_ended"
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 50
    assert payload.general_meta_data_raw.ai_models.asr_model == tuner_config.asr_model


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
    payload = acc.build_payload(tuner_config)
    assert payload.general_meta_data_raw.usage_token.llm_token == 500


def test_llm_token_is_none_when_pipecat_zero(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config)
    assert payload.general_meta_data_raw.usage_token.llm_token is None


def test_tts_char_count_uses_pipecat_value(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=999)]))
    payload = acc.build_payload(tuner_config)
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 999


def test_tts_char_count_is_none_when_pipecat_zero(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config)
    assert payload.general_meta_data_raw.usage_token.tts_character_count is None


def test_build_payload_includes_disconnection_reason(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.set_disconnection_reason("user_hangup")
    payload = acc.build_payload(tuner_config)
    assert payload.disconnection_reason == "user_hangup"


def test_build_payload_disconnection_reason_none_when_unset(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config)
    assert payload.disconnection_reason is None


def test_build_payload_disconnection_reason_omitted_from_dict_when_none(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config)
    assert "disconnection_reason" not in payload.to_dict()


def test_cost_calculator_invoked_and_stored_in_payload(tuner_config):
    base_ns = 1_000_000_000
    acc = CallAccumulator()
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 60_000_000_000  # 60s call
    acc.done = True
    token_value = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("LLMUsageMetricsData", value=token_value)]))
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=200)]))

    def calculate_cost(usage):
        llm_cost = (usage.llm_prompt_tokens or 0) * 0.000_003
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_015
        tts_cost = (usage.tts_characters or 0) * 0.000_030
        stt_cost = usage.stt_audio_seconds * 0.000_006
        return llm_cost + tts_cost + stt_cost

    payload = acc.build_payload(tuner_config, calculate_cost)

    expected = 100 * 0.000_003 + 50 * 0.000_015 + 200 * 0.000_030 + 60 * 0.000_006
    assert payload.cost == pytest.approx(expected)
    assert "call_cost" in payload.to_dict()
    assert payload.to_dict()["call_cost"] == pytest.approx(expected)


def test_cost_calculator_none_omits_cost_from_payload(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config)
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

    acc.build_payload(tuner_config, capture_usage)

    assert captured["prompt"] == 80
    assert captured["completion"] == 20
    assert captured["total"] == 100
    assert captured["tts"] == 500
    assert captured["stt_secs"] == 30

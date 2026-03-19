"""Accumulator payload-shape and metadata tests."""

from types import SimpleNamespace

from tuner_pipecat_sdk.accumulator import FlowsAccumulator
from tuner_pipecat_sdk.models import LatencyTurn


def _metric(cls_name: str, **kwargs):
    return type(cls_name, (), kwargs)()


def test_build_payload_basic(tuner_config):
    acc = FlowsAccumulator()
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
    acc = FlowsAccumulator()
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
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [{"role": "user", "content": "A" * 400}])
    assert payload.general_meta_data_raw.usage_token.llm_token is None


def test_tts_char_count_uses_pipecat_value(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.on_metrics_frame(SimpleNamespace(data=[_metric("TTSUsageMetricsData", value=999)]))
    payload = acc.build_payload(tuner_config, [])
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 999


def test_tts_char_count_is_none_when_pipecat_zero(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    payload = acc.build_payload(tuner_config, [])
    assert payload.general_meta_data_raw.usage_token.tts_character_count is None


def test_enrich_transcript_user_and_assistant(tuner_config):
    acc = FlowsAccumulator()
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
    acc = FlowsAccumulator()
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
    acc = FlowsAccumulator()
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
            "tool_calls": [
                {"id": "tc-1", "function": {"name": "transfer", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done"},
    ]

    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert roles == ["user", "agent_function", "agent_result", "agent"]



def test_payload_keeps_initial_greeting_before_first_user(tuner_config):
    acc = FlowsAccumulator()
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



"""Accumulator payload-shape and metadata tests."""

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn


def test_build_payload_basic(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc._tts_chars = 50
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

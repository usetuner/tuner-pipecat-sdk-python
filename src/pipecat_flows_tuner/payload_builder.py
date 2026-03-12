"""Transform concern: build API payload objects from collected call data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import AiModels, CallPayload, GeneralMetaData, UsageToken
from .transcript_enricher import enrich_transcript

if TYPE_CHECKING:
    from .accumulator import FlowsAccumulator
    from .config import TunerConfig


def build_payload(
    acc: FlowsAccumulator,
    config: TunerConfig,
    transcript: list[dict[str, Any]],
) -> CallPayload:
    enriched = enrich_transcript(acc, transcript)
    start_ts = acc.call_start_abs_ns // 1_000_000_000
    end_ts = acc.call_end_abs_ns // 1_000_000_000

    return CallPayload(
        call_id=config.call_id,
        call_type=config.call_type,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        recording_url=config.recording_url,
        transcript_with_tool_calls=enriched,
        call_status="call_ended",
        duration_ms=max(0, (acc.call_end_abs_ns - acc.call_start_abs_ns) // 1_000_000),
        general_meta_data_raw=GeneralMetaData(
            ai_models=AiModels(
                asr_model=config.asr_model,
                llm_model=config.llm_model,
                tts_model=config.tts_model,
            ),
            usage_token=UsageToken(
                asr_duration=max(0, end_ts - start_ts),
                llm_token=acc._pipecat_llm_total_tokens or None,
                tts_character_count=acc._pipecat_tts_chars or None,
            ),
        ),
    )

"""Transform concern: build API payload objects from collected call data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import AiModels, CallPayload, GeneralMetaData, UsageToken
from .transcript_enricher import enrich_transcript

if TYPE_CHECKING:
    from .accumulator import CallAccumulator
    from .config import TunerConfig


def _ensure_monotonic_bounds(segments: list[Any]) -> list[Any]:
    for segment in segments:
        end_ms = getattr(segment, "end_ms", None)
        if end_ms is not None and end_ms < getattr(segment, "start_ms", 0):
            segment.end_ms = segment.start_ms
    return segments


def build_payload(
    acc: CallAccumulator,
    config: TunerConfig,
    transcript: list[dict[str, Any]],
) -> CallPayload:
    enriched = _ensure_monotonic_bounds(enrich_transcript(acc, transcript))
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
                llm_token=acc.get_total_llm_tokens() or None,
                tts_character_count=acc.get_total_tts_characters() or None,
            ),
        ),
    )

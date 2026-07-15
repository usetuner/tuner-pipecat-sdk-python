"""Transform concern: build API payload objects from collected call data."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .models import AiModels, CallPayload, CallUsage, GeneralMetaData, SegmentEnricher, UsageToken
from .transcript_builder import build_segments_from_turns

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
    cost_calculator: Callable[[CallUsage], float] | None = None,
    segment_enrichers: list[SegmentEnricher] | None = None,
) -> CallPayload:
    """Build the final `CallPayload` sent to Tuner from everything collected during the call.

    Applies every registered ``segment_enrichers`` transform (e.g. merging in
    LangChain/LangGraph segments) to the native pipecat transcript, in order,
    then clamps segment end times so none end before they start. This module
    has no idea what a given enricher does or where it came from -- see
    ``CallAccumulator.register_segment_enricher()``.
    """
    enriched_segments = build_segments_from_turns(acc)
    for enricher in segment_enrichers or []:
        enriched_segments = enricher(enriched_segments)
    enriched = _ensure_monotonic_bounds(enriched_segments)
    start_ts = acc.call_start_abs_ns // 1_000_000_000
    end_ts = acc.call_end_abs_ns // 1_000_000_000

    usage = CallUsage(
        llm_prompt_tokens=acc.get_llm_prompt_tokens() or None,
        llm_completion_tokens=acc.get_llm_completion_tokens() or None,
        llm_total_tokens=acc.get_total_llm_tokens() or None,
        tts_characters=acc.get_total_tts_characters() or None,
        stt_audio_seconds=max(0, end_ts - start_ts),
    )
    cost = cost_calculator(usage) if cost_calculator is not None else None

    return CallPayload(
        call_id=config.call_id,
        call_type=config.call_type,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        recording_url=config.recording_url,
        transcript_with_tool_calls=enriched,
        call_status="call_ended",
        disconnection_reason=acc.disconnection_reason,
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
        sip_call_id=config.sip_call_id,
        sip_headers=config.sip_headers,
        agent_version=config.agent_version,
        recipient=config.recipient,
        cost=cost,
    )

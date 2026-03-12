"""Transform concern: compute latency metrics from Pipecat MetricsFrame data only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import LatencyTurn

if TYPE_CHECKING:
    from .accumulator import FlowsAccumulator


def flush_latency_turn(acc: FlowsAccumulator) -> None:
    """Append one LatencyTurn using only Pipecat-sourced metrics (no timestamp fallback)."""
    ttfb_ms: int | None = None
    if acc._pending_pipecat_tts_ttfb_s:
        ttfb_ms = round(acc._pending_pipecat_tts_ttfb_s * 1000)
    elif acc._pending_pipecat_llm_ttfb_s:
        ttfb_ms = round(acc._pending_pipecat_llm_ttfb_s * 1000)

    llm_ms: int | None = (
        round(acc._pending_pipecat_llm_processing_s * 1000)
        if acc._pending_pipecat_llm_processing_s
        else None
    )
    tts_ms: int | None = (
        round(acc._pending_pipecat_tts_processing_s * 1000)
        if acc._pending_pipecat_tts_processing_s
        else None
    )

    acc.latency_turns.append(
        LatencyTurn(
            turn_index=acc._turn_index,
            node=acc._latency_node,
            ttfb_ms=ttfb_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            bot_started_ms=acc._rel_ms(acc._bot_started_ns),
            user_stopped_ms=acc._rel_ms(acc._user_stopped_ns),
            user_started_ms=acc._rel_ms(acc._user_started_ns),
        )
    )

    acc._turn_index += 1
    acc._user_stopped_ns = 0
    acc._llm_started_ns = 0
    acc._tts_started_ns = 0
    acc._bot_started_ns = 0
    acc._latency_node = None
    acc._user_started_ns = 0
    acc._pending_pipecat_llm_ttfb_s = 0.0
    acc._pending_pipecat_tts_ttfb_s = 0.0
    acc._pending_pipecat_llm_processing_s = 0.0
    acc._pending_pipecat_tts_processing_s = 0.0

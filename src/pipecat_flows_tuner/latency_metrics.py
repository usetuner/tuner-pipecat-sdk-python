"""Transform concern: compute latency metrics from collected turn timings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import LatencyTurn

if TYPE_CHECKING:
    from .accumulator import FlowsAccumulator


def flush_latency_turn(acc: FlowsAccumulator) -> None:
    if acc._tts_started_ns and acc._tts_started_ns <= acc._bot_started_ns:
        ttfb_ms = acc._ns_to_ms(acc._user_stopped_ns, acc._tts_started_ns)
        tts_ms = acc._ns_to_ms(acc._tts_started_ns, acc._bot_started_ns)
    else:
        ttfb_ms = acc._ns_to_ms(acc._user_stopped_ns, acc._bot_started_ns)
        tts_ms = 0

    acc.latency_turns.append(
        LatencyTurn(
            turn_index=acc._turn_index,
            node=acc._latency_node,
            ttfb_ms=ttfb_ms,
            llm_ms=acc._ns_to_ms(acc._user_stopped_ns, acc._llm_started_ns),
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

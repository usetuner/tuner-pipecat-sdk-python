"""Transform concern: assemble the API transcript from the live, event-sourced turn list.

The accumulator builds ``live_turns`` incrementally from the frame stream (the event-sourced
mirror of pipecat's own conversation construction): each turn is materialized when its frames
arrive, in arrival order. This module just renders that ordered list into ``TranscriptSegment``s,
pulling per-turn timing/latency from the linked SpeechSegment/LatencyMeasurement (the latency
substrate) by segment id — never by text matching. There is no reconstruction, no injected/ghost
filtering: a turn exists iff its frames occurred.

A user row groups exactly the transcriptions between two UserStoppedSpeaking boundaries — the same
unit pipecat's aggregator commits as one user message — so the row grouping matches pipecat exactly
(no silence-gap heuristic).
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from .models import LatencyMeasurement, LiveTurn, SpeechSegment, ToolInfo, TranscriptSegment

if TYPE_CHECKING:
    from .accumulator import CallAccumulator


def build_segment_metadata(*, interrupted: bool = False, **extra: Any) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "interrupted": interrupted,
        **{k: v for k, v in extra.items() if v is not None},
    }


def parse_json_value(value: Any) -> Any:
    """Parse `value` as JSON if it's a string; return it unchanged if parsing fails or it isn't a string."""
    try:
        return json.loads(value) if isinstance(value, str) else value
    except Exception:
        return value


def build_user_segment(
    text: str,
    *,
    start_ms: int,
    end_ms: int,
    interrupted: bool,
    stt_ms: int | None = None,
    eou_ms: int | None = None,
    eou_confidence: float | None = None,
    eou_reason: str | None = None,
    fragments: int = 1,
    turn_index: int | None = None,
) -> TranscriptSegment:
    """Build one user row (one UserStoppedSpeaking-bounded turn)."""
    return TranscriptSegment(
        role="user",
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        metadata=build_segment_metadata(
            interrupted=interrupted,
            turn_index=turn_index,
            stt_node_ttfb=stt_ms,
            eou_delay=eou_ms,
            eou_confidence=eou_confidence,
            eou_reason=eou_reason,
            fragments=fragments if fragments > 1 else None,
        ),
    )


def build_agent_function_segment(
    tool_call: dict[str, Any], invocation_ms: int
) -> TranscriptSegment:
    """Build the transcript row for a tool invocation, formatted as `name(arg=value, ...)`."""
    function_name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments", "{}")
    parsed_args = parse_json_value(raw_args) or {}
    argument_items = parsed_args.items() if isinstance(parsed_args, dict) else []
    arg_str = ", ".join(f"{key}={value}" for key, value in argument_items)
    return TranscriptSegment(
        role="agent_function",
        text=f"{function_name}({arg_str})",
        start_ms=invocation_ms,
        end_ms=None,
        tool=ToolInfo(
            name=function_name,
            request_id=tool_call.get("id"),
            params=parsed_args if isinstance(parsed_args, dict) else {},
            start_ms=invocation_ms,
        ),
        metadata=build_segment_metadata(),
    )


def build_agent_result_segment(turn: LiveTurn, result_ms: int) -> TranscriptSegment:
    """Build the transcript row for a tool's result. Structured (dict) results are stored on
    `tool.result`; plain strings are stored as `text` instead."""
    parsed_result = parse_json_value(turn.result)
    is_structured = isinstance(parsed_result, dict)
    return TranscriptSegment(
        role="agent_result",
        text=None if is_structured else (parsed_result if isinstance(parsed_result, str) else None),
        start_ms=result_ms,
        end_ms=None,
        tool=ToolInfo(
            name=turn.function_name,
            request_id=turn.tool_call_id or None,
            result=parsed_result if is_structured else None,
            start_ms=result_ms,
        ),
        metadata=build_segment_metadata(),
    )


def build_agent_text_segment(
    text: str,
    bot_seg: SpeechSegment | None,
    measurement: LatencyMeasurement | None,
    *,
    interrupted: bool,
    turn_index: int,
    fallback_ms: int = 0,
) -> TranscriptSegment:
    """Build the transcript row for a spoken agent turn, timed from `bot_seg` when the turn was
    voiced, or from `fallback_ms` (generation time) for a draft that was never spoken."""
    is_proactive = bool(
        (measurement and measurement.is_proactive) or (bot_seg and bot_seg.is_proactive)
    )
    e2e = measurement.e2e_ms if measurement else None
    interrupted_at_ms = (
        measurement.interrupted_at_ms
        if measurement and measurement.interrupted_at_ms is not None
        else (bot_seg.interrupted_at_ms if bot_seg else None)
    )
    # Voiced rows take their (final, latency-adjusted) timing from the bot segment. A generated-
    # but-never-voiced draft has no segment — time it at its generation moment so it stays in the
    # right place in the conversation (never 00:00).
    start_ms = bot_seg.start_ms if bot_seg else fallback_ms
    end_ms = (bot_seg.stop_ms if bot_seg and bot_seg.stop_ms is not None else None) or fallback_ms
    return TranscriptSegment(
        role="agent",
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        metadata=build_segment_metadata(
            e2e_latency=(e2e if e2e and e2e > 0 and not is_proactive else None),
            interrupted=interrupted,
            llm_node_ttft=measurement.llm_ms if measurement else None,
            tts_node_ttfb=measurement.ttfb_ms if measurement else None,
            node=bot_seg.node if bot_seg else None,
            turn_index=turn_index,
            interrupted_at_ms=interrupted_at_ms,
        ),
    )


def build_segments_from_turns(acc: CallAccumulator) -> list[TranscriptSegment]:
    """Render the live, ordered turn list into API transcript rows, joining each turn to its
    latency/speech data by segment id."""
    seg_by_id = {s.id: s for s in acc.speech_segments}
    meas_by_bot_id = {
        m.bot_segment_id: m for m in acc.latency_measurements if m.bot_segment_id is not None
    }

    out: list[TranscriptSegment] = []
    turn_index = 0
    last_agent_interrupted = False

    for turn in acc.live_turns:
        if turn.kind == "user":
            # One user row per UserStoppedSpeaking boundary — pipecat's exact user-message unit.
            # All transcriptions captured in the turn are joined (no silence-gap split/merge).
            text = " ".join(t for t, _ in turn.chunks).strip()
            if not text:
                continue
            out.append(
                build_user_segment(
                    text,
                    start_ms=turn.start_ms or (turn.chunks[0][1] if turn.chunks else 0),
                    end_ms=turn.end_ms if turn.end_ms is not None else 0,
                    interrupted=last_agent_interrupted,
                    stt_ms=turn.stt_ms,
                    eou_ms=turn.eou_ms,
                    eou_confidence=turn.eou_confidence,
                    eou_reason=turn.eou_reason,
                    fragments=len(turn.chunks),
                    turn_index=turn_index,
                )
            )
            turn_index += 1
            last_agent_interrupted = False

        elif turn.kind == "agent":
            bot_seg = seg_by_id.get(turn.bot_segment_id)
            meas = meas_by_bot_id.get(turn.bot_segment_id)
            # Agent text mirrors pipecat's UI render: the LLM-generated text. Falls back to the
            # voiced text for a templated/static reply with no LLM generation.
            text = turn.text.strip() or (bot_seg.spoken_text if bot_seg else "") or ""
            if not text:
                continue  # a response that only made a tool call → no agent text row
            interrupted = bool(
                (meas and meas.interrupted_at_ms is not None)
                or (bot_seg and bot_seg.interrupted_at_ms is not None)
            )
            out.append(
                build_agent_text_segment(
                    text,
                    bot_seg,
                    meas,
                    interrupted=interrupted,
                    turn_index=turn_index,
                    fallback_ms=turn.generated_ms or 0,
                )
            )
            last_agent_interrupted = interrupted
            turn_index += 1

        elif turn.kind == "agent_function":
            inv = acc.get_tool_invocation_ms(turn.tool_call_id) if turn.tool_call_id else None
            out.append(
                build_agent_function_segment(
                    {
                        "id": turn.tool_call_id,
                        "function": {"name": turn.function_name, "arguments": turn.arguments},
                    },
                    invocation_ms=inv if inv is not None else turn.occurrence_ms,
                )
            )

        elif turn.kind == "agent_result":
            comp = acc.get_tool_completion_ms(turn.tool_call_id) if turn.tool_call_id else None
            out.append(
                build_agent_result_segment(
                    turn, result_ms=comp if comp is not None else turn.occurrence_ms
                )
            )

    return out

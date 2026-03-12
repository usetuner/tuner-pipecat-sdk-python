"""Transform concern: enrich transcript messages into structured call segments for tuner api."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from .models import NodeInfo, ToolInfo, TranscriptSegment

if TYPE_CHECKING:
    from .accumulator import FlowsAccumulator


def build_segment_metadata(*, interrupted: bool = False, **extra: Any) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "interrupted": interrupted,
        **{k: v for k, v in extra.items() if v is not None},
    }


def parse_json_value(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except Exception:
        return value


def index_transitions_by_function(acc: FlowsAccumulator) -> dict[str, Any]:
    return {
        transition.trigger_function: transition
        for transition in acc.node_transitions
        if transition.trigger_function
    }


def calculate_user_interruptions(acc: FlowsAccumulator) -> dict[int, bool]:
    interrupted: dict[int, bool] = {}
    for idx, latency_turn in enumerate(acc.latency_turns):
        if idx == 0:
            interrupted[idx] = False
            continue
        prev_bot_stopped = acc.latency_turns[idx - 1].bot_stopped_ms or 0
        interrupted[idx] = bool(latency_turn.user_started_ms < prev_bot_stopped)
    return interrupted


def calculate_agent_interruptions(acc: FlowsAccumulator) -> dict[int, bool]:
    interrupted: dict[int, bool] = {}
    for idx, latency_turn in enumerate(acc.latency_turns):
        if idx + 1 < len(acc.latency_turns):
            next_user_started = acc.latency_turns[idx + 1].user_started_ms
            interrupted[idx] = bool(next_user_started < (latency_turn.bot_stopped_ms or 0))
        else:
            interrupted[idx] = False
    return interrupted


def collect_consecutive_user_messages(
    messages: list[dict[str, Any]], start_idx: int
) -> tuple[list[dict[str, Any]], int]:
    grouped_messages: list[dict[str, Any]] = []
    idx = start_idx
    while idx < len(messages) and messages[idx].get("role") == "user":
        grouped_messages.append(messages[idx])
        idx += 1
    return grouped_messages, idx


def build_user_segment(
    grouped_messages: list[dict[str, Any]],
    turn: Any | None,
    user_index: int,
    user_interrupted: dict[int, bool],
) -> TranscriptSegment:
    text = " ".join(message.get("content", "") for message in grouped_messages).strip()
    return TranscriptSegment(
        role="user",
        text=text,
        start_ms=turn.user_started_ms if turn else 0,
        end_ms=turn.user_stopped_ms if turn else 0,
        metadata=build_segment_metadata(
            interrupted=user_interrupted.get(user_index, False),
            node=turn.node if turn else None,
            turn_index=turn.turn_index if turn else None,
            fragments=len(grouped_messages) if len(grouped_messages) > 1 else None,
        ),
    )


def build_agent_function_segment(
    message: dict[str, Any], transitions_by_function: dict[str, Any]
) -> TranscriptSegment:
    tool_call = message["tool_calls"][0]
    function_name = tool_call["function"]["name"]
    raw_args = tool_call["function"].get("arguments", "{}")
    parsed_args = parse_json_value(raw_args) or {}
    transition = transitions_by_function.get(function_name)
    argument_items = parsed_args.items() if isinstance(parsed_args, dict) else []
    arg_str = ", ".join(f"{key}={value}" for key, value in argument_items)
    return TranscriptSegment(
        role="agent_function",
        text=f"{function_name}({arg_str})",
        start_ms=transition.timestamp_ms if transition else 0,
        end_ms=transition.timestamp_ms if transition else 0,
        tool=ToolInfo(
            name=function_name,
            request_id=tool_call.get("id"),
            params=parsed_args if isinstance(parsed_args, dict) else {},
        ),
        metadata=build_segment_metadata(node=transition.from_node if transition else None),
    )


def find_matching_tool_call(
    messages: list[dict[str, Any]], tool_call_id: str
) -> dict[str, Any] | None:
    return next(
        (
            tool_call
            for message in messages
            if message.get("role") == "assistant" and "tool_calls" in message
            for tool_call in message["tool_calls"]
            if tool_call.get("id") == tool_call_id
        ),
        None,
    )


def build_agent_result_segments(
    message: dict[str, Any],
    messages: list[dict[str, Any]],
    transitions_by_function: dict[str, Any],
) -> list[TranscriptSegment]:
    tool_call_id = message.get("tool_call_id", "")
    matched_tool_call = find_matching_tool_call(messages, tool_call_id)
    function_name = matched_tool_call["function"]["name"] if matched_tool_call else None
    transition = transitions_by_function.get(function_name) if function_name else None
    parsed_result = parse_json_value(message.get("content"))

    result_segments = [
        TranscriptSegment(
            role="agent_result",
            text=(
                json.dumps(parsed_result, default=str)
                if parsed_result is not None
                else message.get("content", "")
            ),
            start_ms=transition.timestamp_ms if transition else 0,
            end_ms=transition.timestamp_ms if transition else 0,
            tool=ToolInfo(
                name=function_name,
                request_id=tool_call_id or None,
                result=(
                    parsed_result
                    if isinstance(parsed_result, dict)
                    else {"value": parsed_result}
                ),
            ),
            metadata=build_segment_metadata(
                node=transition.from_node if transition else None,
                triggered_transition_to=transition.to_node if transition else None,
            ),
        )
    ]

    if transition:
        result_segments.append(
            TranscriptSegment(
                role="node_transition",
                text=f"{transition.from_node} → {transition.to_node}",
                start_ms=transition.timestamp_ms,
                end_ms=transition.timestamp_ms,
                node=NodeInfo(
                    from_node=transition.from_node,
                    to=transition.to_node,
                    reason=function_name,
                ),
                metadata=build_segment_metadata(
                    trigger_args=transition.trigger_args,
                    state_snapshot=transition.state_snapshot,
                    functions_available=transition.functions_available,
                ),
            )
        )
    return result_segments


def build_agent_text_segment(
    acc: FlowsAccumulator,
    message: dict[str, Any],
    turn: Any | None,
    assistant_index: int,
    agent_interrupted: dict[int, bool],
) -> TranscriptSegment:
    end_to_end_latency = ((turn.bot_started_ms - turn.user_stopped_ms) or None) if turn else None
    return TranscriptSegment(
        role="agent",
        text=message.get("content", ""),
        start_ms=turn.bot_started_ms if turn else 0,
        end_ms=(
            turn.bot_stopped_ms
            if turn and turn.bot_stopped_ms is not None
            else acc._rel_ms(acc.call_end_abs_ns)
        ),
        metadata=build_segment_metadata(
            e2e_latency=(
                end_to_end_latency
                if end_to_end_latency and end_to_end_latency > 0
                else None
            ),
            interrupted=agent_interrupted.get(assistant_index, False),
            llm_node_ttft=turn.llm_ms if turn else None,
            tts_node_ttfb=turn.ttfb_ms if turn else None,
            node=turn.node if turn else None,
            turn_index=turn.turn_index if turn else None,
        ),
    )


def build_initial_transition_segment(acc: FlowsAccumulator) -> TranscriptSegment | None:
    initial_transition = next(
        (transition for transition in acc.node_transitions if transition.trigger_function is None),
        None,
    )
    if not initial_transition:
        return None

    return TranscriptSegment(
        role="node_transition",
        text=f"→ {initial_transition.to_node}",
        start_ms=initial_transition.timestamp_ms,
        end_ms=initial_transition.timestamp_ms,
        node=NodeInfo(from_node="", to=initial_transition.to_node, reason=""),
        metadata=build_segment_metadata(
            state_snapshot=initial_transition.state_snapshot,
            functions_available=initial_transition.functions_available,
        ),
    )


def enrich_transcript(
    acc: FlowsAccumulator, messages: list[dict[str, Any]]
) -> list[TranscriptSegment]:
    transitions_by_function = index_transitions_by_function(acc)
    user_interrupted = calculate_user_interruptions(acc)
    agent_interrupted = calculate_agent_interruptions(acc)

    result: list[TranscriptSegment] = []
    message_idx = 0
    user_idx = 0
    assistant_idx = 0

    while message_idx < len(messages):
        message = messages[message_idx]
        role = message.get("role", "")

        if role == "system":
            message_idx += 1
            continue

        if role == "user":
            grouped_messages, message_idx = collect_consecutive_user_messages(messages, message_idx)
            user_turn = acc.latency_turns[user_idx] if user_idx < len(acc.latency_turns) else None
            result.append(
                build_user_segment(grouped_messages, user_turn, user_idx, user_interrupted)
            )
            user_idx += 1
            continue

        if role == "assistant" and "tool_calls" in message:
            result.append(build_agent_function_segment(message, transitions_by_function))
            message_idx += 1
            continue

        if role == "tool":
            result.extend(
                build_agent_result_segments(message, messages, transitions_by_function)
            )
            message_idx += 1
            continue

        if role == "assistant":
            assistant_turn = (
                acc.latency_turns[assistant_idx] if assistant_idx < len(acc.latency_turns) else None
            )
            result.append(
                build_agent_text_segment(
                    acc=acc,
                    message=message,
                    turn=assistant_turn,
                    assistant_index=assistant_idx,
                    agent_interrupted=agent_interrupted,
                )
            )
            assistant_idx += 1

        message_idx += 1

    initial_transition_segment = build_initial_transition_segment(acc)
    if initial_transition_segment:
        result.insert(0, initial_transition_segment)

    return result

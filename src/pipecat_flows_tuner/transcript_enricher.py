"""Transform concern: enrich transcript messages into structured call segments for tuner api."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from .models import ToolInfo, TranscriptSegment

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


def calculate_user_interruptions(acc: FlowsAccumulator) -> dict[int, bool]:
    """User turn idx is an interruption if the previous turn was_interrupted by the user."""
    interrupted: dict[int, bool] = {}
    for idx in range(len(acc.latency_turns)):
        if idx == 0:
            interrupted[idx] = False
        else:
            prev_turn = acc.latency_turns[idx - 1]
            interrupted[idx] = bool(prev_turn.was_interrupted)
    return interrupted


def calculate_agent_interruptions(acc: FlowsAccumulator) -> dict[int, bool]:
    """Agent turn idx was interrupted if TurnTrackingObserver reported was_interrupted=True."""
    interrupted: dict[int, bool] = {}
    for idx, turn in enumerate(acc.latency_turns):
        interrupted[idx] = bool(turn.was_interrupted)
    return interrupted


def collect_consecutive_assistant_messages(
    messages: list[dict[str, Any]], start_idx: int
) -> tuple[list[dict[str, Any]], int]:
    """Collect consecutive plain assistant text messages (no tool_calls)."""
    grouped: list[dict[str, Any]] = []
    idx = start_idx
    while idx < len(messages):
        msg = messages[idx]
        if msg.get("role") == "assistant" and "tool_calls" not in msg:
            grouped.append(msg)
            idx += 1
        else:
            break
    return grouped, idx


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
    tool_call: dict[str, Any],
    invocation_ms: int,
) -> TranscriptSegment:
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


def build_agent_result_segment(
    acc: FlowsAccumulator,
    message: dict[str, Any],
    messages: list[dict[str, Any]],
) -> TranscriptSegment:
    tool_call_id = message.get("tool_call_id", "")
    matched_tool_call = find_matching_tool_call(messages, tool_call_id)
    function_name = matched_tool_call["function"]["name"] if matched_tool_call else None
    parsed_result = parse_json_value(message.get("content"))
    completion_ms = acc.get_tool_completion_ms(tool_call_id) if tool_call_id else None
    result_ms = completion_ms if completion_ms is not None else 0

    return TranscriptSegment(
        role="agent_result",
        text=(
            json.dumps(parsed_result, default=str)
            if parsed_result is not None
            else message.get("content", "")
        ),
        start_ms=result_ms,
        end_ms=None,
        tool=ToolInfo(
            name=function_name,
            request_id=tool_call_id or None,
            result=(
                parsed_result
                if isinstance(parsed_result, dict)
                else {"value": parsed_result}
            ),
            start_ms=result_ms,
        ),
        metadata=build_segment_metadata(),
    )


def build_agent_text_segment(
    acc: FlowsAccumulator,
    messages: list[dict[str, Any]],
    turn: Any | None,
    assistant_index: int,
    agent_interrupted: dict[int, bool],
) -> TranscriptSegment:
    end_to_end_latency = ((turn.bot_started_ms - turn.user_stopped_ms) or None) if turn else None
    text = " ".join(m.get("content", "") for m in messages).strip()
    node = (turn.bot_node or turn.node) if turn else None
    return TranscriptSegment(
        role="agent",
        text=text,
        start_ms=turn.bot_started_ms if turn else 0,
        end_ms=turn.bot_stopped_ms if turn and turn.bot_stopped_ms is not None else 0,
        metadata=build_segment_metadata(
            e2e_latency=(
                end_to_end_latency
                if end_to_end_latency and end_to_end_latency > 0
                else None
            ),
            interrupted=agent_interrupted.get(assistant_index, False),
            llm_node_ttft=turn.llm_ms if turn else None,
            tts_node_ttfb=turn.ttfb_ms if turn else None,
            node=node,
            turn_index=turn.turn_index if turn else None,
        ),
    )


def find_spoken_assistant_message_indices(messages: list[dict[str, Any]]) -> set[int]:
    """Return the set of context message indices that are 'spoken' (final) assistant text.

    The last plain assistant text before each user message (or end of context) is the one
    that was actually spoken. All earlier ones in the same window are ghost messages
    (generated but not spoken due to immediate tool-call-triggered node transitions).
    """
    last_per_window: dict[int, int] = {}
    trailing_assistant_indices: set[int] = set()

    # Treat the final contiguous plain-assistant block at end-of-context as spoken.
    i = len(messages) - 1
    while i >= 0:
        role = messages[i].get("role", "")
        if role == "system":
            i -= 1
            continue
        if role == "assistant" and "tool_calls" not in messages[i]:
            trailing_assistant_indices.add(i)
            i -= 1
            continue
        break

    user_idx = -1
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "system":
            continue
        if role == "user":
            user_idx += 1
        elif role == "assistant" and "tool_calls" not in msg:
            last_per_window[user_idx] = i  # overwrite → keeps the last one
    return set(last_per_window.values()) | trailing_assistant_indices


def enrich_transcript(
    acc: FlowsAccumulator, messages: list[dict[str, Any]]
) -> list[TranscriptSegment]:
    user_interrupted = calculate_user_interruptions(acc)
    agent_interrupted = calculate_agent_interruptions(acc)

    result: list[TranscriptSegment] = []
    message_idx = 0
    user_idx = 0
    assistant_idx = 0
    latency_turn_idx = 0
    spoken_indices = find_spoken_assistant_message_indices(messages)

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
            for tool_call in message.get("tool_calls", []):
                tool_call_id = tool_call.get("id")
                invocation_ms = (
                    acc.get_tool_invocation_ms(tool_call_id) or 0 if tool_call_id else 0
                )
                result.append(
                    build_agent_function_segment(
                        tool_call=tool_call,
                        invocation_ms=invocation_ms,
                    )
                )
            message_idx += 1
            continue

        if role == "tool":
            result.append(
                build_agent_result_segment(
                    acc=acc,
                    message=message,
                    messages=messages,
                )
            )
            message_idx += 1
            continue

        if role == "assistant":
            grouped_messages, message_idx = collect_consecutive_assistant_messages(
                messages, message_idx
            )
            final_msg_idx = message_idx - 1  # last message in the consecutive group
            is_preamble = user_idx == 0
            if (
                final_msg_idx in spoken_indices
                and latency_turn_idx < len(acc.latency_turns)
                and not is_preamble
            ):
                assistant_turn = (
                    acc.latency_turns[latency_turn_idx]
                    if latency_turn_idx < len(acc.latency_turns)
                    else None
                )
                latency_turn_idx += 1
            else:
                assistant_turn = None  # ghost — not actually spoken
            result.append(
                build_agent_text_segment(
                    acc=acc,
                    messages=grouped_messages,
                    turn=assistant_turn,
                    assistant_index=latency_turn_idx - 1 if assistant_turn else -1,
                    agent_interrupted=agent_interrupted,
                )
            )
            assistant_idx += 1
            continue

        message_idx += 1

    return result

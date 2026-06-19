"""Transform concern: enrich transcript messages into structured call segments for tuner api.

The accumulator emits an append-only stream of SpeechSegments (user and bot) plus
LatencyMeasurements linked to them by id. This module walks the LLM context messages
(the source of truth for text and tool structure) and pairs each one with its segment
and measurement by id-ordered consumption — no positional turn-index arithmetic.

Two consecutive user utterances are merged into one transcript row only when the silence
between them is below ``merge_gap_ms``; a longer gap (e.g. the user waiting on an
unresponsive agent) stays as separate rows so the silence is visible.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Any

from .models import LatencyMeasurement, SpeechSegment, ToolInfo, TranscriptSegment

if TYPE_CHECKING:
    from .accumulator import CallAccumulator

DEFAULT_MERGE_GAP_MS = 1500


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for matching an assistant context
    message against the text the bot actually voiced (TTS), which differs in punctuation."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


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
    *,
    start_ms: int,
    end_ms: int,
    interrupted: bool,
    stt_ms: int | None = None,
    turn_index: int | None = None,
) -> TranscriptSegment:
    """Build one user row from one-or-more user messages and their merged speech window."""
    text = " ".join(message.get("content", "") for message in grouped_messages).strip()
    return TranscriptSegment(
        role="user",
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        metadata=build_segment_metadata(
            interrupted=interrupted,
            turn_index=turn_index,
            stt_node_ttfb=stt_ms,
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
    acc: CallAccumulator,
    message: dict[str, Any],
    messages: list[dict[str, Any]],
    bot_started_ms: int = 0,
    user_stopped_ms: int = 0,
) -> TranscriptSegment:
    tool_call_id = message.get("tool_call_id", "")
    matched_tool_call = find_matching_tool_call(messages, tool_call_id)
    function_name = matched_tool_call["function"]["name"] if matched_tool_call else None
    parsed_result = parse_json_value(message.get("content", ""))
    is_structured = isinstance(parsed_result, dict)
    completion_ms = acc.get_tool_completion_ms(tool_call_id) if tool_call_id else None
    result_ms = completion_ms if completion_ms is not None else 0
    # FunctionCallResultFrame can arrive late (after the bot spoke). Fall back to the
    # user_stopped time of the response this tool belongs to when the recorded time is
    # missing or implausibly after the bot started speaking.
    if result_ms == 0 or (bot_started_ms > 0 and result_ms > bot_started_ms):
        result_ms = user_stopped_ms

    return TranscriptSegment(
        role="agent_result",
        text=None if is_structured else parsed_result,
        start_ms=result_ms,
        end_ms=None,
        tool=ToolInfo(
            name=function_name,
            request_id=tool_call_id or None,
            result=parsed_result if is_structured else None,
            start_ms=result_ms,
        ),
        metadata=build_segment_metadata(),
    )


def build_agent_text_segment(
    grouped_messages: list[dict[str, Any]],
    bot_seg: SpeechSegment | None,
    measurement: LatencyMeasurement | None,
    *,
    interrupted: bool,
) -> TranscriptSegment:
    text = " ".join(m.get("content", "") for m in grouped_messages).strip()
    is_proactive = bool(
        (measurement and measurement.is_proactive) or (bot_seg and bot_seg.is_proactive)
    )
    e2e = measurement.e2e_ms if measurement else None
    interrupted_at_ms = (
        measurement.interrupted_at_ms
        if measurement and measurement.interrupted_at_ms is not None
        else (bot_seg.interrupted_at_ms if bot_seg else None)
    )
    return TranscriptSegment(
        role="agent",
        text=text,
        start_ms=bot_seg.start_ms if bot_seg else 0,
        end_ms=bot_seg.stop_ms if bot_seg and bot_seg.stop_ms is not None else 0,
        metadata=build_segment_metadata(
            e2e_latency=(e2e if e2e and e2e > 0 and not is_proactive else None),
            interrupted=interrupted,
            llm_node_ttft=measurement.llm_ms if measurement else None,
            tts_node_ttfb=measurement.ttfb_ms if measurement else None,
            node=bot_seg.node if bot_seg else None,
            turn_index=bot_seg.id if bot_seg else None,
            interrupted_at_ms=interrupted_at_ms,
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


def _spoken_text_matches(group_norm: str, spoken_text: str | None) -> bool:
    """True if an assistant message group matches text the bot actually voiced. An empty
    spoken_text means no TTS text was captured for that window — fall back to assuming
    spoken (positional)."""
    st = _normalize(spoken_text or "")
    if not st:
        return True
    return group_norm in st or st in group_norm


def find_spoken_assistant_message_indices_via_tts(
    messages: list[dict[str, Any]], bot_segs: list[SpeechSegment]
) -> set[int]:
    """Authoritative spoken-detection: an assistant text was spoken iff it matches the next
    bot segment's actually-voiced text (TTS). Pre-seeded instructions and superseded drafts
    match nothing and are correctly treated as ghosts; a real speak→tool→speak first
    utterance matches its own segment and is no longer wrongly ghosted."""
    spoken: set[int] = set()
    cursor = 0
    idx = 0
    while idx < len(messages):
        msg = messages[idx]
        if msg.get("role") == "assistant" and "tool_calls" not in msg:
            group, nxt = collect_consecutive_assistant_messages(messages, idx)
            group_norm = _normalize(" ".join(m.get("content", "") for m in group))
            if cursor < len(bot_segs) and _spoken_text_matches(
                group_norm, bot_segs[cursor].spoken_text
            ):
                spoken.add(nxt - 1)  # last message of the group is the spoken one
                cursor += 1
            idx = nxt
        else:
            idx += 1
    return spoken


def enrich_transcript(
    acc: CallAccumulator,
    messages: list[dict[str, Any]],
    merge_gap_ms: int = DEFAULT_MERGE_GAP_MS,
) -> list[TranscriptSegment]:
    # Real user utterances, in order. Proactive (ghost) user segments preceding the
    # greeting are excluded — they have no user message dict to align to.
    user_segs = [s for s in acc.speech_segments if s.speaker == "user" and not s.is_proactive]
    bot_segs = [s for s in acc.speech_segments if s.speaker == "bot"]
    meas_by_bot_id = {
        m.bot_segment_id: m for m in acc.latency_measurements if m.bot_segment_id is not None
    }
    seg_by_id = {s.id: s for s in acc.speech_segments}

    # Flatten per-utterance user speech windows, in order: (start, stop, segment, is_first).
    # A coalesced turn (several utterances before a bot reply) is ONE segment with several
    # windows — flattening gives each real utterance its own timing, so context user
    # messages map 1:1 to windows regardless of how the accumulator grouped them into turns.
    user_windows: list[tuple[int, int | None, SpeechSegment, bool]] = []
    for s in user_segs:
        wins = s.windows or [[s.start_ms, s.stop_ms]]
        for i, w in enumerate(wins):
            user_windows.append((w[0] or 0, w[1], s, i == 0))

    # Prefer the authoritative spoken-detection when the bot's actually-voiced text (TTS)
    # was captured; otherwise fall back to the positional last-in-window heuristic.
    if any(s.spoken_text for s in bot_segs):
        spoken_indices = find_spoken_assistant_message_indices_via_tts(messages, bot_segs)
    else:
        spoken_indices = find_spoken_assistant_message_indices(messages)

    result: list[TranscriptSegment] = []
    message_idx = 0
    win_cursor = 0
    bot_cursor = 0
    seen_user_message = False
    last_agent_interrupted = False  # the user interrupts when the prior agent turn was cut off

    while message_idx < len(messages):
        message = messages[message_idx]
        role = message.get("role", "")

        if role == "system":
            message_idx += 1
            continue

        if role == "user":
            seen_user_message = True
            grouped_messages, message_idx = collect_consecutive_user_messages(messages, message_idx)
            n = len(grouped_messages)
            # Each user message maps to the next speech window (1:1 with real utterances).
            group_wins = user_windows[win_cursor : win_cursor + n]
            win_cursor += len(group_wins)

            # Group the messages into rows, starting a new row wherever the silence between
            # consecutive utterances is large enough (a real pause) — else merge fragments.
            rows: list[tuple[list[dict[str, Any]], list[tuple[int, int | None, Any, bool]]]] = []
            cur_msgs: list[dict[str, Any]] = []
            cur_wins: list[tuple[int, int | None, Any, bool]] = []
            for i, msg in enumerate(grouped_messages):
                win = group_wins[i] if i < len(group_wins) else None
                if cur_wins and win is not None:
                    prev_stop = cur_wins[-1][1] or 0
                    if prev_stop > 0 and win[0] > 0 and (win[0] - prev_stop) >= merge_gap_ms:
                        rows.append((cur_msgs, cur_wins))
                        cur_msgs, cur_wins = [], []
                cur_msgs.append(msg)
                if win is not None:
                    cur_wins.append(win)
            if cur_msgs:
                rows.append((cur_msgs, cur_wins))

            for row_msgs, row_wins in rows:
                if not row_wins:
                    # No speech window → no VAD/STT ever fired for this message. Real speech
                    # always produces a window (even mid-tool-call, on the active segment),
                    # so this is a developer-injected context message — drop it. (Only when
                    # windows exist at all; with none recorded we render everything untimed
                    # rather than drop a whole degenerate-capture transcript.)
                    if user_windows:
                        continue
                    result.append(
                        build_user_segment(
                            row_msgs, start_ms=0, end_ms=0, interrupted=last_agent_interrupted
                        )
                    )
                    last_agent_interrupted = False
                    continue
                first_seg = row_wins[0][2]
                result.append(
                    build_user_segment(
                        row_msgs,
                        start_ms=row_wins[0][0],
                        end_ms=row_wins[-1][1] or 0,
                        interrupted=last_agent_interrupted,
                        stt_ms=first_seg.stt_ms if row_wins[0][3] else None,
                        turn_index=first_seg.id,
                    )
                )
                last_agent_interrupted = False
            continue

        if role == "assistant" and "tool_calls" in message:
            # Tool calls belong to the upcoming bot response. FunctionCallInProgressFrame
            # arrives late at the observer (after the bot has spoken), so fall back to the
            # response's user_stopped time when the recorded invocation looks too late.
            up_bot = bot_segs[bot_cursor] if bot_cursor < len(bot_segs) else None
            up_meas = meas_by_bot_id.get(up_bot.id) if up_bot else None
            up_user = seg_by_id.get(up_meas.user_segment_id) if up_meas else None
            bot_started_ms = up_bot.start_ms if up_bot else 0
            user_stopped_ms = (up_user.stop_ms or 0) if up_user else 0
            for tool_call in message.get("tool_calls", []):
                tool_call_id = tool_call.get("id")
                invocation_ms = acc.get_tool_invocation_ms(tool_call_id) or 0 if tool_call_id else 0
                if invocation_ms == 0 or (bot_started_ms > 0 and invocation_ms > bot_started_ms):
                    invocation_ms = user_stopped_ms
                result.append(
                    build_agent_function_segment(tool_call=tool_call, invocation_ms=invocation_ms)
                )
            message_idx += 1
            continue

        if role == "tool":
            up_bot = bot_segs[bot_cursor] if bot_cursor < len(bot_segs) else None
            up_meas = meas_by_bot_id.get(up_bot.id) if up_bot else None
            up_user = seg_by_id.get(up_meas.user_segment_id) if up_meas else None
            result.append(
                build_agent_result_segment(
                    acc=acc,
                    message=message,
                    messages=messages,
                    bot_started_ms=up_bot.start_ms if up_bot else 0,
                    user_stopped_ms=(up_user.stop_ms or 0) if up_user else 0,
                )
            )
            message_idx += 1
            continue

        if role == "assistant":
            grouped_messages, message_idx = collect_consecutive_assistant_messages(
                messages, message_idx
            )
            # In preamble position (before any user turn) consecutive assistant messages
            # are a pre-seeded developer instruction followed by the real LLM response.
            # Only the last message in the group was generated (and spoken) by the LLM.
            if not seen_user_message and len(grouped_messages) > 1:
                grouped_messages = [grouped_messages[-1]]
            final_msg_idx = message_idx - 1  # last message in the consecutive group

            if final_msg_idx in spoken_indices and bot_cursor < len(bot_segs):
                bot_seg = bot_segs[bot_cursor]
                measurement = meas_by_bot_id.get(bot_seg.id)
                bot_cursor += 1
                # Genuine interruption only: the user cut in while the bot was speaking, which
                # sets interrupted_at_ms. TurnTrackingObserver also reports was_interrupted=True
                # when the pipeline ends (a clean end_call hangup) — exclude that false positive.
                interrupted = bool(
                    (measurement and measurement.interrupted_at_ms is not None)
                    or bot_seg.interrupted_at_ms is not None
                )
                last_agent_interrupted = interrupted
            else:
                bot_seg = None  # ghost — generated but not spoken
                measurement = None
                interrupted = False
            result.append(
                build_agent_text_segment(
                    grouped_messages, bot_seg, measurement, interrupted=interrupted
                )
            )
            continue

        message_idx += 1

    return result

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
    """Lowercase, drop punctuation, collapse whitespace — for matching context message
    text against STT transcriptions, which may differ in punctuation/casing."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def consume_transcriptions(
    group_norm: str, norm_transcriptions: list[tuple[str, int]], cursor: int
) -> tuple[int, int | None]:
    """Greedily match a user message group against transcriptions from ``cursor``.

    Returns ``(matched_count, start_ms)``. ``matched_count == 0`` means no real speech
    matched this group — i.e. it was injected into the context by the developer rather
    than spoken. The aggregator joins consecutive transcriptions into one message, so we
    accumulate transcriptions while they remain a prefix of (or contain) the group text.
    """
    if not group_norm or cursor >= len(norm_transcriptions):
        return 0, None
    start_ms = norm_transcriptions[cursor][1]
    concat = ""
    count = 0
    while cursor + count < len(norm_transcriptions):
        nxt = norm_transcriptions[cursor + count][0]
        trial = f"{concat} {nxt}".strip()
        if group_norm.startswith(trial) or trial.startswith(group_norm):
            concat = trial
            count += 1
            if len(concat) >= len(group_norm):
                break
        else:
            break
    return count, (start_ms if count else None)


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
    segs: list[SpeechSegment],
    *,
    interrupted: bool,
    fallback_start_ms: int = 0,
) -> TranscriptSegment:
    """Build one user row from one-or-more user messages merged onto their segments.

    ``fallback_start_ms`` times un-turned utterances (real speech with no turn segment,
    e.g. the user speaking during a tool call) from their STT transcription.
    """
    text = " ".join(message.get("content", "") for message in grouped_messages).strip()
    first = segs[0] if segs else None
    last = segs[-1] if segs else None
    return TranscriptSegment(
        role="user",
        text=text,
        start_ms=first.start_ms if first else fallback_start_ms,
        end_ms=(last.stop_ms or 0) if last else 0,
        metadata=build_segment_metadata(
            interrupted=interrupted,
            node=first.node if first else None,
            turn_index=first.id if first else None,
            stt_node_ttfb=first.stt_ms if first else None,
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
    norm_transcriptions = [(_normalize(t), ms) for t, ms in acc.user_transcriptions]

    # Prefer the authoritative spoken-detection when the bot's actually-voiced text (TTS)
    # was captured; otherwise fall back to the positional last-in-window heuristic.
    if any(s.spoken_text for s in bot_segs):
        spoken_indices = find_spoken_assistant_message_indices_via_tts(messages, bot_segs)
    else:
        spoken_indices = find_spoken_assistant_message_indices(messages)
    # Spoken assistant messages map 1:1, in order, to bot segments. Record which user
    # segment each response answers so we can detect injected user messages (a developer
    # injected {"role":"user"} with no speech, whose bot reply answers an earlier segment).
    spoken_sorted = sorted(spoken_indices)
    answered_user_id_by_msg: dict[int, int | None] = {}
    for k, msg_idx in enumerate(spoken_sorted):
        meas = meas_by_bot_id.get(bot_segs[k].id) if k < len(bot_segs) else None
        answered_user_id_by_msg[msg_idx] = meas.user_segment_id if meas else None

    def following_response_answer(after_idx: int) -> int | None:
        for msg_idx in spoken_sorted:
            if msg_idx >= after_idx:
                return answered_user_id_by_msg.get(msg_idx)
        return None

    result: list[TranscriptSegment] = []
    message_idx = 0
    user_cursor = 0
    bot_cursor = 0
    trans_cursor = 0
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
            candidate_segs = user_segs[user_cursor : user_cursor + n]
            answered = following_response_answer(message_idx)
            candidate_ids = {s.id for s in candidate_segs}

            # Does this group's bot reply (if any) answer a segment outside this group?
            # That happens both for developer-injected messages AND for real utterances
            # the user spoke without a turn-start firing (e.g. while a tool ran).
            seg_injected = answered is not None and answered != -1 and answered not in candidate_ids

            # Authoritative real-vs-injected signal: was this text actually transcribed?
            group_norm = _normalize(" ".join(m.get("content", "") for m in grouped_messages))
            matched, trans_start = consume_transcriptions(
                group_norm, norm_transcriptions, trans_cursor
            )
            # Drop only when both signals agree it is not real speech: no transcription
            # matched AND no turn segment of its own. (Without transcriptions captured,
            # never drop — degrade to rendering everything.)
            if norm_transcriptions and matched == 0 and seg_injected:
                last_agent_interrupted = False
                continue
            trans_cursor += matched

            if candidate_segs and not seg_injected:
                # Turned utterance(s): rich segment timing. Split into rows wherever the
                # silence gap between fragments is large enough (keeps a real pause visible).
                row_msgs: list[dict[str, Any]] = []
                row_segs: list[SpeechSegment] = []
                for msg, seg in zip(grouped_messages, candidate_segs, strict=False):
                    if row_segs:
                        prev_stop = row_segs[-1].stop_ms or 0
                        gap = seg.start_ms - prev_stop
                        if prev_stop > 0 and seg.start_ms > 0 and gap >= merge_gap_ms:
                            result.append(
                                build_user_segment(
                                    row_msgs, row_segs, interrupted=last_agent_interrupted
                                )
                            )
                            last_agent_interrupted = False
                            row_msgs, row_segs = [], []
                    row_msgs.append(msg)
                    row_segs.append(seg)
                if row_segs:
                    result.append(
                        build_user_segment(row_msgs, row_segs, interrupted=last_agent_interrupted)
                    )
                    last_agent_interrupted = False
                user_cursor += len(candidate_segs)
            else:
                # Un-turned real utterance (spoke during a tool call / interruption) or a
                # degenerate no-segment case: render one row, timed from its transcription.
                result.append(
                    build_user_segment(
                        grouped_messages,
                        [],
                        interrupted=last_agent_interrupted,
                        fallback_start_ms=trans_start or 0,
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
                interrupted = bool(
                    (measurement and measurement.was_interrupted) or bot_seg.interrupted
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

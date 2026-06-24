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
    turn_index: int,
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
            turn_index=turn_index,
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


# A flattened per-utterance user window: (start_ms, stop_ms, segment, is_segment_first_window).
UserWindow = tuple[int, int | None, SpeechSegment, bool]


def _upcoming_response_anchor(
    bot_segs: list[SpeechSegment],
    bot_cursor: int,
    meas_by_bot_id: dict[int, LatencyMeasurement],
    seg_by_id: dict[int, SpeechSegment],
) -> tuple[int, int]:
    """``(bot_started_ms, user_stopped_ms)`` of the response a tool call belongs to — used as
    the late-frame fallback when a recorded tool time is missing or implausibly late."""
    bot = bot_segs[bot_cursor] if bot_cursor < len(bot_segs) else None
    meas = meas_by_bot_id.get(bot.id) if bot else None
    user = seg_by_id.get(meas.user_segment_id) if meas else None
    return (bot.start_ms if bot else 0, (user.stop_ms or 0) if user else 0)


def _split_user_rows(
    grouped_messages: list[dict[str, Any]],
    group_wins: list[UserWindow],
    merge_gap_ms: int,
) -> list[tuple[list[dict[str, Any]], int | None, int | None, int | None]]:
    """Pair each user message with its speech window and split into rows wherever the silence
    between consecutive utterances is large enough (a real pause) — else merge fragments.

    Returns ``(row_messages, start_ms, end_ms, stt_ms)`` per row. ``start_ms is None`` means
    the row had no window (no VAD ever fired → developer-injected, not real speech).
    """

    def finalize(msgs: list[dict[str, Any]], wins: list[UserWindow]):
        if not wins:
            return (msgs, None, None, None)
        stt = wins[0][2].stt_ms if wins[0][3] else None  # stt only on the segment's first window
        return (msgs, wins[0][0], wins[-1][1] or 0, stt)

    rows = []
    cur_msgs: list[dict[str, Any]] = []
    cur_wins: list[UserWindow] = []
    for i, msg in enumerate(grouped_messages):
        win = group_wins[i] if i < len(group_wins) else None
        if cur_wins and win is not None:
            prev_stop = cur_wins[-1][1] or 0
            if prev_stop > 0 and win[0] > 0 and (win[0] - prev_stop) >= merge_gap_ms:
                rows.append(finalize(cur_msgs, cur_wins))
                cur_msgs, cur_wins = [], []
        cur_msgs.append(msg)
        if win is not None:
            cur_wins.append(win)
    if cur_msgs:
        rows.append(finalize(cur_msgs, cur_wins))
    return rows


def _drop_injected_user_messages(
    grouped_messages: list[dict[str, Any]],
    norm_transcriptions: list[tuple[str, int]],
    trans_cursor: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep only user messages that were actually transcribed, consuming transcriptions in
    order. Real speech produces an STT transcription; a developer-injected ``{"role":"user"}``
    message matches none and is dropped — regardless of position (a leading kickoff, a
    mid-call silence handler, etc.). This is the user-side mirror of the bot-side TTS
    ``spoken_text`` matching, and the authoritative real-vs-injected signal.

    Matching uses the same normalized containment rule as the TTS side, since the aggregator
    builds the context user message from the very transcription(s) we are matching against.
    Returns ``(kept_messages, new_trans_cursor)``.
    """
    kept: list[dict[str, Any]] = []
    cursor = trans_cursor
    for msg in grouped_messages:
        msg_norm = _normalize(msg.get("content", ""))
        matched_at = -1
        for i in range(cursor, len(norm_transcriptions)):
            trans_norm = norm_transcriptions[i][0]
            if trans_norm and (msg_norm in trans_norm or trans_norm in msg_norm):
                matched_at = i
                break
        if matched_at == -1:
            continue  # no transcription → developer-injected; drop
        cursor = matched_at + 1
        kept.append(msg)
    return kept, cursor


def _drop_injected_assistant_messages(
    grouped_messages: list[dict[str, Any]],
    norm_generations: list[tuple[str, int]],
    gen_cursor: int,
) -> tuple[list[dict[str, Any]], list[int], int]:
    """Keep only assistant messages the LLM actually generated, consuming generations in order.
    A genuinely-generated message matches an ``LLMFullResponse`` capture; a developer-injected
    ``{"role":"assistant"}`` context message matches none and is dropped — regardless of position.
    This is the bot-side mirror of :func:`_drop_injected_user_messages` and uses the same
    normalized bidirectional-containment rule, since the assistant aggregator builds the context
    message from the very LLMTextFrames we are matching against (so the committed text equals the
    generation, or is a prefix of it when the turn was interrupted).

    Returns ``(kept_messages, kept_positions, new_cursor)`` — ``kept_positions`` are the indices
    within ``grouped_messages`` that survived, so the caller can map back to physical message
    indices for spoken/ghost detection.
    """
    kept: list[dict[str, Any]] = []
    kept_positions: list[int] = []
    cursor = gen_cursor
    for pos, msg in enumerate(grouped_messages):
        msg_norm = _normalize(msg.get("content", ""))
        matched_at = -1
        for i in range(cursor, len(norm_generations)):
            gen_norm = norm_generations[i][0]
            if gen_norm and (msg_norm in gen_norm or gen_norm in msg_norm):
                matched_at = i
                break
        if matched_at == -1:
            continue  # no LLM generation → developer-injected; drop
        cursor = matched_at + 1
        kept.append(msg)
        kept_positions.append(pos)
    return kept, kept_positions, cursor


def enrich_transcript(
    acc: CallAccumulator,
    messages: list[dict[str, Any]],
    merge_gap_ms: int = DEFAULT_MERGE_GAP_MS,
) -> list[TranscriptSegment]:
    # Real user utterances, in order. Proactive (ghost) user segments preceding the
    # greeting are excluded — they have no user message dict to align to.
    user_segs = [s for s in acc.speech_segments if s.speaker == "user" and not s.is_proactive]
    bot_segs = [s for s in acc.speech_segments if s.speaker == "bot"]
    # A proactive greeting means the bot spoke first, before any real user turn. Any user
    # context message appearing before that greeting therefore cannot be real speech — it is a
    # developer kickoff injected as role=user (e.g. "Greet the customer..."). The assistant/
    # system-role form of the same kickoff is handled by the preamble-collapse below.
    starts_with_proactive_greeting = bool(bot_segs) and bot_segs[0].is_proactive
    # Authoritative real-vs-injected signal: a real user message was transcribed by STT; a
    # developer-injected one was not. Empty when no TranscriptionFrames were captured (e.g. a
    # provider that doesn't emit them) — then we fall back to the proactive-greeting and
    # no-window heuristics below.
    norm_transcriptions = [(_normalize(t), ms) for t, ms in acc.user_transcriptions]
    # Authoritative real-vs-injected signal for assistant messages: a genuinely-generated message
    # matches an LLMFullResponse capture; a developer-injected {"role":"assistant"} message matches
    # none. Empty when no LLM frames were captured — then we fall back to the preamble-collapse.
    norm_generations = [(_normalize(t), ms) for t, ms in acc.assistant_generations]
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
    trans_cursor = 0  # next unconsumed STT transcription (real-vs-injected user matching)
    gen_cursor = 0  # next unconsumed LLM generation (real-vs-injected assistant matching)
    next_turn_index = 0  # stable, unique, sequential index per rendered user/agent row
    seen_user_message = False
    last_agent_interrupted = False  # the user interrupts when the prior agent turn was cut off

    while message_idx < len(messages):
        message = messages[message_idx]
        role = message.get("role", "")

        if role == "system":
            message_idx += 1
            continue

        if role == "user":
            grouped_messages, message_idx = collect_consecutive_user_messages(messages, message_idx)
            # Primary, position-independent filter: drop user messages with no matching STT
            # transcription (developer-injected). Active only when transcriptions were captured.
            if norm_transcriptions:
                grouped_messages, trans_cursor = _drop_injected_user_messages(
                    grouped_messages, norm_transcriptions, trans_cursor
                )
            # Fallback when no transcriptions were captured: a user message before the proactive
            # greeting (bot_cursor still 0) is a role=user kickoff — the user hasn't spoken yet.
            elif starts_with_proactive_greeting and bot_cursor == 0:
                continue
            if not grouped_messages:
                continue
            seen_user_message = True
            # Each user message maps 1:1 to the next speech window (real utterances only).
            group_wins = user_windows[win_cursor : win_cursor + len(grouped_messages)]
            win_cursor += len(group_wins)

            for row_msgs, start_ms, end_ms, stt_ms in _split_user_rows(
                grouped_messages, group_wins, merge_gap_ms
            ):
                # No window → no VAD ever fired → developer-injected; drop it. (Unless nothing
                # was captured at all, in which case render untimed rather than drop the whole
                # degenerate transcript.)
                if start_ms is None and user_windows:
                    continue
                result.append(
                    build_user_segment(
                        row_msgs,
                        start_ms=start_ms or 0,
                        end_ms=end_ms or 0,
                        interrupted=last_agent_interrupted,
                        stt_ms=stt_ms,
                        turn_index=next_turn_index,
                    )
                )
                next_turn_index += 1
                last_agent_interrupted = False
            continue

        if role == "assistant" and "tool_calls" in message:
            # Tool calls belong to the upcoming bot response. FunctionCallInProgressFrame
            # arrives late at the observer (after the bot has spoken), so fall back to the
            # response's user_stopped time when the recorded invocation looks too late.
            bot_started_ms, user_stopped_ms = _upcoming_response_anchor(
                bot_segs, bot_cursor, meas_by_bot_id, seg_by_id
            )
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
            bot_started_ms, user_stopped_ms = _upcoming_response_anchor(
                bot_segs, bot_cursor, meas_by_bot_id, seg_by_id
            )
            result.append(
                build_agent_result_segment(
                    acc=acc,
                    message=message,
                    messages=messages,
                    bot_started_ms=bot_started_ms,
                    user_stopped_ms=user_stopped_ms,
                )
            )
            message_idx += 1
            continue

        if role == "assistant":
            group_start = message_idx
            grouped_messages, message_idx = collect_consecutive_assistant_messages(
                messages, message_idx
            )
            phys_indices = list(range(group_start, message_idx))  # 1:1 with grouped_messages
            # Primary, position-independent filter: drop assistant messages with no matching LLM
            # generation (developer-injected). Active only when LLM frames were captured.
            if norm_generations:
                grouped_messages, kept_positions, gen_cursor = _drop_injected_assistant_messages(
                    grouped_messages, norm_generations, gen_cursor
                )
            # Fallback when no generations were captured: in preamble position (before any user
            # turn) consecutive assistant messages are a pre-seeded developer instruction followed
            # by the real LLM response — only the last was generated (and spoken) by the LLM.
            elif not seen_user_message and len(grouped_messages) > 1:
                grouped_messages = [grouped_messages[-1]]
                kept_positions = [len(phys_indices) - 1]
            else:
                kept_positions = list(range(len(phys_indices)))
            if not grouped_messages:
                continue  # all injected → no row
            # Resolve the last *kept* physical index for spoken/ghost detection (a dropped
            # trailing message must not be mistaken for the spoken one).
            final_msg_idx = phys_indices[kept_positions[-1]]

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
                    grouped_messages,
                    bot_seg,
                    measurement,
                    interrupted=interrupted,
                    turn_index=next_turn_index,
                )
            )
            next_turn_index += 1
            continue

        message_idx += 1

    return result

"""FlowsAccumulator: minimal state for pipecat-flows call data.

Responsibilities:
  1. Call timing (call_start_abs_ns, call_end_abs_ns)
  2. Node transitions (patched _set_node via on_node_entered)
  3. Latency turns (4 frame events per turn)
  4. Call-level usage metrics (tts_chars, asr_duration, approx llm tokens)

Transcript assembly is delegated to flow_manager.get_current_context().
"""

from dataclasses import dataclass, field
from typing import Optional

from .models import (
    AiModels,
    CallPayload,
    GeneralMetaData,
    LatencyTurn,
    NodeInfo,
    NodeTransitionRecord,
    PendingTransition,
    ToolInfo,
    TranscriptSegment,
    UsageToken,
)


@dataclass
class FlowsAccumulator:
    # ── call-level timing ──────────────────────────────────────────────────────
    call_start_abs_ns: int = 0
    call_end_abs_ns: int = 0

    # ── flow state ─────────────────────────────────────────────────────────────
    _current_node: Optional[str] = field(default=None, repr=False)

    # ── node transitions ───────────────────────────────────────────────────────
    node_transitions: list = field(default_factory=list)
    _pending_transition: Optional[PendingTransition] = field(default=None, repr=False)

    # ── latency tracking ──────────────────────────────────────────────────────
    latency_turns: list = field(default_factory=list)
    _turn_index: int = field(default=0, repr=False)
    _user_stopped_ns: int = field(default=0, repr=False)
    _llm_started_ns: int = field(default=0, repr=False)
    _tts_started_ns: int = field(default=0, repr=False)
    _bot_started_ns: int = field(default=0, repr=False)
    _bot_stopped_ns: int = field(default=0, repr=False)
    _user_started_ns: int = field(default=0, repr=False)
    _latency_node: Optional[str] = field(default=None, repr=False)
    _current_turn_confidences: list = field(default_factory=list, repr=False)

    # ── call-level usage counters ──────────────────────────────────────────────
    _tts_chars: int = field(default=0, repr=False)

    # ── misc ───────────────────────────────────────────────────────────────────
    done: bool = False

    # ── helpers ────────────────────────────────────────────────────────────────

    def _rel_ms(self, abs_ns: int) -> int:
        if self.call_start_abs_ns == 0 or abs_ns == 0:
            return 0
        return (abs_ns - self.call_start_abs_ns) // 1_000_000

    def _ns_to_ms(self, a_ns: int, b_ns: int) -> Optional[int]:
        if a_ns == 0 or b_ns == 0:
            return None
        return max(0, (b_ns - a_ns) // 1_000_000)

    # ── flow-specific API ──────────────────────────────────────────────────────

    def get_pending_transition(self) -> Optional[PendingTransition]:
        return self._pending_transition

    def on_node_entered(
        self,
        from_node: Optional[str],
        to_node: str,
        node_config: dict,
        trigger: Optional[PendingTransition],
        state_snapshot: dict,
        timestamp_ns: int,
    ) -> None:
        functions_available = [
            f.name if hasattr(f, "name") else (f.get("name") if isinstance(f, dict) else str(f))
            for f in node_config.get("functions", [])
        ]
        self.node_transitions.append(NodeTransitionRecord(
            from_node=from_node,
            to_node=to_node,
            trigger_function=trigger.function_name if trigger else None,
            trigger_args=trigger.arguments if trigger else None,
            state_snapshot=state_snapshot,
            task_messages=node_config.get("task_messages", []),
            functions_available=functions_available,
            timestamp_ms=self._rel_ms(timestamp_ns),
        ))
        self._current_node = to_node
        self._pending_transition = None

    # ── frame handlers ─────────────────────────────────────────────────────────

    def on_start(self, timestamp_ns: int) -> None:
        self.call_start_abs_ns = timestamp_ns

    def on_transcription(self, frame) -> None:
        # Collect per-word/phrase ASR confidence scores.
        # TranscriptionFrame arrives *after* VADUserStoppedSpeakingFrame, so we
        # accumulate here and snapshot in _flush_latency_turn (called on BotStarted).
        try:
            conf = frame.result.channel.alternatives[0].confidence
            if conf is not None:
                self._current_turn_confidences.append(float(conf))
        except Exception:
            pass

    def on_user_started(self, timestamp_ns: int) -> None:
        if not self._user_started_ns:
            self._user_started_ns = timestamp_ns

    def on_user_stopped(self, frame, timestamp_ns: int) -> None:
        stop_correction_ns = int(getattr(frame, "stop_secs", 0) * 1_000_000_000)
        self._user_stopped_ns = timestamp_ns - stop_correction_ns
        self._llm_started_ns = 0
        self._tts_started_ns = 0
        self._bot_started_ns = 0
        self._latency_node = self._current_node
        # NOTE: do NOT clear _current_turn_confidences here — Deepgram's final
        # TranscriptionFrame arrives *after* VADUserStopped in the async queue.
        # The snapshot happens in _flush_latency_turn after BotStarted.

    def on_llm_started(self, timestamp_ns: int) -> None:
        if self._user_stopped_ns and self._llm_started_ns == 0:
            self._llm_started_ns = timestamp_ns

    def on_tts_started(self, timestamp_ns: int) -> None:
        """Called on TTSStartedFrame (downstream from TTS service).

        More reliable than TTSTextFrame for ttfb_ms because TTSStartedFrame
        always travels downstream and arrives before BotStartedSpeakingFrame
        (which travels upstream from transport.output()).
        """
        if self._user_stopped_ns and self._tts_started_ns == 0:
            self._tts_started_ns = timestamp_ns

    def on_tts_text_chars(self, frame) -> None:
        """Accumulate total characters sent to TTS for call-level billing metadata."""
        text = getattr(frame, "text", "") or ""
        self._tts_chars += len(text)

    def on_bot_started_speaking(self, timestamp_ns: int) -> None:
        if self._user_stopped_ns and self._bot_started_ns == 0:
            self._bot_started_ns = timestamp_ns
            self._flush_latency_turn()

    def on_bot_stopped(self, timestamp_ns: int) -> None:
        if self.done:
            return
        self._bot_stopped_ns = timestamp_ns
        if self.latency_turns:
            self.latency_turns[-1].bot_stopped_ms = self._rel_ms(timestamp_ns)

    def on_function_call_in_progress(self, frame, timestamp_ns: int) -> None:
        name = getattr(frame, "function_name", "") or "function"
        arguments = getattr(frame, "arguments", None)
        ms = self._rel_ms(timestamp_ns)
        self._pending_transition = PendingTransition(
            function_name=name,
            arguments=arguments,
            timestamp_ms=ms,
        )

    def on_call_end(self, timestamp_ns: int) -> None:
        if self.done:
            return
        self.done = True
        self.call_end_abs_ns = timestamp_ns

    # ── internal helpers ───────────────────────────────────────────────────────

    def _flush_latency_turn(self) -> None:
        # Snapshot ASR confidence — TranscriptionFrame arrives after VADUserStopped
        # but well before BotStarted (LLM + TTS gap), so all fragments are here by now.
        confidence = (
            sum(self._current_turn_confidences) / len(self._current_turn_confidences)
            if self._current_turn_confidences else None
        )

        # ttfb_ms split: TTSStartedFrame (downstream) vs BotStartedSpeakingFrame (upstream).
        # TTSStartedFrame should always arrive first, but guard against edge cases.
        if self._tts_started_ns and self._tts_started_ns <= self._bot_started_ns:
            ttfb_ms = self._ns_to_ms(self._user_stopped_ns, self._tts_started_ns)
            tts_ms  = self._ns_to_ms(self._tts_started_ns, self._bot_started_ns)
        else:
            # Fallback: report full user→bot latency as ttfb, tts=0
            ttfb_ms = self._ns_to_ms(self._user_stopped_ns, self._bot_started_ns)
            tts_ms  = 0

        self.latency_turns.append(LatencyTurn(
            turn_index=self._turn_index,
            node=self._latency_node,
            ttfb_ms=ttfb_ms,
            llm_ms=self._ns_to_ms(self._user_stopped_ns, self._llm_started_ns),
            tts_ms=tts_ms,
            bot_started_ms=self._rel_ms(self._bot_started_ns),
            user_stopped_ms=self._rel_ms(self._user_stopped_ns),
            user_started_ms=self._rel_ms(self._user_started_ns),
            user_confidence=confidence,
        ))
        self._turn_index += 1
        self._user_stopped_ns = 0
        self._llm_started_ns = 0
        self._tts_started_ns = 0
        self._bot_started_ns = 0
        self._latency_node = None
        self._user_started_ns = 0
        self._current_turn_confidences = []

    # ── payload builder ────────────────────────────────────────────────────────

    def build_payload(self, config, transcript: list) -> CallPayload:
        enriched  = self._enrich_transcript(transcript)
        start_ts  = self.call_start_abs_ns // 1_000_000_000
        end_ts    = self.call_end_abs_ns   // 1_000_000_000
        total_chars = sum(
            len(m.get("content", "") or "")
            for m in transcript if isinstance(m.get("content"), str)
        )
        return CallPayload(
            call_id=config.call_id,
            call_type=config.call_type,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            recording_url=config.recording_url,
            transcript_with_tool_calls=enriched,
            call_status="call_ended",
            duration_ms=max(0, (self.call_end_abs_ns - self.call_start_abs_ns) // 1_000_000),
            general_meta_data_raw=GeneralMetaData(
                ai_models=AiModels(
                    asr_model=config.asr_model,
                    llm_model=config.llm_model,
                    tts_model=config.tts_model,
                ),
                usage_token=UsageToken(
                    asr_duration=max(0, end_ts - start_ts),
                    llm_token=round(total_chars / 4) if total_chars else None,
                    tts_character_count=self._tts_chars or None,
                ),
            ),
        )

    def _segment_meta(
        self,
        *,
        e2e_latency=None,
        interrupted=False,
        llm_node_ttft=None,
        tts_node_ttfb=None,
        transcript_confidence=None,
        **extra,
    ) -> dict:
        import uuid as _uuid
        return {
            "id":                    str(_uuid.uuid4()),
            "e2e_latency":           e2e_latency,
            "interrupted":           interrupted,
            "llm_node_ttft":         llm_node_ttft,
            "tts_node_ttfb":         tts_node_ttfb,
            "transcript_confidence": transcript_confidence,
            "asr_node_ttft":         None,
            **extra,
        }

    def _enrich_transcript(self, messages: list) -> list:
        import json as _json

        # Index node_transitions by trigger_function for O(1) lookup.
        transitions_by_fn = {
            t.trigger_function: t
            for t in self.node_transitions
            if t.trigger_function
        }

        user_turns      = list(self.latency_turns)
        assistant_turns = list(self.latency_turns)
        user_idx        = 0
        assistant_idx   = 0

        # Pre-compute interrupted flags over all latency_turns.
        user_interrupted = {}
        for j, t in enumerate(self.latency_turns):
            if j == 0:
                user_interrupted[j] = False
            else:
                prev_bot_stopped = self.latency_turns[j - 1].bot_stopped_ms or 0
                user_interrupted[j] = bool(t.user_started_ms < prev_bot_stopped)

        agent_interrupted = {}
        for k, t in enumerate(self.latency_turns):
            if k + 1 < len(self.latency_turns):
                next_user_started = self.latency_turns[k + 1].user_started_ms
                agent_interrupted[k] = bool(next_user_started < (t.bot_stopped_ms or 0))
            else:
                agent_interrupted[k] = False

        result = []
        i = 0
        while i < len(messages):
            msg  = messages[i]
            role = msg.get("role", "")

            if role == "system":
                i += 1
                continue

            elif role == "user":
                # Collect all consecutive user messages (one VAD stop → possibly many fragments).
                group = []
                while i < len(messages) and messages[i].get("role") == "user":
                    group.append(messages[i])
                    i += 1
                turn = user_turns[user_idx] if user_idx < len(user_turns) else None
                combined_text = " ".join(m.get("content", "") for m in group).strip()
                result.append(TranscriptSegment(
                    role="user",
                    text=combined_text,
                    start_ms=turn.user_started_ms if turn else 0,
                    end_ms=turn.user_stopped_ms if turn else 0,
                    metadata=self._segment_meta(
                        interrupted=user_interrupted.get(user_idx, False),
                        transcript_confidence=turn.user_confidence if turn else None,
                        node=turn.node if turn else None,
                        turn_index=turn.turn_index if turn else None,
                        fragments=len(group) if len(group) > 1 else None,
                    ),
                ))
                user_idx += 1
                continue

            elif role == "assistant" and "tool_calls" in msg:
                tc       = msg["tool_calls"][0]
                fn_name  = tc["function"]["name"]
                raw_args = tc["function"].get("arguments", "{}")
                args     = _try_parse(raw_args) or {}
                t        = transitions_by_fn.get(fn_name)
                arg_str  = ", ".join(
                    f"{k}={v}" for k, v in (args.items() if isinstance(args, dict) else [])
                )
                result.append(TranscriptSegment(
                    role="agent_function",
                    text=f"{fn_name}({arg_str})",
                    start_ms=t.timestamp_ms if t else 0,
                    end_ms=t.timestamp_ms if t else 0,
                    tool=ToolInfo(
                        name=fn_name,
                        request_id=tc.get("id"),
                        params=args if isinstance(args, dict) else {},
                    ),
                    metadata=self._segment_meta(node=t.from_node if t else None),
                ))

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                # Resolve which assistant tool_call this result belongs to.
                matched_tc = next(
                    (
                        tc
                        for m2 in messages
                        if m2.get("role") == "assistant" and "tool_calls" in m2
                        for tc in m2["tool_calls"]
                        if tc.get("id") == tool_call_id
                    ),
                    None,
                )
                fn_name = matched_tc["function"]["name"] if matched_tc else None
                t       = transitions_by_fn.get(fn_name) if fn_name else None
                parsed  = _try_parse(msg.get("content"))
                result.append(TranscriptSegment(
                    role="agent_result",
                    text=_json.dumps(parsed, default=str) if parsed is not None else msg.get("content", ""),
                    start_ms=t.timestamp_ms if t else 0,
                    end_ms=t.timestamp_ms if t else 0,
                    tool=ToolInfo(
                        name=fn_name,
                        request_id=tool_call_id or None,
                        result=parsed if isinstance(parsed, dict) else {"value": parsed},
                    ),
                    metadata=self._segment_meta(
                        node=t.from_node if t else None,
                        triggered_transition_to=t.to_node if t else None,
                    ),
                ))
                # Inject node_transition entry immediately after agent_result.
                if t:
                    result.append(TranscriptSegment(
                        role="node_transition",
                        text=f"{t.from_node} → {t.to_node}",
                        start_ms=t.timestamp_ms,
                        end_ms=t.timestamp_ms,
                        node=NodeInfo(
                            from_node=t.from_node,
                            to=t.to_node,
                            reason=fn_name,
                        ),
                        metadata=self._segment_meta(
                            trigger_args=t.trigger_args,
                            state_snapshot=t.state_snapshot,
                            functions_available=t.functions_available,
                        ),
                    ))

            elif role == "assistant":
                turn = assistant_turns[assistant_idx] if assistant_idx < len(assistant_turns) else None
                e2e  = ((turn.bot_started_ms - turn.user_stopped_ms) or None) if turn else None
                result.append(TranscriptSegment(
                    role="agent",
                    text=msg.get("content", ""),
                    start_ms=turn.bot_started_ms if turn else 0,
                    end_ms=(turn.bot_stopped_ms if turn and turn.bot_stopped_ms is not None
                            else self._rel_ms(self.call_end_abs_ns)),
                    metadata=self._segment_meta(
                        e2e_latency=e2e if e2e and e2e > 0 else None,
                        interrupted=agent_interrupted.get(assistant_idx, False),
                        llm_node_ttft=turn.llm_ms if turn else None,
                        tts_node_ttfb=turn.tts_ms if turn else None,
                        node=turn.node if turn else None,
                        turn_index=turn.turn_index if turn else None,
                    ),
                ))
                assistant_idx += 1

            i += 1

        # Inject the initial node_transition (null → first_node, no trigger function).
        initial = next(
            (t for t in self.node_transitions if t.trigger_function is None), None
        )
        if initial:
            result.insert(0, TranscriptSegment(
                role="node_transition",
                text=f"→ {initial.to_node}",
                start_ms=initial.timestamp_ms,
                end_ms=initial.timestamp_ms,
                node=NodeInfo(
                    from_node="",
                    to=initial.to_node,
                    reason="",
                ),
                metadata=self._segment_meta(
                    state_snapshot=initial.state_snapshot,
                    functions_available=initial.functions_available,
                ),
            ))

        return result


def _try_parse(s):
    import json as _json
    try:
        return _json.loads(s) if isinstance(s, str) else s
    except Exception:
        return s

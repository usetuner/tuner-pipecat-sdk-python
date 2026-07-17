"""Data models for tuner_pipecat_sdk."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SpeechSegment(BaseModel):
    """A single raw, timestamped unit of speech — either the user or the bot.

    Append-only and makes no cardinality assumption: a user may produce several
    consecutive segments before any bot reply (mid-sentence pauses, silence gaps),
    a bot may produce several segments for one response (speak → tool → speak),
    and a user segment with no following bot reply is a natural orphan state.

    ``stt_ms`` lives here (user segments only) because STT finalization latency is
    a property of the user's own utterance — it exists whether or not the bot ever
    replies, so it must survive on orphan turns.
    """

    id: int
    speaker: str  # "user" | "bot"
    turn_number: int | None = None  # TurnTrackingObserver turn this segment belongs to
    start_ms: int = 0
    stop_ms: int | None = None
    stt_ms: int | None = None
    spoken_text: str | None = None  # bot segments only — text actually voiced via TTS
    # User segments only — one [start, stop] per VAD utterance. A coalesced turn (several
    # utterances before any bot reply) holds several windows, giving per-utterance timing
    # the single start/stop can't. Used by the enricher to align and gap-split user rows.
    windows: list[list[int | None]] = Field(default_factory=list)
    node: str | None = None
    interrupted: bool | None = None  # bot segment cut off by the user
    interrupted_at_ms: int | None = None  # bot segments only — when the user cut in
    is_proactive: bool = False  # bot segment preceding any user speech


class LatencyMeasurement(BaseModel):
    """Latency of one bot response, derived only when a bot reply follows a user
    segment (or the proactive greeting). Linked to its speech segments by id.

    Carries only response-derived metrics: nothing here exists independent of a
    reply. An orphan user segment (no bot response) has no LatencyMeasurement.
    """

    user_segment_id: int  # FK → SpeechSegment.id (-1 for the proactive greeting)
    bot_segment_id: int | None = None  # FK → first bot SpeechSegment of the response
    ttfb_ms: int | None = None
    llm_ms: int | None = None
    tts_ms: int | None = None
    e2e_ms: int | None = None  # bot.start_ms - user.stop_ms
    is_proactive: bool = False
    was_interrupted: bool | None = None
    interrupted_at_ms: int | None = None


@dataclass
class LiveTurn:
    """A transcript row built LIVE from the frame stream, in arrival order — the event-sourced
    mirror of pipecat's own conversation construction (usePipecatConversation). The transcript is
    *not* reconstructed from the final LLM context; each turn is materialized when its frames
    arrive. Timing is read from the linked SpeechSegment/LatencyMeasurement at assembly (where it
    is final, after latency adjustments); only an unvoiced draft falls back to ``generated_ms``.

    kind: "user" | "agent" | "agent_function" | "agent_result".
    """

    kind: str
    order: int
    # user: each finalized STT transcription in this turn, as (text, rel_ms). A user row groups
    # exactly the transcriptions between two UserStoppedSpeaking boundaries — the same unit
    # pipecat's aggregator commits as one user message — so there is no silence-gap split/merge.
    chunks: list[tuple[str, int]] = field(default_factory=list)
    # agent: joined LLM-generated text for this response.
    text: str = ""
    # user row timing, captured live from the frames (not derived from segments).
    start_ms: int = 0
    end_ms: int | None = None
    stt_ms: int | None = None
    eou_ms: int | None = None
    eou_confidence: float | None = None
    eou_reason: str | None = None
    # Links into the latency/speech substrate (never matched by text).
    user_segment_id: int | None = None
    bot_segment_id: int | None = None
    # agent fallback timing when the turn was generated but never voiced (no bot segment).
    generated_ms: int | None = None
    # tool rows:
    tool_call_id: str | None = None
    function_name: str | None = None
    arguments: Any = None
    result: Any = None
    occurrence_ms: int = 0


class TranscriptWord(BaseModel):
    """Word-level timing within a transcript segment."""

    word: str
    start_ms: int
    end_ms: int
    confidence: float | None = None


class TranscriptMetadata(BaseModel):
    """Known metadata fields on a transcript segment.

    The backend accepts extra keys (extra="allow"), so provider-specific fields
    can be included alongside these known ones.
    """

    model_config = ConfigDict(extra="allow")

    e2e_latency: float | None = None
    interrupted: bool | None = None
    llm_node_ttft: float | None = None
    tts_node_ttfb: float | None = None
    transcript_confidence: float | None = None
    stt_node_ttfb: float | None = None
    asr_node_ttfb: float | None = None
    eou_delay: float | None = None
    eou_confidence: float | None = None
    eou_reason: str | None = None


class ToolInfo(BaseModel):
    """Tool call or result details for agent_function / agent_result segments."""

    name: str | None = None
    request_id: str | None = None
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    is_error: bool | None = None
    error: str | None = None
    start_ms: int | None = None


class NodeInfo(BaseModel):
    """LangGraph node transition details for node_transition segments.

    Mirrors the ``{"to", "from", "reason"}`` shape tuner-langchain's
    ``segment_builder.build_node_segment`` already produces.
    """

    model_config = ConfigDict(populate_by_name=True)

    to: str | None = None
    from_: str | None = Field(default=None, alias="from")
    reason: str | None = None


class TranscriptSegment(BaseModel):
    role: str
    text: str | None = None
    start_ms: int
    end_ms: int | None = None
    metadata: dict[str, Any]
    words: list[TranscriptWord] | None = None
    duration_ms: int | None = None
    tool: ToolInfo | None = None
    node: NodeInfo | None = None


# A transform applied to the final transcript segment list at payload-build
# time -- e.g. merging LangChain-sourced node/tool segments into the native
# pipecat transcript. Lets optional integrations contribute to the payload
# (via CallAccumulator.register_segment_enricher()) without payload_builder.py
# or CallAccumulator needing to import or know about any specific integration.
SegmentEnricher = Callable[[list[TranscriptSegment]], list[TranscriptSegment]]


class AiModels(BaseModel):
    asr_model: str
    llm_model: str
    tts_model: str


class UsageToken(BaseModel):
    asr_duration: int  # seconds (call wall-clock duration: end_ts - start_ts)
    llm_token: int | None = None
    tts_character_count: int | None = None


class CallUsage(BaseModel):
    """Usage data passed to a cost_calculator callable at the end of each call."""

    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None
    llm_total_tokens: int | None = None
    tts_characters: int | None = None
    stt_audio_seconds: int = 0


class GeneralMetaData(BaseModel):
    ai_models: AiModels
    usage_token: UsageToken


class DisconnectReason(str, Enum):
    """Well-known call ended-reason strings.

    Extends str so values compare equal to their string representations —
    DisconnectReason.USER_HANGUP == "user_hangup" is True — meaning no
    casting is needed when passing to disconnection_reason_resolver.

    Usage::

        Observer(
            ...
            disconnection_reason_resolver=lambda: DisconnectReason.USER_HANGUP
        )
    """

    USER_HANGUP = "user_hangup"
    AGENT_HANGUP = "agent_hangup"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class CallPayload(BaseModel):
    call_id: str
    call_type: str
    start_timestamp: int
    end_timestamp: int
    recording_url: str
    transcript_with_tool_calls: list[TranscriptSegment]
    call_status: str
    duration_ms: int
    general_meta_data_raw: GeneralMetaData
    disconnection_reason: str | None = None
    sip_call_id: str | None = None
    sip_headers: dict[str, str] | None = None
    agent_version: int | None = None
    recipient: str | None = None
    cost: float | None = Field(None, serialization_alias="call_cost")
    """Phone number or SIP URL of the callee party (e.g. ``+15551234567`` or ``sip:alice@example.com``).

    Not populated automatically — pass this field explicitly when the callee identity is known.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize payload to a JSON-ready dict."""
        return self.model_dump(exclude_none=True, by_alias=True)


__all__ = [
    "SpeechSegment",
    "LatencyMeasurement",
    "TranscriptWord",
    "TranscriptMetadata",
    "ToolInfo",
    "NodeInfo",
    "TranscriptSegment",
    "SegmentEnricher",
    "AiModels",
    "UsageToken",
    "CallUsage",
    "GeneralMetaData",
    "CallPayload",
    "DisconnectReason",
]

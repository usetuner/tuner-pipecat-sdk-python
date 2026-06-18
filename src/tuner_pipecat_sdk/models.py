"""Data models for tuner_pipecat_sdk."""

from __future__ import annotations

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
    start_ms: int = 0
    stop_ms: int | None = None
    stt_ms: int | None = None
    spoken_text: str | None = None  # bot segments only — text actually voiced via TTS
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


class ToolInfo(BaseModel):
    """Tool call or result details for agent_function / agent_result segments."""

    name: str | None = None
    request_id: str | None = None
    params: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    is_error: bool | None = None
    error: str | None = None
    start_ms: int | None = None


class TranscriptSegment(BaseModel):
    role: str
    text: str | None = None
    start_ms: int
    end_ms: int | None = None
    metadata: dict[str, Any]
    words: list[TranscriptWord] | None = None
    duration_ms: int | None = None
    tool: ToolInfo | None = None


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
    "TranscriptSegment",
    "AiModels",
    "UsageToken",
    "CallUsage",
    "GeneralMetaData",
    "CallPayload",
    "DisconnectReason",
]

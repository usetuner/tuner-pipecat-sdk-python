"""Data models for pipecat_flows_tuner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PendingTransition(BaseModel):
    function_name: str
    arguments: Any
    timestamp_ms: int


class NodeTransitionRecord(BaseModel):
    from_node: str | None = None
    to_node: str
    trigger_function: str | None = None
    trigger_args: Any | None = None
    timestamp_ms: int
    trigger_timestamp_ms: int | None = None  # when the function call was initiated


class LatencyTurn(BaseModel):
    turn_index: int
    node: str | None = None
    bot_node: str | None = None
    ttfb_ms: int | None = None
    llm_ms: int | None = None
    tts_ms: int | None = None
    bot_started_ms: int = 0
    user_stopped_ms: int = 0
    user_started_ms: int = 0
    bot_stopped_ms: int | None = None
    was_interrupted: bool | None = None


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


class NodeInfo(BaseModel):
    from_node: str | None = Field(None, serialization_alias="from")
    to: str
    reason: str | None = None


class TranscriptSegment(BaseModel):
    role: str
    text: str
    start_ms: int
    end_ms: int | None = None
    metadata: dict[str, Any]
    words: list[TranscriptWord] | None = None
    duration_ms: int | None = None
    tool: ToolInfo | None = None
    node: NodeInfo | None = None


class AiModels(BaseModel):
    asr_model: str
    llm_model: str
    tts_model: str


class UsageToken(BaseModel):
    asr_duration: int
    llm_token: int | None = None
    tts_character_count: int | None = None


class GeneralMetaData(BaseModel):
    ai_models: AiModels
    usage_token: UsageToken


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize payload to a JSON-ready dict."""
        return self.model_dump(exclude_none=True, by_alias=True)


__all__ = [
    "PendingTransition",
    "NodeTransitionRecord",
    "LatencyTurn",
    "TranscriptWord",
    "TranscriptMetadata",
    "ToolInfo",
    "NodeInfo",
    "TranscriptSegment",
    "AiModels",
    "UsageToken",
    "GeneralMetaData",
    "CallPayload",
]

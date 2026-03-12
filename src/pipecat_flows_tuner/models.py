"""Data models for pipecat_flows_tuner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PendingTransition(BaseModel):
    function_name: str
    arguments: Any
    timestamp_ms: int


class NodeTransitionRecord(BaseModel):
    from_node: str | None = None
    to_node: str
    trigger_function: str | None = None
    trigger_args: Any | None = None
    state_snapshot: dict[str, Any]
    task_messages: list[dict[str, Any]]
    functions_available: list[str]
    timestamp_ms: int


class LatencyTurn(BaseModel):
    turn_index: int
    node: str | None = None
    ttfb_ms: int | None = None
    llm_ms: int | None = None
    tts_ms: int | None = None
    bot_started_ms: int
    user_stopped_ms: int
    user_started_ms: int
    bot_stopped_ms: int | None = None


class ToolInfo(BaseModel):
    name: str | None = None
    request_id: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None


class NodeInfo(BaseModel):
    from_node: str | None = Field(None, serialization_alias="from")
    to: str
    reason: str | None = None


class TranscriptSegment(BaseModel):
    role: str
    text: str
    start_ms: int
    end_ms: int
    metadata: dict[str, Any]
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
    "ToolInfo",
    "NodeInfo",
    "TranscriptSegment",
    "AiModels",
    "UsageToken",
    "GeneralMetaData",
    "CallPayload",
]

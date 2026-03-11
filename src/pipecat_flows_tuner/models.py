"""Data models for pipecat_flows_tuner."""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Internal / accumulator models ─────────────────────────────────────────────

class PendingTransition(BaseModel):
    function_name: str
    arguments: Any
    timestamp_ms: int


class NodeTransitionRecord(BaseModel):
    from_node: Optional[str] = None
    to_node: str
    trigger_function: Optional[str] = None
    trigger_args: Optional[Any] = None
    state_snapshot: dict
    task_messages: list
    functions_available: list[str]
    timestamp_ms: int


class LatencyTurn(BaseModel):
    turn_index: int
    node: Optional[str] = None
    ttfb_ms: Optional[int] = None
    llm_ms: Optional[int] = None
    tts_ms: Optional[int] = None
    bot_started_ms: int
    user_stopped_ms: int
    user_started_ms: int
    user_confidence: Optional[float] = None
    bot_stopped_ms: Optional[int] = None  # filled in later by on_bot_stopped


# ── Transcript segment models ──────────────────────────────────────────────────

class ToolInfo(BaseModel):
    name: Optional[str] = None
    request_id: Optional[str] = None
    params: Optional[dict] = None    # agent_function segments
    result: Optional[Any] = None     # agent_result segments


class NodeInfo(BaseModel):
    # "from" is a reserved keyword; serialized as "from" via alias
    from_node: Optional[str] = Field(None, serialization_alias="from")
    to: str
    reason: Optional[str] = None


class TranscriptSegment(BaseModel):
    role: str          # "user" | "agent" | "agent_function" | "agent_result" | "node_transition"
    text: str
    start_ms: int
    end_ms: int
    metadata: dict
    tool: Optional[ToolInfo] = None     # agent_function, agent_result
    node: Optional[NodeInfo] = None     # node_transition


# ── Payload models ─────────────────────────────────────────────────────────────

class AiModels(BaseModel):
    asr_model: str
    llm_model: str
    tts_model: str


class UsageToken(BaseModel):
    asr_duration: int
    llm_token: Optional[int] = None
    tts_character_count: Optional[int] = None


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

    def to_dict(self) -> dict:
        """Serialize to JSON-ready dict for the Tuner API."""
        return self.model_dump(exclude_none=True, by_alias=True)

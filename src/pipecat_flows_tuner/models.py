"""Data models for pipecat_flows_tuner."""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ── Internal / accumulator models ─────────────────────────────────────────────

@dataclass
class PendingTransition:
    function_name: str
    arguments: Any
    timestamp_ms: int


@dataclass
class NodeTransitionRecord:
    from_node: Optional[str]
    to_node: str
    trigger_function: Optional[str]
    trigger_args: Optional[Any]
    state_snapshot: dict
    task_messages: list
    functions_available: list
    timestamp_ms: int


@dataclass
class LatencyTurn:
    turn_index: int
    node: Optional[str]
    ttfb_ms: Optional[int]
    llm_ms: Optional[int]
    tts_ms: Optional[int]
    bot_started_ms: int
    user_stopped_ms: int
    user_started_ms: int
    user_confidence: Optional[float]
    bot_stopped_ms: Optional[int] = None  # filled in later by on_bot_stopped


# ── Transcript segment models ──────────────────────────────────────────────────

@dataclass
class ToolInfo:
    name: Optional[str]
    request_id: Optional[str]
    params: Optional[dict] = None    # agent_function segments
    result: Optional[Any] = None     # agent_result segments


@dataclass
class NodeInfo:
    # "from" is a keyword; serialized as "from" in to_dict()
    from_node: Optional[str]
    to: str
    reason: Optional[str]


@dataclass
class TranscriptSegment:
    role: str          # "user" | "agent" | "agent_function" | "agent_result" | "node_transition"
    text: str
    start_ms: int
    end_ms: int
    metadata: dict
    tool: Optional[ToolInfo] = None     # agent_function, agent_result
    node: Optional[NodeInfo] = None     # node_transition


# ── Payload models ─────────────────────────────────────────────────────────────

@dataclass
class AiModels:
    asr_model: str
    llm_model: str
    tts_model: str


@dataclass
class UsageToken:
    asr_duration: int
    llm_token: Optional[int]
    tts_character_count: Optional[int]


@dataclass
class GeneralMetaData:
    ai_models: AiModels
    usage_token: UsageToken


@dataclass
class CallPayload:
    call_id: str
    call_type: str
    start_timestamp: int
    end_timestamp: int
    recording_url: str
    transcript_with_tool_calls: list
    call_status: str
    duration_ms: int
    general_meta_data_raw: GeneralMetaData

    def to_dict(self) -> dict:
        """Serialize to JSON-ready dict for the Tuner API, omitting None values."""
        d = asdict(self)
        # Rename from_node → from in NodeInfo before stripping None
        for seg in d.get("transcript_with_tool_calls", []):
            if seg.get("node") and "from_node" in seg["node"]:
                seg["node"]["from"] = seg["node"].pop("from_node")
        return _strip_none(d)


def _strip_none(obj):
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj

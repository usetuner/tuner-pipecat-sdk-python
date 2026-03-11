"""Configuration model for pipecat_flows_tuner."""

from pydantic import BaseModel, field_validator


class TunerConfig(BaseModel):
    api_key: str
    workspace_id: int
    agent_id: str
    call_id: str
    call_type: str = "web_call"
    base_url: str = "http://localhost:8000"
    recording_url: str = "pipecat://no-recording"
    debug: bool = False
    asr_model: str = ""
    llm_model: str = ""
    tts_model: str = ""

    @field_validator("api_key", "call_id", "agent_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("workspace_id")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

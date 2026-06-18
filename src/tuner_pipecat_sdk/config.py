"""Configuration model for tuner_pipecat_sdk."""

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
    sip_call_id: str | None = None
    sip_headers: dict[str, str] | None = None
    agent_version: int | None = None
    recipient: str | None = None
    """Phone number or SIP URL of the callee party (e.g. ``+15551234567`` or ``sip:alice@example.com``).

    Not collected automatically — the caller is responsible for supplying this value when known.
    """
    user_segment_merge_gap_ms: int = 1500
    """Silence threshold (ms) for merging consecutive user speech segments into one
    transcript row. Mid-sentence VAD pauses (~300-600ms) merge; gaps at or above this
    (e.g. the user waiting on an unresponsive agent) stay as separate rows so the
    silence is visible. See transcript_enricher merge-by-gap logic."""

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

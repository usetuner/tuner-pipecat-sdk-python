"""Configuration model for tuner_pipecat_sdk."""

from pydantic import BaseModel, field_validator


class TunerConfig(BaseModel):
    api_key: str
    workspace_id: int
    agent_id: str
    call_id: str
    call_type: str = "web_call"
    base_url: str = "https://api.usetuner.ai"
    recording_url: str = "pipecat://no-recording"
    # Forward the call's OTel spans to Tuner so its Trace tab is populated. No-ops
    # unless the optional OTel packages are installed:
    #   pip install 'tuner-pipecat-sdk[traces]'
    forward_traces: bool = True
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
    extra_metadata: dict | None = None
    """Arbitrary call-level key/value metadata, merged into ``general_meta_data_raw``."""

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

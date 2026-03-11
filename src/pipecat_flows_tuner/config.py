"""Configuration dataclass for pipecat_flows_tuner."""

from dataclasses import dataclass


@dataclass
class TunerConfig:
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

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("TunerConfig: api_key must not be empty")
        if not self.call_id:
            raise ValueError("TunerConfig: call_id must not be empty")
        if self.workspace_id <= 0:
            raise ValueError("TunerConfig: workspace_id must be a positive integer")
        if not self.agent_id:
            raise ValueError("TunerConfig: agent_id must not be empty")

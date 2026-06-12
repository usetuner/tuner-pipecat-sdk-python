"""Shared pytest fixtures for tuner_pipecat_sdk tests."""

import pytest

from tuner_pipecat_sdk.config import TunerConfig


@pytest.fixture
def tuner_config():
    return TunerConfig(
        api_key="test-api-key",
        workspace_id=42,
        agent_id="test-agent",
        call_id="call-123",
        call_type="web_call",
        base_url="https://tuner.example.com",
        recording_url="https://example.com/recording.mp3",
        debug=False,
        asr_model="deepgram",
        llm_model="gpt-4",
        tts_model="eleven",
    )

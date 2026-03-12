"""Shared pytest fixtures for pipecat_flows_tuner tests."""

from unittest.mock import MagicMock

import pytest

from pipecat_flows_tuner.config import TunerConfig


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


@pytest.fixture
def mock_flow_manager():
    """Minimal flow manager mock with state and get_current_context."""
    fm = MagicMock()
    fm._current_node = None
    fm.state = {}
    fm.get_current_context.return_value = []
    return fm

"""Tests for TunerConfig validation."""

import pytest
from pydantic import ValidationError

from tuner_pipecat_sdk.config import TunerConfig


def test_config_valid(tuner_config):
    assert tuner_config.api_key == "test-api-key"
    assert tuner_config.workspace_id == 42
    assert tuner_config.agent_id == "test-agent"
    assert tuner_config.call_id == "call-123"
    assert tuner_config.call_type == "web_call"
    assert tuner_config.base_url == "https://tuner.example.com"
    assert tuner_config.recording_url == "https://example.com/recording.mp3"
    assert tuner_config.debug is False
    assert tuner_config.asr_model == "deepgram"
    assert tuner_config.llm_model == "gpt-4"
    assert tuner_config.tts_model == "eleven"


def test_config_defaults():
    c = TunerConfig(
        api_key="key",
        workspace_id=1,
        agent_id="agent",
        call_id="call",
    )
    assert c.call_type == "web_call"
    assert c.base_url == "http://localhost:8000"
    assert c.recording_url == "pipecat://no-recording"
    assert c.debug is False
    assert c.asr_model == ""
    assert c.llm_model == ""
    assert c.tts_model == ""


def test_config_api_key_empty_raises():
    with pytest.raises(ValidationError) as exc_info:
        TunerConfig(
            api_key="   ",
            workspace_id=1,
            agent_id="a",
            call_id="c",
        )
    assert "must not be empty" in str(exc_info.value).lower() or "api_key" in str(exc_info.value)


def test_config_call_id_empty_raises():
    with pytest.raises(ValidationError) as exc_info:
        TunerConfig(
            api_key="key",
            workspace_id=1,
            agent_id="a",
            call_id="",
        )
    assert "must not be empty" in str(exc_info.value).lower() or "call_id" in str(exc_info.value)


def test_config_agent_id_empty_raises():
    with pytest.raises(ValidationError) as exc_info:
        TunerConfig(
            api_key="key",
            workspace_id=1,
            agent_id="  ",
            call_id="c",
        )
    assert "must not be empty" in str(exc_info.value).lower() or "agent_id" in str(exc_info.value)


def test_config_workspace_id_zero_raises():
    with pytest.raises(ValidationError) as exc_info:
        TunerConfig(
            api_key="key",
            workspace_id=0,
            agent_id="a",
            call_id="c",
        )
    assert "positive" in str(exc_info.value).lower() or "workspace_id" in str(exc_info.value)


def test_config_workspace_id_negative_raises():
    with pytest.raises(ValidationError) as exc_info:
        TunerConfig(
            api_key="key",
            workspace_id=-1,
            agent_id="a",
            call_id="c",
        )
    assert "positive" in str(exc_info.value).lower() or "workspace_id" in str(exc_info.value)

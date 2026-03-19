"""Tests for post_call HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tuner_pipecat_sdk.client import post_call
from tuner_pipecat_sdk.models import (
    AiModels,
    CallPayload,
    GeneralMetaData,
    TranscriptSegment,
    UsageToken,
)


@pytest.fixture
def sample_payload():
    return CallPayload(
        call_id="call-123",
        call_type="web_call",
        start_timestamp=1000,
        end_timestamp=2000,
        recording_url="https://example.com/rec",
        transcript_with_tool_calls=[
            TranscriptSegment(role="user", text="Hi", start_ms=0, end_ms=100, metadata={}),
        ],
        call_status="call_ended",
        duration_ms=1000,
        general_meta_data_raw=GeneralMetaData(
            ai_models=AiModels(asr_model="dg", llm_model="gpt", tts_model="eleven"),
            usage_token=UsageToken(asr_duration=1000, llm_token=10, tts_character_count=50),
        ),
    )


def _make_client_mock(response):
    """Build a mock AsyncClient whose context manager returns an object with .post."""
    inner = MagicMock()
    inner.post = AsyncMock(return_value=response)
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=inner)
    client.__aexit__ = AsyncMock(return_value=None)
    return client, inner


@pytest.mark.asyncio
async def test_post_call_success(tuner_config, sample_payload):
    with patch("tuner_pipecat_sdk.client.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "tuner-1", "call_id": "call-123"}
        mock_client, inner = _make_client_mock(mock_response)
        mock_client_cls.return_value = mock_client

        await post_call(tuner_config, sample_payload)

        inner.post.assert_called_once()
        call_args, call_kw = inner.post.call_args
        url = call_args[0] if call_args else call_kw.get("url", "")
        assert "api/v1/public/call" in url
        assert "workspace_id=42" in url
        assert "agent_remote_identifier=test-agent" in url
        assert "json" in call_kw
        assert call_kw["json"]["call_id"] == "call-123"
        assert "Authorization" in call_kw["headers"]
        assert call_kw["headers"]["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_post_call_409_skipped(tuner_config, sample_payload):
    with patch("tuner_pipecat_sdk.client.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_client, inner = _make_client_mock(mock_response)
        mock_client_cls.return_value = mock_client

        await post_call(tuner_config, sample_payload)

        inner.post.assert_called_once()
        # no raise


@pytest.mark.asyncio
async def test_post_call_http_error_logged_not_raised(tuner_config, sample_payload):
    with patch("tuner_pipecat_sdk.client.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )
        mock_client, _ = _make_client_mock(mock_response)
        mock_client_cls.return_value = mock_client

        await post_call(tuner_config, sample_payload)
        # should not raise


@pytest.mark.asyncio
async def test_post_call_request_error_logged_not_raised(tuner_config, sample_payload):
    with patch("tuner_pipecat_sdk.client.httpx.AsyncClient") as mock_client_cls:
        mock_response = MagicMock()
        mock_client, inner = _make_client_mock(mock_response)
        inner.post.side_effect = httpx.RequestError("network error")
        mock_client_cls.return_value = mock_client

        await post_call(tuner_config, sample_payload)
        # should not raise

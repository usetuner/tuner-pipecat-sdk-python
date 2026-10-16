"""Tests for tuner_pipecat_sdk.recording.upload_recording — path, bytes and stream sources."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import respx
from httpx import Response

from tuner_pipecat_sdk import TunerConfig
from tuner_pipecat_sdk.recording import upload_recording

UPLOAD_URL = "https://s3.example.com/bucket"


@pytest.fixture
def config() -> TunerConfig:
    return TunerConfig(api_key="test-key", agent_id="agent-456", workspace_id=1, call_id="call-123")


def _response(**target_extra: Any) -> dict[str, Any]:
    return {
        "recording_upload": {
            "url": UPLOAD_URL,
            "fields": {"key": "calls/call-123.wav", "policy": "abc"},
            "content_type": "audio/wav",
            "key": "calls/call-123.wav",
            **target_extra,
        }
    }


def _sent_filename(route: Any) -> str:
    body = route.calls.last.request.content.decode("utf-8", "replace")
    marker = 'name="file"; filename="'
    start = body.index(marker) + len(marker)
    return body[start : body.index('"', start)]


class TestUploadRecording:
    @respx.mock
    async def test_uploads_from_path(self, config: TunerConfig, tmp_path: Path) -> None:
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))
        audio = tmp_path / "call.wav"
        audio.write_bytes(b"RIFFfake")

        assert await upload_recording(_response(), audio, config) is True
        assert _sent_filename(route) == "call.wav"

    @respx.mock
    async def test_uploads_from_bytes(self, config: TunerConfig) -> None:
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))

        assert await upload_recording(_response(), b"RIFFfake", config) is True
        # No path to borrow a name from: falls back to the target key.
        assert _sent_filename(route) == "call-123.wav"

    @respx.mock
    async def test_uploads_from_stream_with_explicit_filename(self, config: TunerConfig) -> None:
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))
        stream = io.BytesIO(b"RIFFfake")

        result = await upload_recording(_response(), stream, config, filename="stream.wav")

        assert result is True
        assert _sent_filename(route) == "stream.wav"

    @respx.mock
    async def test_bytes_over_max_bytes_are_skipped(self, config: TunerConfig) -> None:
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))

        result = await upload_recording(_response(max_bytes=4), b"RIFFfake", config)

        assert result is False
        assert not route.called

    @respx.mock
    async def test_seekable_stream_over_max_bytes_is_skipped(self, config: TunerConfig) -> None:
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))

        result = await upload_recording(_response(max_bytes=4), io.BytesIO(b"RIFFfake"), config)

        assert result is False
        assert not route.called

    @respx.mock
    async def test_unmeasurable_stream_uploads_whole_body(self, config: TunerConfig) -> None:
        """A non-seekable stream skips the local guard; S3's policy enforces it."""
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))

        class Unseekable(io.RawIOBase):
            def __init__(self, data: bytes) -> None:
                self._buf = io.BytesIO(data)

            def readable(self) -> bool:
                return True

            def seekable(self) -> bool:
                return False

            def readinto(self, b: Any) -> int:
                return self._buf.readinto(b)

        result = await upload_recording(
            _response(max_bytes=4), Unseekable(b"RIFFfake"), config, filename="s.wav"
        )

        assert result is True
        assert b"RIFFfake" in route.calls.last.request.content

    @respx.mock
    async def test_seekable_stream_is_sent_whole_and_measured_whole(
        self, config: TunerConfig
    ) -> None:
        """httpx rewinds a seekable stream, so the guard must size it from 0."""
        route = respx.post(UPLOAD_URL).mock(return_value=Response(204))
        stream = io.BytesIO(b"SKIPRIFFfake")
        stream.seek(4)

        # 8 bytes remain from the current position, but 12 will be uploaded.
        assert await upload_recording(_response(max_bytes=10), stream, config) is False
        assert not route.called

        stream.seek(4)
        assert await upload_recording(_response(), stream, config) is True
        assert b"SKIPRIFFfake" in route.calls.last.request.content

    async def test_missing_path_is_reported(self, config: TunerConfig, tmp_path: Path) -> None:
        result = await upload_recording(_response(), tmp_path / "nope.wav", config)
        assert result is False

    async def test_response_without_target_is_skipped(self, config: TunerConfig) -> None:
        assert await upload_recording({}, b"RIFFfake", config) is False
        assert await upload_recording(None, b"RIFFfake", config) is False

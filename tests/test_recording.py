"""Tests for recording opt-in, the audio_buffer() factory, and to_wav()."""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tuner_pipecat_sdk import to_wav
from tuner_pipecat_sdk.observer import Observer

TARGET = {
    "url": "https://s3.example.com/bucket",
    "fields": {"key": "calls/abc.wav", "policy": "eyJ...", "x-amz-signature": "sig"},
    "key": "calls/abc.wav",
    "content_type": "audio/wav",
    "max_bytes": 50_000_000,
}


def _observer(**kwargs) -> Observer:
    return Observer(
        api_key="k",
        workspace_id=1,
        agent_id="agent",
        call_id="call-1",
        forward_traces=False,
        **kwargs,
    )


# ── to_wav ────────────────────────────────────────────────────────────────────


def test_to_wav_wraps_pcm_without_altering_the_samples():
    pcm = bytes(range(256)) * 8  # whole number of stereo 16-bit frames

    result = to_wav(pcm, 24000, 2)

    assert result[:4] == b"RIFF"
    with wave.open(io.BytesIO(result)) as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == 24000
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == pcm


def test_to_wav_handles_mono():
    pcm = b"\x01\x00" * 50

    with wave.open(io.BytesIO(to_wav(pcm, 16000, 1))) as w:
        assert w.getnchannels() == 1
        assert w.getnframes() == 50


@pytest.mark.parametrize(
    ("audio", "rate", "channels"),
    [
        (b"\x00\x00", 0, 1),  # non-positive rate
        (b"\x00\x00", 16000, 0),  # non-positive channels
        (b"\x00\x00\x00", 16000, 2),  # not a whole number of frames
    ],
)
def test_to_wav_rejects_input_that_would_produce_a_broken_file(audio, rate, channels):
    with pytest.raises(ValueError):
        to_wav(audio, rate, channels)


# ── audio_buffer() ────────────────────────────────────────────────────────────


def test_audio_buffer_is_none_unless_recording_is_enabled():
    assert _observer().audio_buffer() is None


def test_audio_buffer_is_built_and_layout_is_derived_when_enabled():
    obs = _observer(recording_file_extension="wav")

    buffer = obs.audio_buffer()

    assert buffer is not None
    assert buffer.num_channels == 2
    # Derived from the buffer, not asserted by the caller.
    assert obs._config.recording_s3_multi_channel is True
    assert obs._config.recording_s3_caller_channel == 0


def test_mono_buffer_reports_no_channel_split():
    obs = _observer(recording_file_extension="wav")

    buffer = obs.audio_buffer(num_channels=1)

    assert buffer.num_channels == 1
    assert obs._config.recording_s3_multi_channel is False
    assert obs._config.recording_s3_caller_channel is None


def test_recording_fields_reach_the_payload():
    obs = _observer(recording_file_extension="wav")
    obs.audio_buffer()
    obs._acc.on_start(0)
    obs._acc.on_call_end(1_000_000_000)

    payload = obs._acc.build_payload(obs._config, None)

    assert payload.recording_file_extension == "wav"
    assert payload.recording_s3_multi_channel is True
    assert payload.recording_s3_caller_channel == 0


def test_payload_omits_recording_fields_when_not_opted_in():
    obs = _observer()
    obs._acc.on_start(0)
    obs._acc.on_call_end(1_000_000_000)

    sent = obs._acc.build_payload(obs._config, None).to_dict()

    assert "recording_file_extension" not in sent
    assert "recording_s3_multi_channel" not in sent
    assert "recording_s3_caller_channel" not in sent


# ── on_recording_upload ───────────────────────────────────────────────────────


async def _flush_with_response(obs, response):
    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        post_mock.return_value = response
        obs._acc.on_start(0)
        obs._acc.on_call_end(1_000_000_000)
        await obs._flush()


@pytest.mark.asyncio
async def test_upload_target_is_handed_to_the_application():
    obs = _observer(recording_file_extension="wav")
    seen = []

    @obs.event_handler("on_recording_upload")
    async def _(_observer, target):
        seen.append(target)

    await _flush_with_response(obs, {"recording_upload": TARGET})

    assert seen == [TARGET]


@pytest.mark.asyncio
async def test_the_sdk_never_uploads_the_recording_itself():
    """The whole point of the design: no HTTP beyond the call POST."""
    obs = _observer(recording_file_extension="wav")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as http_post:
        await _flush_with_response(obs, {"recording_upload": TARGET})

    http_post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [None, {}, {"recording_upload": None}, {"recording_upload": {}}, "not-a-dict"],
)
async def test_no_event_when_no_target_was_issued(response):
    obs = _observer(recording_file_extension="wav")
    seen = []

    @obs.event_handler("on_recording_upload")
    async def _(_observer, target):
        seen.append(target)

    await _flush_with_response(obs, response)

    assert seen == []


@pytest.mark.asyncio
async def test_a_raising_handler_does_not_break_the_flush():
    """Guaranteed by pipecat's _run_handler, which catches and logs. Pinned here
    because the flush runs in a task nobody awaits, so a handler that escaped
    would surface as an unretrieved-exception warning and nothing else."""
    obs = _observer(recording_file_extension="wav")

    @obs.event_handler("on_recording_upload")
    async def _(_observer, _target):
        raise RuntimeError("customer code blew up")

    # Must not propagate: _flush runs in a task nobody awaits.
    await _flush_with_response(obs, {"recording_upload": TARGET})


@pytest.mark.asyncio
async def test_flush_still_posts_the_call_when_recording_is_off():
    obs = _observer()

    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        post_mock.return_value = None
        obs._acc.on_start(0)
        obs._acc.on_call_end(1_000_000_000)
        await obs._flush()

    post_mock.assert_awaited_once()


# ── post_call return contract ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_call_returns_the_response_body():
    from tuner_pipecat_sdk.client import post_call

    obs = _observer()
    payload = MagicMock()
    payload.to_dict.return_value = {"call_id": "call-1"}
    payload.call_id = "call-1"
    payload.transcript_with_tool_calls = []

    response = MagicMock(status_code=200)
    response.json.return_value = {"recording_upload": TARGET}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as http_post:
        http_post.return_value = response
        assert await post_call(obs._config, payload) == {"recording_upload": TARGET}


@pytest.mark.asyncio
async def test_post_call_returns_none_for_a_non_json_body():
    from tuner_pipecat_sdk.client import post_call

    obs = _observer()
    payload = MagicMock()
    payload.to_dict.return_value = {"call_id": "call-1"}
    payload.call_id = "call-1"
    payload.transcript_with_tool_calls = []

    response = MagicMock(status_code=200)
    response.json.side_effect = ValueError("not json")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as http_post:
        http_post.return_value = response
        assert await post_call(obs._config, payload) is None

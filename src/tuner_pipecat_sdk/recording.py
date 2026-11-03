"""Helpers for turning pipecat's captured audio into an uploadable file.

The SDK never uploads a recording. It carries ``recording_file_extension`` in
the call payload, which makes the Tuner API issue a presigned POST target, and
hands that target to the ``on_recording_upload`` event handler — when and how
the audio is uploaded is the agent owner's decision, so an upload never runs on
the call's critical path or races the process exiting.

What the SDK does provide is this conversion, because it is the one place the
hand-off invites a silent mistake: pipecat's ``AudioBufferProcessor`` emits
**raw PCM**, not a WAV. It has no RIFF header. Posting those bytes to a target
opened with ``recording_file_extension="wav"`` produces a file that uploads
successfully and will not play.
"""

from __future__ import annotations

import io
import wave

__all__ = ["to_wav"]

# pipecat emits signed 16-bit PCM.
_SAMPLE_WIDTH_BYTES = 2


def to_wav(audio: bytes, sample_rate: int, num_channels: int) -> bytes:
    """Wrap raw PCM from ``AudioBufferProcessor`` in a WAV container.

    The three arguments are exactly what pipecat's ``on_audio_data`` /
    ``on_track_audio_data`` handlers are called with, in that order::

        @audio_buffer.event_handler("on_audio_data")
        async def _(buffer, audio, sample_rate, num_channels):
            wav_bytes = to_wav(audio, sample_rate, num_channels)

    Args:
        audio: Raw signed 16-bit PCM. Interleaved when ``num_channels`` > 1.
        sample_rate: Frames per second, as handed to the callback.
        num_channels: 1 for mixed mono, 2 for pipecat's caller-left /
            agent-right split.

    Returns:
        The same audio as a complete WAV file.

    Raises:
        ValueError: If ``sample_rate`` or ``num_channels`` is not positive, or
            if ``audio`` does not contain a whole number of frames — which
            means it is not the 16-bit PCM this expects, and writing it anyway
            would produce a file with a misaligned final frame.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")

    frame_bytes = _SAMPLE_WIDTH_BYTES * num_channels
    if len(audio) % frame_bytes:
        raise ValueError(
            f"audio is {len(audio)} bytes, not a whole number of "
            f"{frame_bytes}-byte frames ({num_channels}ch x 16-bit)"
        )

    buffer = io.BytesIO()
    # wave closes the file object it is given on __exit__, so getvalue() has to
    # happen after the block — BytesIO keeps its contents readable once closed
    # only if we read it first. Grab it inside the outer scope instead.
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(audio)
    return buffer.getvalue()

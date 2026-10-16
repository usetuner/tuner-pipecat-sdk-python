"""Upload a call recording to Tuner's storage.

Setting ``recording_file_extension`` on a call opts it into a presigned upload
target, returned as ``recording_upload`` in the submit response. That target is
an **S3 presigned POST**, not a PUT: the upload is multipart/form-data carrying
the policy fields verbatim, with the file appended last. S3 ignores anything
sent after the file part, so field order is part of the contract, not a style
choice.

The audio can be a path, raw bytes, or any binary file-like object, so callers
holding a provider's response in memory need not stage a temp file.

Usage::

    response = await observer.wait_for_submit()
    await upload_recording(response, Path("call.wav"), config)
    await upload_recording(response, wav_bytes, config, filename="call.wav")
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Any

import httpx
from loguru import logger


def _measure(stream: IO[bytes]) -> int | None:
    """Total byte length of a seekable stream, else None.

    Measured from the start, not the current position: httpx rewinds a seekable
    stream before sending it, so the whole thing is what actually goes up.
    """
    if not stream.seekable():
        return None
    start = stream.tell()
    end = stream.seek(0, io.SEEK_END)
    stream.seek(start)
    return end


async def upload_recording(
    submit_response: dict[str, Any] | None,
    audio: Path | str | bytes | bytearray | IO[bytes],
    config: Any,
    *,
    filename: str | None = None,
) -> bool:
    """POST a recording to the presigned target in a submit response.

    Args:
        submit_response: What ``submit_call`` returned. A response without a
            ``recording_upload`` key means the call was not opted in, which is
            not an error — the upload is simply skipped.
        audio: The audio to upload: a filesystem path, the raw bytes, or an
            open binary stream. A seekable stream is rewound and sent whole; a
            non-seekable one is sent from wherever it stands.
        config: TunerConfig (e.g. ``observer.config``), for the request timeout.
        filename: Name for the uploaded part. Defaults to the path's name, or
            to the target's key for bytes and streams.

    Returns:
        True if the recording was uploaded.
    """
    if not submit_response:
        return False

    target = submit_response.get("recording_upload")
    if not target:
        # No target: the call did not set recording_file_extension, or the API
        # declined to issue one.
        return False

    url = target.get("url")
    if not url:
        logger.error("[tuner] upload target has no url — skipping")
        return False

    key = target.get("key")
    max_bytes = target.get("max_bytes")

    handle: IO[bytes] | None = None
    payload: bytes | IO[bytes]
    size: int | None
    part_name = filename

    if isinstance(audio, str | Path):
        path = Path(audio)
        if not path.is_file():
            logger.error("[tuner] recording not found for upload — {}", path)
            return False
        size = path.stat().st_size
        handle = path.open("rb")
        payload = handle
        part_name = part_name or path.name
    elif isinstance(audio, bytes | bytearray):
        payload = bytes(audio)
        size = len(payload)
    else:
        payload = audio
        # A non-seekable stream cannot be measured without consuming it; leave
        # the size unknown and let S3's content-length-range policy enforce it.
        size = _measure(audio)

    part_name = part_name or (Path(key).name if key else "recording")

    # Only guard on a size we actually know.
    if max_bytes and size is not None and size > max_bytes:
        if handle is not None:
            handle.close()
        logger.error(
            "[tuner] recording is {} bytes, over the {} byte upload limit — skipping",
            size,
            max_bytes,
        )
        return False

    fields = target.get("fields") or {}
    content_type = target.get("content_type")

    # The file part must come last: S3 ignores fields that follow it.
    form: dict[str, Any] = {k: (None, str(v)) for k, v in fields.items()}
    form["file"] = (part_name, payload, content_type)

    try:
        async with httpx.AsyncClient(timeout=getattr(config, "timeout_seconds", 30.0)) as client:
            response = await client.post(url, files=form)
    except httpx.RequestError as exc:
        logger.error("[tuner] recording upload failed — {}", exc)
        return False
    finally:
        if handle is not None:
            handle.close()

    # A presigned POST answers 204 on success, or 201 when the policy asks for it.
    if response.status_code not in (200, 201, 204):
        logger.error(
            "[tuner] recording upload rejected {} — {}",
            response.status_code,
            response.text[:300],
        )
        return False

    logger.info(
        "[tuner] uploaded recording ({} bytes) → {}",
        size if size is not None else "unknown",
        key or "?",
    )
    return True

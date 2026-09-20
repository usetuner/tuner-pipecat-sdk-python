"""Transport concern: async fire-and-forget HTTP client for Tuner API.

Posts a completed call payload to POST /api/v1/public/call.
Never blocks or raises — all failures are logged and swallowed.

The response body is returned so callers can read ``recording_upload``, the
presigned POST target issued when the payload sets ``recording_file_extension``.
Every failure path returns ``None``: a caller that ignores the value behaves
exactly as it did before this was added.
"""

import json
from typing import Any

import httpx
from loguru import logger

from .config import TunerConfig
from .models import CallPayload


async def post_call(config: TunerConfig, payload: CallPayload) -> dict[str, Any] | None:
    """POST `payload` to Tuner. Fire-and-forget — see module docstring for failure behavior.

    Returns:
        The decoded response body, or ``None`` if the call already existed (409),
        the request failed, or the body was not JSON.
    """
    url = (
        f"{config.base_url}/api/v1/public/call"
        f"?workspace_id={config.workspace_id}"
        f"&agent_remote_identifier={config.agent_id}"
    )
    headers = {"Authorization": f"Bearer {config.api_key}"}

    payload_dict = payload.to_dict()
    payload_json = json.dumps(payload_dict, default=str)

    logger.info(
        "[tuner] sending call  call_id={}  transcript_messages={}  url={}  "
        "payload_bytes={}",
        payload.call_id,
        len(payload.transcript_with_tool_calls),
        url,
        len(payload_json),
    )
    logger.info("[tuner] request body: {}", payload_json)

    if config.debug:
        print("[tuner] --- request payload ---")
        print(json.dumps(payload_dict, indent=2, default=str))
        print("[tuner] --- end payload ---")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload_dict, headers=headers)

        if response.status_code == 409:
            logger.info(
                "[tuner] call {} already exists (409) — skipping",
                config.call_id,
            )
            return None

        if response.status_code >= 400:
            logger.error(
                "[tuner] POST → {}  call_id={}  response_body={}",
                response.status_code,
                config.call_id,
                response.text[:2000],
            )

        response.raise_for_status()

        logger.info(
            "[tuner] POST → {}  call_id={}",
            response.status_code,
            config.call_id,
        )

        try:
            body = response.json()
        except ValueError:
            # A 2xx with a non-JSON body is not an error for the call itself,
            # only for anything reading recording_upload out of it.
            logger.warning("[tuner] response for {} was not JSON", config.call_id)
            return None
        return body if isinstance(body, dict) else None

    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error(
            "[tuner] failed to deliver call {}: {}",
            config.call_id,
            exc,
        )
        return None

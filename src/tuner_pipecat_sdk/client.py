"""Transport concern: async fire-and-forget HTTP client for Tuner API.

Posts a completed call payload to POST /api/v1/public/call.
Never blocks or raises — all failures are logged and swallowed.
"""

import json
from typing import Any

import httpx
from loguru import logger

from .config import TunerConfig
from .models import CallPayload


async def post_call(config: TunerConfig, payload: CallPayload) -> dict[str, Any] | None:
    """POST `payload` to Tuner and return the parsed response, or None on failure."""
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
        # Carries `recording_upload` when recording_file_extension is set.
        try:
            parsed: Any = response.json()
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error(
            "[tuner] failed to deliver call {}: {}",
            config.call_id,
            exc,
        )
        return None

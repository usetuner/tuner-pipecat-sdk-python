"""Transport concern: async fire-and-forget HTTP client for Tuner API.

Posts a completed call payload to POST /api/v1/public/call.
Never blocks or raises — all failures are logged and swallowed.
"""

import json

import httpx
from loguru import logger

from .config import TunerConfig
from .models import CallPayload


async def post_call(config: TunerConfig, payload: CallPayload) -> None:
    url = (
        f"{config.base_url}/api/v1/public/call"
        f"?workspace_id={config.workspace_id}"
        f"&agent_remote_identifier={config.agent_id}"
    )
    headers = {"Authorization": f"Bearer {config.api_key}"}

    logger.info(
        "[flows-tuner] sending call  call_id={}  transcript_messages={}  url={}",
        payload.call_id,
        len(payload.transcript_with_tool_calls),
        url,
    )

    payload_dict = payload.to_dict()

    if config.debug:
        print("[flows-tuner] --- request payload ---")
        print(json.dumps(payload_dict, indent=2, default=str))
        print("[flows-tuner] --- end payload ---")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload_dict, headers=headers)

        if response.status_code == 409:
            logger.info(
                "[flows-tuner] call {} already exists (409) — skipping",
                config.call_id,
            )
            return

        response.raise_for_status()

        if config.debug:
            try:
                data = response.json()
                print(
                    f"[flows-tuner] POST → {response.status_code} "
                    f"id={data.get('id')} call_id={config.call_id}"
                )
            except Exception:
                print(f"[flows-tuner] POST → {response.status_code}")

    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error(
            "[flows-tuner] failed to deliver call {}: {}",
            config.call_id,
            exc,
        )

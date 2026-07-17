"""Telnyx Call Control API + Media Streaming server for the Nova Clinic bot.

Unlike TeXML (which auto-answers via XML), Telnyx **Call Control** sends
JSON events and requires our webhook to POST commands back to drive the
call.

Flow
----
1. Caller dials the SIP URI → Telnyx fires ``call.initiated`` (JSON).
2. We extract SIP info from the payload, cache it by ``call_control_id``,
   then POST ``actions/answer`` with ``stream_url`` to instruct Telnyx to
   both answer the call **and** start Media Streams.
3. Telnyx opens a WebSocket to ``/ws``.
4. ``/ws`` parses the handshake, recovers the cached SIP info, builds the
   transport, and hands off to ``bot.run_bot``.

Required env::

    TELNYX_API_KEY=...   # Mission Control Portal → Account Settings → API Keys

Run::

    cd examples/nova_clinic_pipecat
    ./.venv/bin/python telnyx_server.py
    # or: uvicorn telnyx_server:app --host 0.0.0.0 --port 7860

Point your Telnyx Call Control Application's webhook at
``https://<your-host>/`` (or ``/webhooks/inbound_call``) and route your SIP
Connection to that application.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import bot as bot_module
import httpx
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
from pipecat.runner.types import WebSocketRunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

app = FastAPI()

TELNYX_API_BASE = "https://api.telnyx.com/v2"


# Bridge the Call Control event webhook to the Media Streams WebSocket.
# Telnyx delivers SIP info on the webhook; the bot lives in the WS handler.
_PENDING: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _telnyx_headers() -> dict[str, str]:
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TELNYX_API_KEY env var is required for Call Control mode. "
            "Get one from Mission Control Portal → Account Settings → API Keys."
        )
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def _telnyx_action(
    call_control_id: str, action: str, body: dict[str, Any]
) -> None:
    """POST a Call Control action and log the result."""
    url = f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/{action}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=body, headers=_telnyx_headers())
    except Exception as e:
        logger.exception("[telnyx][api] {} failed: {}", action, e)
        return
    if r.status_code >= 300:
        logger.error(
            "[telnyx][api] {} → {}  body={}",
            action,
            r.status_code,
            r.text[:500],
        )
    else:
        logger.info("[telnyx][api] {} → {}", action, r.status_code)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/")
@app.post("/webhooks/inbound_call")
async def webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Telnyx Call Control webhook. Drives call answering + media streaming.

    Returns 200 immediately and schedules the answer API call as a background
    task so Telnyx never waits on us. Only ``state == "parked"`` legs are
    answered — outbound legs and legs already in flight return errors when
    you try to answer them.
    """
    try:
        data = await request.json()
    except Exception as e:
        raw = await request.body()
        logger.warning(
            "[telnyx][webhook] could not parse JSON body: {}  raw[:300]={!r}",
            e,
            raw[:300],
        )
        return Response(status_code=200)

    event = data.get("data") or {}
    event_type = event.get("event_type") or ""
    payload = event.get("payload") or {}
    call_control_id = payload.get("call_control_id") or ""

    logger.info(
        "[telnyx][webhook] event={!r}  call_control_id={!r}  state={!r}  "
        "direction={!r}  from={!r}  to={!r}  call_leg_id={!r}",
        event_type,
        call_control_id,
        payload.get("state"),
        payload.get("direction"),
        payload.get("from"),
        payload.get("to"),
        payload.get("call_leg_id"),
    )

    # Only act on call.initiated for an answerable leg. Telnyx routes a single
    # SIP call as TWO call_control_ids: an inbound leg (state=parked, can be
    # answered) and an outbound leg (state != parked, returns 90102 if you
    # try to answer). We ignore everything that isn't parked.
    state = payload.get("state")
    if event_type != "call.initiated":
        return Response(status_code=200)
    if state != "parked":
        logger.info(
            "[telnyx][webhook] skip leg  state={!r}  call_control_id={!r}",
            state,
            call_control_id,
        )
        return Response(status_code=200)

    # Cache the raw Call Control event so the WS handler can read SIP info
    # from it. The full event (with ``data.payload`` wrapper) is what the
    # bot receives.
    if call_control_id:
        _PENDING[call_control_id] = data

    logger.info(
        "[telnyx][webhook] cached call_control_id={!r}  from={!r}  to={!r}",
        call_control_id,
        payload.get("from"),
        payload.get("to"),
    )

    # Fire the answer API call in the background so we return 200 NOW.
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = os.getenv("PUBLIC_WS_URL") or f"wss://{host}/ws"
    background_tasks.add_task(
        _telnyx_action,
        call_control_id,
        "answer",
        {
            "stream_url": ws_url,
            "stream_track": "both_tracks",
            "stream_bidirectional_mode": "rtp",
        },
    )

    return Response(status_code=200)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Telnyx Media Streams WebSocket — build transport, recover SIP info, run bot."""
    await websocket.accept()
    logger.info("[telnyx][ws] websocket accepted")

    _, sip_call_data = await parse_telephony_websocket(websocket)
    call_control_id = sip_call_data.get("call_control_id") or ""

    # Hand the bot either the cached Call Control event (richer — has SIP
    # headers) or the WS call_data (leaner — falls back to call_control_id).
    sip_payload = _PENDING.pop(call_control_id, sip_call_data)

    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        serializer=TelnyxFrameSerializer(
            stream_id=sip_call_data["stream_id"],
            call_control_id=sip_call_data["call_control_id"],
            outbound_encoding=sip_call_data["outbound_encoding"],
            inbound_encoding="PCMU",
            api_key=os.getenv("TELNYX_API_KEY") or None,
            params=TelnyxFrameSerializer.InputParams(auto_hang_up=False),
        ),
    )
    transport = FastAPIWebsocketTransport(websocket=websocket, params=params)

    runner_args = WebSocketRunnerArguments(websocket=websocket)
    runner_args.handle_sigint = False
    await bot_module.run_bot(
        transport,
        runner_args,
        sip_call_data=sip_payload,
        sip_provider="telnyx",
    )


@app.get("/")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    if not os.getenv("TELNYX_API_KEY"):
        logger.warning(
            "TELNYX_API_KEY is not set — webhook will fail to answer calls. "
            "Get a key from Mission Control Portal → Account Settings → API Keys."
        )

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "DEBUG"))
    uvicorn.run(
        "telnyx_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )

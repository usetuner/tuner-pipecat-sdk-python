"""Twilio webhook + Media Streams WebSocket server for the Nova Clinic bot.

Owns everything Twilio-specific:
* ``POST /twiml`` (and ``POST /``) — reads the SIP data with ``extract_sip``
  and returns ``forward_to_websocket(...).twiml``, which puts ``SipCallId``
  and Tuner's ``X-Correlation-Id`` on the stream as ``<Parameter>`` tags.
* ``WebSocket /ws`` — reads them back from the ``start`` message with
  ``extract_sip``, builds the ``TwilioFrameSerializer`` +
  ``FastAPIWebsocketTransport``, then hands the ``SipInfo`` to
  ``bot.run_bot``.

Run::

    cd examples/nova_clinic_pipecat
    ./.venv/bin/python twilio_server.py
    # or: uvicorn twilio_server:app --host 0.0.0.0 --port 7860

Point the Voice URL of the Twilio SIP Domain that Tuner dials at
``https://<your-host>/twiml`` (or ``/`` — both are served).
"""

from __future__ import annotations

import os
import sys

import bot as bot_module
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
from pipecat.runner.types import WebSocketRunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from tuner_pipecat_sdk import extract_sip, forward_to_websocket

app = FastAPI()


@app.post("/")
@app.post("/twiml")
async def twiml(request: Request) -> Response:
    """Twilio voice webhook → enriched TwiML with SIP fields forwarded."""
    form = await request.form()

    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = os.getenv("PUBLIC_WS_URL") or f"wss://{host}/ws"

    info = extract_sip(dict(form), provider="twilio")
    fwd = forward_to_websocket(info, ws_url=ws_url)

    logger.info(
        "[twilio][twiml] CallSid={!r}  has_SipCallId={}  has_correlation_id={}  ws_url={}",
        info.carrier_call_id,
        not info.is_fallback,
        bool(info.correlation_id),
        ws_url,
    )
    if info.is_fallback:
        logger.warning(
            "[twilio][twiml] no SipCallId on form — call may not be "
            "SIP-originated. Bot will only see CallSid."
        )
    return Response(content=fwd.twiml, media_type="application/xml")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Twilio Media Streams WebSocket — build transport, then hand off to bot."""
    await websocket.accept()
    logger.info("[twilio][ws] websocket accepted")

    _, sip_call_data = await parse_telephony_websocket(websocket)
    sip_info = extract_sip(sip_call_data, provider="twilio")
    logger.info(
        "[twilio][ws] resolved SIP data  call_sid={!r}  sip_call_id={!r}  "
        "header_keys={}",
        sip_info.carrier_call_id,
        sip_info.sip_call_id,
        list(sip_info.sip_headers.keys()),
    )

    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        serializer=TwilioFrameSerializer(
            stream_sid=sip_call_data["stream_id"],
            call_sid=sip_call_data["call_id"],
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        ),
    )
    transport = FastAPIWebsocketTransport(websocket=websocket, params=params)

    runner_args = WebSocketRunnerArguments(websocket=websocket)
    runner_args.handle_sigint = False
    await bot_module.run_bot(transport, runner_args, sip_info=sip_info)


@app.get("/")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "DEBUG"))
    uvicorn.run(
        "twilio_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )

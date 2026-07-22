"""Twilio webhook + Media Streams WebSocket server for the Nova Clinic bot.

Owns everything Twilio-specific:
* ``POST /twiml`` (and ``POST /``) — returns TwiML that forwards SIP-layer
  fields (``SipCallId``, ``Caller``, every ``SipHeader_*``) to the WebSocket
  as ``<Parameter>`` tags. We use the SDK's ``build_sip_forwarding_twiml``
  helper so the XML stays correct and escaped.
* ``WebSocket /ws`` — parses the Twilio handshake into a
  ``TwilioCallContext``, builds the ``TwilioFrameSerializer`` +
  ``FastAPIWebsocketTransport``, then hands the typed context off to
  ``bot.run_bot``.

Run::

    cd examples/nova_clinic_pipecat
    ./.venv/bin/python twilio_server.py
    # or: uvicorn twilio_server:app --host 0.0.0.0 --port 7860

Point Twilio's Voice URL at ``https://<your-host>/twiml`` (or ``/`` —
both are served).
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

from tuner_pipecat_sdk.providers.twilio import (
    TwilioCallContext,
    build_sip_forwarding_twiml,
)

app = FastAPI()


@app.post("/")
@app.post("/twiml")
async def twiml(request: Request) -> Response:
    """Twilio voice webhook → enriched TwiML with SIP fields forwarded."""
    form = await request.form()

    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = os.getenv("PUBLIC_WS_URL") or f"wss://{host}/ws"

    xml = build_sip_forwarding_twiml(form, ws_url=ws_url)

    logger.info(
        "[twilio][twiml] CallSid={!r}  has_SipCallId={}  ws_url={}",
        form.get("CallSid"),
        bool(form.get("SipCallId")),
        ws_url,
    )
    if not form.get("SipCallId"):
        logger.warning(
            "[twilio][twiml] no SipCallId on form — call may not be "
            "SIP-originated. Bot will only see CallSid."
        )
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Twilio Media Streams WebSocket — build transport, then hand off to bot."""
    await websocket.accept()
    logger.info("[twilio][ws] websocket accepted")

    _, sip_call_data = await parse_telephony_websocket(websocket)
    ctx = TwilioCallContext.from_call_data(sip_call_data)
    logger.info(
        "[twilio][ws] resolved context  call_sid={!r}  sip_call_id={!r}  "
        "header_keys={}",
        ctx.call_sid,
        ctx.sip_call_id,
        list(ctx.raw_headers.keys()),
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
    await bot_module.run_bot(transport, runner_args, sip_context=ctx)


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

"""Twilio webhook + Media Streams WebSocket server for the Nova Clinic bot.

Owns everything Twilio-specific:
* ``POST /twiml`` (and ``POST /``) — returns TwiML that forwards SIP-layer
  fields (``SipCallId``, ``Caller``, every ``SipHeader_*``) to the WebSocket
  as ``<Parameter>`` tags so the SDK can pick them up.
* ``WebSocket /ws`` — parses the Twilio handshake, builds the
  ``TwilioFrameSerializer`` + ``FastAPIWebsocketTransport``, then hands off
  to the provider-agnostic ``bot.run_bot``.

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

import bot as bot_module

app = FastAPI()


# Twilio voice-webhook fields we forward as <Parameter> tags. Anything
# starting with ``SipHeader_`` is also forwarded automatically (Twilio's
# convention for arbitrary SIP headers received on the inbound INVITE).
_FORWARDED_FIELDS: tuple[str, ...] = (
    "SipCallId",
    "Caller",
    "Called",
    "CallSid",
    "From",
    "To",
    "AccountSid",
    "Direction",
)


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@app.post("/")
@app.post("/twiml")
async def twiml(request: Request) -> Response:
    """Twilio voice webhook → enriched TwiML with SIP fields forwarded."""
    form = await request.form()

    forwarded: dict[str, str] = {}
    for field in _FORWARDED_FIELDS:
        v = form.get(field)
        if v:
            forwarded[field] = str(v)
    for key, value in form.items():
        if key.startswith("SipHeader_") and value:
            forwarded[key] = str(value)

    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = os.getenv("PUBLIC_WS_URL") or f"wss://{host}/ws"

    params_xml = "".join(
        f'      <Parameter name="{_xml_escape(k)}" value="{_xml_escape(v)}"/>\n'
        for k, v in forwarded.items()
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{_xml_escape(ws_url)}">
{params_xml.rstrip()}
    </Stream>
  </Connect>
  <Pause length="40"/>
</Response>"""

    logger.info(
        "[twilio][twiml] forwarded_keys={}  ws_url={}",
        list(forwarded.keys()),
        ws_url,
    )
    if "SipCallId" not in forwarded:
        logger.warning(
            "[twilio][twiml] no SipCallId on form — call may not be "
            "SIP-originated. SDK will fall back to CallSid."
        )
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Twilio Media Streams WebSocket — build transport, then hand off to bot."""
    await websocket.accept()
    logger.info("[twilio][ws] websocket accepted")

    _, sip_call_data = await parse_telephony_websocket(websocket)
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
    await bot_module.run_bot(transport, runner_args, sip_call_data=sip_call_data)


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


"""Jambonz call hook + audio WebSocket server for the Nova Clinic bot.

Jambonz is an open-source SIP application server. Unlike Twilio/Telnyx,
Pipecat ships no built-in Jambonz serializer, so we provide one inline
here.

Flow
----
1. SIP call arrives at Jambonz. Jambonz's Application is configured to POST
   a "call hook" to ``https://<your-host>/`` on inbound calls.
2. We capture the SIP info from the JSON body (``call_sid``, ``sip.headers``,
   ``from``, ``to``) and respond with a verb array that instructs Jambonz to
   bridge audio to our WebSocket at ``wss://<your-host>/ws`` via the
   ``listen`` verb (forks audio to the WS).
3. Jambonz opens the WebSocket. We use a custom
   ``JambonzFrameSerializer`` to convert the L16 PCM frames into Pipecat
   ``InputAudioRawFrame`` and back.

Caveats
-------
* Audio framing for the ``listen`` verb is **unidirectional** (Jambonz → us)
  by default. For full bidirectional audio (so the bot can talk back), this
  example uses the ``dial`` verb with ``type: "ws"`` if your Jambonz build
  supports it. If your Jambonz version is older / only supports unidirectional
  ``listen``, change the verb in ``_jambonz_verbs`` accordingly and route the
  bot's audio another way (or upgrade Jambonz).
* The serializer assumes raw 16-bit signed little-endian PCM mono frames.
  Adjust ``JAMBONZ_SAMPLE_RATE`` if your Jambonz Application is configured
  for 8 kHz instead of 16 kHz.

Run::

    cd examples/nova_clinic_pipecat
    ./.venv/bin/python jambonz_server.py
    # or: uvicorn jambonz_server:app --host 0.0.0.0 --port 7860

In the Jambonz portal, configure your Application's Calling Webhook to
``https://<your-host>/`` (POST). No API key required for inbound; the
``listen``/``dial`` verbs are returned in-band on the webhook response.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.runner.types import WebSocketRunnerArguments
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

import bot as bot_module

app = FastAPI()


# Audio rate Jambonz is sending us. Set on the Application side (or in the
# verb's ``sampleRate`` parameter). 16 kHz is Jambonz's default for the
# ``listen``/``dial`` audio fork; 8 kHz is also valid for narrowband SIP.
JAMBONZ_SAMPLE_RATE = int(os.getenv("JAMBONZ_SAMPLE_RATE", "16000"))


# ---------------------------------------------------------------------------
# Inline serializer (Pipecat has no built-in Jambonz)
# ---------------------------------------------------------------------------


class JambonzFrameSerializer(FrameSerializer):
    """Serialize/deserialize raw L16 PCM frames over Jambonz's WebSocket."""

    def __init__(
        self,
        call_sid: str,
        jambonz_sample_rate: int = JAMBONZ_SAMPLE_RATE,
        params: Optional["FrameSerializer.InputParams"] = None,
    ) -> None:
        super().__init__(params or FrameSerializer.InputParams())
        self._call_sid = call_sid
        self._jambonz_sample_rate = jambonz_sample_rate
        self._pipeline_sample_rate = 0
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

    async def setup(self, frame: StartFrame) -> None:
        self._pipeline_sample_rate = frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, AudioRawFrame):
            data = frame.audio
            resampled = await self._output_resampler.resample(
                data, frame.sample_rate, self._jambonz_sample_rate
            )
            if not resampled:
                return None
            # Jambonz expects raw binary L16 PCM frames (no JSON envelope,
            # no base64). Bidirectional ``dial`` with ``type: "ws"`` accepts
            # bytes back the same way.
            return resampled
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, str):
            # First message is typically JSON metadata (call info). Log and
            # skip — we already have the SIP info from the webhook hook.
            logger.debug("[jambonz][ws] non-audio frame: {}", data[:300])
            return None
        if not data:
            return None
        resampled = await self._input_resampler.resample(
            data, self._jambonz_sample_rate, self._pipeline_sample_rate
        )
        if not resampled:
            return None
        return InputAudioRawFrame(
            audio=resampled,
            sample_rate=self._pipeline_sample_rate,
            num_channels=1,
        )

    @property
    def type(self) -> str:
        return "binary"


# ---------------------------------------------------------------------------
# Webhook → WS bridge: park SIP info keyed by call_sid
# ---------------------------------------------------------------------------


_PENDING: dict[str, dict[str, Any]] = {}


def _extract_sip_info(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull SIP info out of Jambonz's webhook payload.

    Jambonz exposes the SIP INVITE headers under ``payload["sip"]["headers"]``
    as a flat ``{name: value}`` dict — much cleaner than Twilio's flow.
    """
    sip = payload.get("sip") or {}
    raw_headers = sip.get("headers") or {}

    headers: dict[str, str] = {}
    if isinstance(raw_headers, dict):
        for k, v in raw_headers.items():
            if v is None:
                continue
            headers[str(k)] = str(v)
    elif isinstance(raw_headers, list):
        # Some Jambonz versions use [{name, value}] list shape.
        for item in raw_headers:
            if isinstance(item, dict):
                name = item.get("name")
                value = item.get("value")
                if name and value is not None:
                    headers[str(name)] = str(value)

    # Native call-level fields useful as context
    for k in ("from", "to", "direction", "callerName"):
        v = payload.get(k)
        if v:
            headers.setdefault(k, str(v))

    # Pick the identifier downstream systems use to correlate the call.
    # Our chain is LiveKit SIP → Jambonz → us, so LiveKit's ``X-CID`` is the
    # canonical id for cross-system lookup (Jambonz's ``Call-ID`` is just a
    # SIP-layer transaction id, regenerated by Jambonz on each hop).
    # Adjust the priority below for your trunk if you want SIP ``Call-ID``
    # or some other ``X-*`` header instead.
    sip_call_id = (
        headers.get("X-CID")
        or headers.get("x-cid")
        or headers.get("Call-ID")
        or headers.get("call-id")
        or headers.get("SipCallId")
        or sip.get("call_id")
    )
    return {"sip_call_id": sip_call_id, "sip_headers": headers or None}


def _jambonz_verbs(ws_url: str, sip_call_id: str | None) -> list[dict[str, Any]]:
    """Build the verb array Jambonz will execute for this call.

    Uses Jambonz's ``listen`` verb with ``bidirectionalAudio`` enabled. This
    is Jambonz's canonical mechanism for bot-style integrations: the call is
    answered, audio is forked to our WebSocket, AND audio sent back from our
    WS is injected into the call so the bot can talk.

    Reference: https://www.jambonz.org/docs/webhooks/listen/
    """
    metadata = {"sip_call_id": sip_call_id} if sip_call_id else {}
    return [
        {
            "verb": "listen",
            "url": ws_url,
            "mixType": "mono",
            "sampleRate": JAMBONZ_SAMPLE_RATE,
            "bidirectionalAudio": {
                "enabled": True,
                "streaming": True,
                "sampleRate": JAMBONZ_SAMPLE_RATE,
            },
            "metadata": metadata or None,
        }
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/")
@app.post("/call-hook")
async def call_hook(request: Request) -> JSONResponse:
    """Jambonz application call hook → capture SIP info + return verb array."""
    try:
        data = await request.json()
    except Exception as e:
        raw = await request.body()
        logger.warning(
            "[jambonz][webhook] could not parse JSON: {}  raw[:300]={!r}",
            e,
            raw[:300],
        )
        return JSONResponse(content=[], status_code=200)

    call_sid = str(data.get("call_sid") or data.get("callSid") or "")
    sip_info = _extract_sip_info(data)

    if call_sid:
        _PENDING[call_sid] = sip_info

    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    ws_url = os.getenv("PUBLIC_WS_URL") or f"wss://{host}/ws"

    logger.info(
        "[jambonz][webhook] call_sid={!r}  sip_call_id={!r}  header_keys={}  ws_url={}",
        call_sid,
        sip_info["sip_call_id"],
        list((sip_info["sip_headers"] or {}).keys()),
        ws_url,
    )

    verbs = _jambonz_verbs(ws_url=ws_url, sip_call_id=sip_info["sip_call_id"])
    # Drop None-valued keys for a clean response
    cleaned = [{k: v for k, v in verb.items() if v is not None} for verb in verbs]
    return JSONResponse(content=cleaned, status_code=200)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Jambonz audio WebSocket — build transport + serializer, run bot."""
    await websocket.accept()
    logger.info("[jambonz][ws] websocket accepted")

    # Jambonz's first WS message is typically JSON metadata containing the
    # call_sid; receive and parse it so we can correlate with the webhook.
    sip_call_data: dict[str, Any] = {}
    call_sid = ""
    try:
        first = await websocket.receive()
        if "text" in first and first["text"]:
            import json as _json

            try:
                meta = _json.loads(first["text"])
                call_sid = str(meta.get("call_sid") or meta.get("callSid") or "")
                logger.info(
                    "[jambonz][ws] first metadata  call_sid={!r}  keys={}",
                    call_sid,
                    list(meta.keys()) if isinstance(meta, dict) else None,
                )
            except Exception as e:
                logger.warning("[jambonz][ws] non-JSON first frame: {}", e)
    except Exception as e:
        logger.error("[jambonz][ws] failed reading first frame: {}", e)
        await websocket.close()
        return

    entry = _PENDING.pop(call_sid, None) if call_sid else None
    if entry:
        # Inject into ``body`` so the SDK's ``attach_sip_from_telephony``
        # auto-extraction picks it up.
        body: dict[str, str] = {}
        if entry.get("sip_call_id"):
            body["SipCallId"] = str(entry["sip_call_id"])
        if entry.get("sip_headers"):
            for k, v in entry["sip_headers"].items():
                body.setdefault(k, v)
        sip_call_data = {"call_id": call_sid, "body": body}
        logger.info(
            "[jambonz][ws] recovered SIP info  sip_call_id={!r}  header_keys={}",
            entry.get("sip_call_id"),
            list((entry.get("sip_headers") or {}).keys()),
        )
    else:
        sip_call_data = {"call_id": call_sid}
        logger.warning(
            "[jambonz][ws] no cached SIP info for call_sid={!r} — SDK will "
            "fall back to call_sid as sip_call_id",
            call_sid,
        )

    params = FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        add_wav_header=False,
        serializer=JambonzFrameSerializer(call_sid=call_sid),
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
        "jambonz_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )

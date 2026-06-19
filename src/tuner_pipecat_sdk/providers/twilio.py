"""Twilio integration helpers.

Twilio splits a call's SIP context across two transports:

* The **HTTP voice webhook** (form-encoded) receives ``CallSid``,
  ``Caller``, ``Called``, every ``SipHeader_*`` from the inbound INVITE,
  and ``SipCallId`` when the call came in over a SIP trunk. We respond
  with TwiML that opens the Media Streams WebSocket.
* The **Media Streams WebSocket** drops those fields by default. To get
  them across, the TwiML ``<Stream>`` element must carry ``<Parameter>``
  tags forwarding each value.

This module gives customers the two helpers they need:

* :func:`build_sip_forwarding_twiml` — builds a TwiML response that
  forwards every SIP-relevant field as a ``<Parameter>`` tag. Drop into
  the webhook handler in one line.
* :class:`TwilioCallContext` — typed view of the WS ``call_data`` produced
  by Pipecat's ``parse_telephony_websocket``. Built once on the WS side,
  handed to the bot, then to the Tuner observer.

Typical wiring (in the customer's server)::

    from tuner_pipecat_sdk.providers.twilio import (
        TwilioCallContext, build_sip_forwarding_twiml,
    )

    @app.post("/twiml")
    async def twiml(request):
        form = await request.form()
        xml = build_sip_forwarding_twiml(form, ws_url=...)
        return Response(content=xml, media_type="application/xml")

    @app.websocket("/ws")
    async def ws(websocket):
        _, call_data = await parse_telephony_websocket(websocket)
        ctx = TwilioCallContext.from_call_data(call_data)
        await run_bot(transport, sip_context=ctx)

The customer's bot then hands the context to Tuner with one call::

    observer.attach_sip_from_context(ctx)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from xml.sax.saxutils import escape

# Standard Twilio voice-webhook fields that carry SIP context. Anything
# starting with ``SipHeader_`` is also forwarded automatically (Twilio's
# convention for arbitrary SIP headers from the inbound INVITE).
_DEFAULT_FORWARDED_FIELDS: tuple[str, ...] = (
    "SipCallId",
    "Caller",
    "Called",
    "CallSid",
    "From",
    "To",
    "AccountSid",
    "Direction",
)

# Case-insensitive aliases the SIP-layer Call-ID is commonly forwarded under
# inside the WS ``customParameters`` (Twilio's ``body``).
_SIP_CALL_ID_ALIASES: tuple[str, ...] = (
    "sipcallid",
    "sip_call_id",
    "sip-call-id",
    "x-sip-call-id",
)


@dataclass
class TwilioCallContext:
    """Typed snapshot of a Twilio call's identity + SIP headers.

    Built from the WS ``call_data`` returned by Pipecat's
    ``parse_telephony_websocket`` via :meth:`from_call_data`.
    """

    call_sid: str
    sip_call_id: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    stream_sid: str | None = None
    raw_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_call_data(cls, call_data: dict[str, Any]) -> TwilioCallContext:
        """Extract a context from ``parse_telephony_websocket`` output.

        Twilio's ``call_data`` shape::

            {
                "stream_id": "MZ-...",
                "call_id":   "CA-...",          # CallSid
                "body": {                       # forwarded <Parameter> tags
                    "SipCallId": "...",
                    "Caller":    "...",
                    "CallSid":   "...",
                    "SipHeader_X-Trace": "...",
                }
            }

        SIP Call-ID resolution: any case-insensitive ``SipCallId`` alias in
        ``body`` → fall back to Twilio's CallSid.

        ``raw_headers`` is the ``body`` dict normalized to string values
        (empty when the TwiML didn't include ``<Parameter>`` tags).
        """
        if not isinstance(call_data, dict):
            return cls.fallback("")

        body = call_data.get("body") if isinstance(call_data.get("body"), dict) else {}
        headers: dict[str, str] = {}
        for k, v in body.items():
            if v is None:
                continue
            headers[str(k)] = str(v)

        lowered = {k.lower(): v for k, v in headers.items()}
        sip_call_id: str | None = None
        for alias in _SIP_CALL_ID_ALIASES:
            v = lowered.get(alias)
            if v:
                sip_call_id = v
                break

        call_sid = str(
            call_data.get("call_id") or headers.get("CallSid") or ""
        )
        if not sip_call_id:
            sip_call_id = call_sid or None

        return cls(
            call_sid=call_sid,
            sip_call_id=sip_call_id,
            from_number=headers.get("Caller") or headers.get("From"),
            to_number=headers.get("Called") or headers.get("To"),
            stream_sid=str(call_data.get("stream_id") or "") or None,
            raw_headers=headers,
        )

    @classmethod
    def fallback(cls, call_sid: str) -> TwilioCallContext:
        """Minimal context when ``call_data`` is missing or malformed."""
        return cls(
            call_sid=call_sid,
            sip_call_id=call_sid or None,
            raw_headers={},
        )


def _attr(value: str) -> str:
    """Escape a string for use inside a double-quoted XML attribute."""
    return escape(str(value), {'"': "&quot;"})


def build_sip_forwarding_twiml(
    form: Mapping[str, Any],
    *,
    ws_url: str,
    forwarded_fields: tuple[str, ...] = _DEFAULT_FORWARDED_FIELDS,
    extra_params: Mapping[str, str] | None = None,
    pause_seconds: int = 40,
) -> str:
    """Build a TwiML response that forwards SIP fields to your WebSocket.

    Twilio's HTTP voice webhook receives SIP context on the form body, but
    the Media Streams WebSocket drops those fields by default. This
    function produces the
    ``<Response><Connect><Stream>...</Stream></Connect></Response>``
    XML with one ``<Parameter>`` tag per SIP-relevant value, so the
    customer's WS handler — and Tuner observer — see the full picture.

    :param form: Twilio's form-encoded webhook payload. FastAPI's
        ``await request.form()`` works directly; any mapping does.
    :param ws_url: The WebSocket URL Twilio should connect to.
    :param forwarded_fields: Fields to forward explicitly. The default set
        covers the standard SIP context. Any key matching ``SipHeader_*``
        on the form is forwarded regardless of this list.
    :param extra_params: Additional name/value pairs to forward (e.g. a
        custom routing key your bot will read off ``raw_headers``).
    :param pause_seconds: Length of the trailing ``<Pause>`` verb so
        Twilio holds the call open while your bot runs.

    The returned string is well-formed XML; pass it as
    ``Response(content=xml, media_type="application/xml")``.
    """
    forwarded: dict[str, str] = {}
    for field_name in forwarded_fields:
        v = form.get(field_name)
        if v:
            forwarded[field_name] = str(v)
    for key, value in form.items():
        key_str = str(key)
        if key_str.startswith("SipHeader_") and value:
            forwarded[key_str] = str(value)
    if extra_params:
        for k, v in extra_params.items():
            if v is not None:
                forwarded[str(k)] = str(v)

    params_xml = "".join(
        f'      <Parameter name="{_attr(k)}" value="{_attr(v)}"/>\n'
        for k, v in forwarded.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        "  <Connect>\n"
        f'    <Stream url="{_attr(ws_url)}">\n'
        f"{params_xml}"
        "    </Stream>\n"
        "  </Connect>\n"
        f'  <Pause length="{int(pause_seconds)}"/>\n'
        "</Response>"
    )


__all__ = ["TwilioCallContext", "build_sip_forwarding_twiml"]

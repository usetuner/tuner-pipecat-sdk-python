"""Provider-specific SIP metadata extractors.

Each extractor takes the raw payload that the user's server receives from
its telephony provider and returns ``(sip_call_id, sip_headers)``. The
public entrypoint is ``attach_sip_from_telephony(payload, provider=...)``
on the observer — the user never imports anything from here directly.

The goal is that the user's server file contains zero tuner-specific
extraction logic. They pass the provider's payload through verbatim and
the SDK does the rest.

If a provider is not listed here, callers can pass their own callable
matching ``ProviderExtractor`` to ``attach_sip_from_telephony``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProviderExtractor = Callable[
    [dict[str, Any]], tuple[str | None, dict[str, str] | None]
]


# Case-insensitive aliases the SIP-layer Call-ID is commonly forwarded under
# inside customParameters / extra_headers / WS body dicts.
_SIP_CALL_ID_ALIASES: tuple[str, ...] = (
    "sipcallid",
    "sip_call_id",
    "sip-call-id",
    "x-sip-call-id",
)


def _find_in_aliases(params: dict[str, Any]) -> str | None:
    """Case-insensitive lookup across known SIP Call-ID aliases."""
    if not params:
        return None
    lowered = {str(k).lower(): v for k, v in params.items()}
    for alias in _SIP_CALL_ID_ALIASES:
        v = lowered.get(alias)
        if v:
            return str(v)
    return None


def _stringify_headers(raw: Any) -> dict[str, str]:
    """Normalize a headers value into a flat ``{str: str}`` dict.

    Accepts a dict ``{name: value}`` (the common Jambonz shape) or a list
    ``[{name, value}]`` (older Jambonz versions). Drops ``None`` values.
    """
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is None:
                continue
            out[str(k)] = str(v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name")
                value = item.get("value")
                if name and value is not None:
                    out[str(name)] = str(value)
    return out


# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------


def _extract_twilio(
    payload: dict[str, Any],
) -> tuple[str | None, dict[str, str] | None]:
    """Twilio Media Streams ``call_data`` extraction.

    Thin wrapper around :class:`TwilioCallContext.from_call_data` so the
    legacy ``attach_sip_from_telephony(payload, provider="twilio")`` path
    stays consistent with the typed ``attach_sip_from_context`` path.
    """
    from .providers.twilio import TwilioCallContext

    ctx = TwilioCallContext.from_call_data(payload)
    return (ctx.sip_call_id, ctx.raw_headers or None)


# ---------------------------------------------------------------------------
# Telnyx
# ---------------------------------------------------------------------------


def _extract_telnyx(
    payload: dict[str, Any],
) -> tuple[str | None, dict[str, str] | None]:
    """Telnyx — accepts either Call Control webhook or Media Streams ``call_data``.

    **Call Control webhook** (the JSON event POSTed to your webhook URL)::

        {"data": {"event_type": "call.initiated",
                  "payload": {"call_control_id": "...",
                              "sip_call_id": "...",
                              "custom_headers": [{"name", "value"}, ...],
                              "sip_headers":   [{"name", "value"}, ...],
                              "from": "...", "to": "...", ...}}}

    **Media Streams WS** (``parse_telephony_websocket`` output)::

        {"stream_id": "...", "call_control_id": "...",
         "customParameters": {"SipCallId": "...", ...}, ...}

    The two shapes are detected by which keys are present; pick whichever
    your server-side flow naturally has and pass it through verbatim.
    """
    if not isinstance(payload, dict):
        return (None, None)

    # Call Control webhook (with or without the ``data`` envelope).
    inner: dict[str, Any] | None = None
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("payload"), dict):
        inner = data["payload"]
    elif "custom_headers" in payload or "sip_headers" in payload:
        inner = payload

    if inner is not None:
        headers: dict[str, str] = {}
        for key in ("custom_headers", "sip_headers"):
            items = inner.get(key) or []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = item.get("name")
                        value = item.get("value")
                        if name and value is not None:
                            headers.setdefault(str(name), str(value))
        for k in ("from", "to", "caller_id_name", "caller_id_number"):
            v = inner.get(k)
            if v:
                headers.setdefault(k, str(v))
        sip_call_id = (
            inner.get("sip_call_id")
            or _find_in_aliases(headers)
            or headers.get("Call-ID")
            or headers.get("call-id")
            or inner.get("call_control_id")
        )
        return (
            str(sip_call_id) if sip_call_id else None,
            headers or None,
        )

    # Media Streams WS call_data.
    params = payload.get("customParameters")
    sip_layer = _find_in_aliases(params) if isinstance(params, dict) else None
    call_id = sip_layer or payload.get("call_control_id") or payload.get("call_id")
    headers = (
        _stringify_headers(params) if isinstance(params, dict) and params else None
    )
    return (str(call_id) if call_id else None, headers or None)


# ---------------------------------------------------------------------------
# Plivo
# ---------------------------------------------------------------------------


def _extract_plivo(
    payload: dict[str, Any],
) -> tuple[str | None, dict[str, str] | None]:
    """Plivo Media Streams ``call_data``.

    Forwarded SIP fields land under ``customParameters`` (or
    ``extra_headers`` on some versions). Native fallback is ``call_id``.
    """
    params = payload.get("customParameters") or payload.get("extra_headers")
    sip_layer = _find_in_aliases(params) if isinstance(params, dict) else None
    call_id = sip_layer or payload.get("call_id")
    headers = (
        _stringify_headers(params) if isinstance(params, dict) and params else None
    )
    return (str(call_id) if call_id else None, headers or None)


# ---------------------------------------------------------------------------
# Exotel
# ---------------------------------------------------------------------------


def _extract_exotel(
    payload: dict[str, Any],
) -> tuple[str | None, dict[str, str] | None]:
    """Exotel Voicebot ``call_data``.

    Forwarded SIP fields land under ``custom_parameters``. Native fallback
    is ``call_id`` (Exotel's CallSid).
    """
    params = payload.get("custom_parameters") or payload.get("customParameters")
    sip_layer = _find_in_aliases(params) if isinstance(params, dict) else None
    call_id = sip_layer or payload.get("call_id")
    headers = (
        _stringify_headers(params) if isinstance(params, dict) and params else None
    )
    return (str(call_id) if call_id else None, headers or None)


# ---------------------------------------------------------------------------
# Jambonz
# ---------------------------------------------------------------------------


# Jambonz extraction is owned by the typed ``JambonzCallContext.from_webhook``
# in ``providers.jambonz``. This thin wrapper exists only so the legacy
# ``attach_sip_from_telephony(payload, provider="jambonz")`` call still
# works for callers who don't want to use the typed context.
def _extract_jambonz(
    payload: dict[str, Any],
) -> tuple[str | None, dict[str, str] | None]:
    from .providers.jambonz import JambonzCallContext

    ctx = JambonzCallContext.from_webhook(payload)
    return (ctx.sip_call_id, ctx.raw_headers or None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


PROVIDER_EXTRACTORS: dict[str, ProviderExtractor] = {
    "twilio": _extract_twilio,
    "telnyx": _extract_telnyx,
    "plivo": _extract_plivo,
    "exotel": _extract_exotel,
    "jambonz": _extract_jambonz,
}

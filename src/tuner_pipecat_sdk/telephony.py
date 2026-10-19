"""Read Tuner's SIP data from a carrier payload and forward it onto the media websocket.

Three steps, the same for every carrier (outbound: build the forward from
``SipInfo(recipient=...)`` and pass it on the dial)::

    info = extract_sip(payload, provider="twilio")          # webhook or websocket start
    fwd = forward_to_websocket(info, ws_url="wss://...")     # in the call webhook
    observer.attach_sip(extract_sip(start, provider="twilio"))  # in the websocket handler

Twilio: return ``fwd.twiml``. Telnyx: ``answer`` the call, then send ``fwd.stream``
as the ``streaming_start`` body. The SIP data arrives on the websocket ``start``
frame (``customParameters`` / ``custom_parameters``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from xml.sax.saxutils import escape

from ._providers import (
    _extract_exotel,
    _extract_jambonz,
    _extract_plivo,
    _extract_telnyx,
    _find_in_aliases,
    _stringify_headers,
)
from .correlation import CORRELATION_HEADER, is_correlation_header, valid_correlation_id

_SIP_HEADER_PREFIX = "SipHeader_"
SIP_CALL_ID_PARAM = "SipCallId"
# The number or SIP URI an outbound call dialed; Tuner links outbound calls on it.
RECIPIENT_PARAM = "Recipient"


@dataclass(frozen=True)
class SipInfo:
    """SIP data read from one carrier payload."""

    sip_call_id: str | None = None
    sip_headers: dict[str, str] = field(default_factory=dict)
    # Carrier-native ID (CallSid, call_control_id); not a SIP Call-ID.
    carrier_call_id: str | None = None
    provider: str | None = None
    # Outbound only: what the call dialed.
    recipient: str | None = None

    @property
    def is_fallback(self) -> bool:
        """True when ``sip_call_id`` is only the carrier ID."""
        return bool(self.sip_call_id) and self.sip_call_id == self.carrier_call_id

    @property
    def correlation_id(self) -> str:
        """The validated ``X-Correlation-Id``, or ``""``."""
        for name, value in self.sip_headers.items():
            if is_correlation_header(name) and value:
                return valid_correlation_id(value)
        return ""


@dataclass(frozen=True)
class Forward:
    """What to send the carrier so the SIP data reaches the media websocket.

    Twilio: ``twiml``. Telnyx: ``stream`` (the ``streaming_start`` body) and
    ``answer`` (the same fields for a one-command ``answer``).
    """

    provider: str
    parameters: dict[str, str] = field(default_factory=dict)
    twiml: str | None = None
    stream: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None


@runtime_checkable
class Provider(Protocol):
    """A carrier: reads ``SipInfo`` from its payloads.

    Add ``forward_to_websocket(info, ws_url) -> Forward`` to support forwarding.
    """

    name: str

    def extract(self, payload: Any) -> SipInfo: ...


class TwilioProvider:
    """Voice webhook form, websocket ``start`` message, or Pipecat ``call_data``."""

    name = "twilio"

    def extract(self, payload: Any) -> SipInfo:
        from .providers.twilio import TwilioCallContext

        if not isinstance(payload, Mapping):
            return SipInfo(provider=self.name)
        start = payload.get("start")
        if isinstance(start, Mapping):
            call_data: Mapping[str, Any] = {
                "call_id": start.get("callSid"),
                "stream_id": start.get("streamSid"),
                "body": start.get("customParameters") or {},
            }
        elif isinstance(payload.get("customParameters"), Mapping):
            call_data = {
                "call_id": payload.get("callSid"),
                "stream_id": payload.get("streamSid"),
                "body": payload["customParameters"],
            }
        elif "body" in payload or "call_id" in payload:
            call_data = payload
        else:  # voice webhook form
            call_data = {"call_id": payload.get("CallSid"), "body": dict(payload)}
        ctx = TwilioCallContext.from_call_data(dict(call_data))
        headers = dict(ctx.raw_headers)
        recipient = headers.pop(RECIPIENT_PARAM, None)
        if not recipient and str(headers.get("Direction", "")).startswith("outbound"):
            recipient = headers.get("To")
        return SipInfo(
            sip_call_id=ctx.sip_call_id,
            sip_headers=headers,
            carrier_call_id=ctx.call_sid or None,
            provider=self.name,
            recipient=recipient or None,
        )

    def forward_to_websocket(self, info: SipInfo, ws_url: str) -> Forward:
        params: dict[str, str] = {}
        if info.sip_call_id and not info.is_fallback:
            params[SIP_CALL_ID_PARAM] = info.sip_call_id
        if info.carrier_call_id:
            params["CallSid"] = info.carrier_call_id
        names = {k.lower() for k in info.sip_headers}
        for name, value in info.sip_headers.items():
            if name.startswith(_SIP_HEADER_PREFIX):
                key = name
            elif name.lower().startswith("x-") and (
                f"{_SIP_HEADER_PREFIX}{name}".lower() not in names
            ):
                key = f"{_SIP_HEADER_PREFIX}{name}"
            else:
                continue
            if is_correlation_header(key):
                value = valid_correlation_id(value)
            if value:
                params[key] = value
        if info.recipient:
            params[RECIPIENT_PARAM] = info.recipient
        tags = "".join(
            f'<Parameter name="{_attr(k)}" value="{_attr(v)}"/>' for k, v in params.items()
        )
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Connect><Stream url="{_attr(ws_url)}">{tags}</Stream></Connect>'
            "</Response>"
        )
        return Forward(provider=self.name, parameters=params, twiml=twiml)


class TelnyxProvider:
    """Call Control webhook, or the media websocket ``start`` frame."""

    name = "telnyx"

    def extract(self, payload: Any) -> SipInfo:
        if not isinstance(payload, Mapping):
            return SipInfo(provider=self.name)
        start = payload.get("start")
        if isinstance(start, Mapping):
            return self._from_start(start)
        if "custom_parameters" in payload:
            return self._from_start(payload)
        sip_call_id, headers = _extract_telnyx(dict(payload))
        data = payload.get("data")
        inner = data.get("payload") if isinstance(data, Mapping) else None
        source: Mapping[str, Any] = inner if isinstance(inner, Mapping) else payload
        recipient = source.get("to") if source.get("direction") == "outgoing" else None
        return SipInfo(
            sip_call_id=sip_call_id,
            sip_headers=headers or {},
            carrier_call_id=_telnyx_call_control_id(payload),
            provider=self.name,
            recipient=str(recipient) if recipient else None,
        )

    def _from_start(self, start: Mapping[str, Any]) -> SipInfo:
        headers = _stringify_headers(start.get("custom_parameters"))
        recipient = headers.pop(RECIPIENT_PARAM, None)
        carrier = start.get("call_control_id")
        carrier_id = str(carrier) if carrier else None
        return SipInfo(
            sip_call_id=_find_in_aliases(headers) or carrier_id,
            sip_headers=headers,
            carrier_call_id=carrier_id,
            provider=self.name,
            recipient=recipient or None,
        )

    def forward_to_websocket(self, info: SipInfo, ws_url: str) -> Forward:
        params: dict[str, str] = {}
        if info.correlation_id:
            params[CORRELATION_HEADER] = info.correlation_id
        if info.sip_call_id and not info.is_fallback:
            params[SIP_CALL_ID_PARAM] = info.sip_call_id
        if info.recipient:
            params[RECIPIENT_PARAM] = info.recipient
        stream: dict[str, Any] = {
            "stream_url": ws_url,
            "stream_track": "inbound_track",
            "stream_codec": "PCMU",
            "stream_bidirectional_mode": "rtp",
            "stream_bidirectional_codec": "PCMU",
        }
        if params:
            stream["custom_parameters"] = [{"name": k, "value": v} for k, v in params.items()]
        return Forward(provider=self.name, parameters=params, stream=stream, answer=dict(stream))


class _ExtractOnly:
    def __init__(self, name: str, extractor: Any) -> None:
        self.name = name
        self._extractor = extractor

    def extract(self, payload: Any) -> SipInfo:
        if not isinstance(payload, Mapping):
            return SipInfo(provider=self.name)
        sip_call_id, headers = self._extractor(dict(payload))
        return SipInfo(sip_call_id=sip_call_id, sip_headers=headers or {}, provider=self.name)


class DailyProvider:
    """Pipecat's ``DialinSettings`` (or its dict), e.g. ``runner_args.body["dialin_settings"]``."""

    name = "daily"

    def extract(self, payload: Any) -> SipInfo:
        def get(key: str) -> Any:
            if isinstance(payload, Mapping):
                return payload.get(key)
            return getattr(payload, key, None)

        call_id = get("call_id")
        headers = get("sip_headers")
        return SipInfo(
            sip_call_id=str(call_id) if call_id else None,
            sip_headers=_stringify_headers(headers) if headers else {},
            provider=self.name,
        )


_PROVIDERS: dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Make ``provider`` available as ``extract_sip(..., provider=provider.name)``."""
    _PROVIDERS[provider.name.lower()] = provider


def get_provider(name: str) -> Provider:
    try:
        return _PROVIDERS[name.lower()]
    except KeyError:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider {name!r}. Available: {available}") from None


for _provider in (
    TwilioProvider(),
    TelnyxProvider(),
    _ExtractOnly("plivo", _extract_plivo),
    _ExtractOnly("exotel", _extract_exotel),
    _ExtractOnly("jambonz", _extract_jambonz),
    DailyProvider(),
):
    register_provider(_provider)


def extract_sip(payload: Any, *, provider: str | Provider) -> SipInfo:
    """Read ``sip_call_id``, SIP headers and the carrier call ID from a raw payload."""
    impl = get_provider(provider) if isinstance(provider, str) else provider
    info = impl.extract(payload)
    if info.provider is None:
        info = dataclasses.replace(info, provider=impl.name)
    return info


def forward_to_websocket(
    info: SipInfo, *, ws_url: str, provider: str | Provider | None = None
) -> Forward:
    """Build what the carrier needs to carry ``info`` onto the media websocket."""
    chosen = provider or info.provider
    if chosen is None:
        raise ValueError("pass provider= or a SipInfo from extract_sip")
    impl = get_provider(chosen) if isinstance(chosen, str) else chosen
    build = getattr(impl, "forward_to_websocket", None)
    if build is None:
        raise ValueError(f"provider {impl.name!r} cannot forward to a websocket")
    result: Forward = build(info, ws_url)
    return result


def _telnyx_call_control_id(payload: Mapping[str, Any]) -> str | None:
    data = payload.get("data")
    inner = data.get("payload") if isinstance(data, Mapping) else None
    source = inner if isinstance(inner, Mapping) else payload
    cid = source.get("call_control_id") or payload.get("call_id")
    return str(cid) if cid else None


def _attr(value: str) -> str:
    return escape(str(value), {'"': "&quot;"})


__all__ = [
    "DailyProvider",
    "Forward",
    "Provider",
    "SipInfo",
    "TelnyxProvider",
    "TwilioProvider",
    "extract_sip",
    "forward_to_websocket",
    "get_provider",
    "register_provider",
]

"""extract_sip / forward_to_websocket / Observer.attach_sip and the deprecated names."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET

import pytest

from tuner_pipecat_sdk import (
    Forward,
    SipInfo,
    _deprecation,
    extract_sip,
    forward_to_websocket,
    register_provider,
)
from tuner_pipecat_sdk.telephony import get_provider

TOKEN = "a1b2c3d4e5f6"


@pytest.fixture(autouse=True)
def _reset_warned():
    _deprecation._warned.clear()
    yield
    _deprecation._warned.clear()


# --- extract_sip -----------------------------------------------------------


def test_twilio_webhook_form():
    info = extract_sip(
        {
            "CallSid": "CA1",
            "SipCallId": "sip@host",
            "SipHeader_X-Correlation-Id": TOKEN,
            "From": "+1",
        },
        provider="twilio",
    )
    assert info.sip_call_id == "sip@host"
    assert info.carrier_call_id == "CA1"
    assert info.provider == "twilio"
    assert info.sip_headers["SipHeader_X-Correlation-Id"] == TOKEN
    assert info.sip_headers["X-Correlation-Id"] == TOKEN
    assert info.correlation_id == TOKEN
    assert not info.is_fallback


def test_twilio_start_frame():
    info = extract_sip(
        {
            "event": "start",
            "start": {
                "callSid": "CA1",
                "streamSid": "MZ1",
                "customParameters": {"SipHeader_X-Correlation-Id": TOKEN, "SipCallId": "s@h"},
            },
        },
        provider="twilio",
    )
    assert info.sip_call_id == "s@h"
    assert info.sip_headers["X-Correlation-Id"] == TOKEN
    bare = extract_sip({"callSid": "CA2", "customParameters": {}}, provider="twilio")
    assert bare.sip_call_id == "CA2" and bare.is_fallback


def test_twilio_pipecat_call_data():
    info = extract_sip({"call_id": "CA1", "body": {"SipCallId": "s"}}, provider="twilio")
    assert info.sip_call_id == "s"
    assert info.carrier_call_id == "CA1"


def test_telnyx_start_custom_parameters_object():
    info = extract_sip(
        {
            "event": "start",
            "start": {
                "call_control_id": "v3:ctrl",
                "custom_parameters": {"X-Correlation-Id": TOKEN, "SipCallId": "sip@host"},
            },
        },
        provider="telnyx",
    )
    assert info.sip_call_id == "sip@host"
    assert info.carrier_call_id == "v3:ctrl"
    assert info.correlation_id == TOKEN


def test_telnyx_start_custom_parameters_list():
    info = extract_sip(
        {
            "call_control_id": "v3:ctrl",
            "custom_parameters": [{"name": "X-Correlation-Id", "value": TOKEN}],
        },
        provider="telnyx",
    )
    assert info.correlation_id == TOKEN
    assert info.sip_call_id == "v3:ctrl" and info.is_fallback


def test_telnyx_webhook():
    info = extract_sip(
        {
            "data": {
                "payload": {
                    "call_control_id": "v3:ctrl",
                    "sip_call_id": "sip@host",
                    "custom_headers": [{"name": "X-Correlation-Id", "value": TOKEN}],
                }
            }
        },
        provider="telnyx",
    )
    assert info.sip_call_id == "sip@host"
    assert info.carrier_call_id == "v3:ctrl"
    assert info.correlation_id == TOKEN


def test_daily_dialin_settings():
    info = extract_sip({"call_id": "d1", "sip_headers": {"X-A": "b"}}, provider="daily")
    assert info.sip_call_id == "d1"
    assert info.sip_headers == {"X-A": "b"}

    class Settings:
        call_id = "d2"
        sip_headers = None

    assert extract_sip(Settings(), provider="daily").sip_call_id == "d2"


@pytest.mark.parametrize("name", ["plivo", "exotel", "jambonz"])
def test_extract_only_providers(name):
    info = extract_sip({"call_id": "x", "call_sid": "x"}, provider=name)
    assert info.provider == name
    assert extract_sip("not a mapping", provider=name).sip_call_id is None


def test_non_mapping_payloads():
    assert extract_sip(None, provider="twilio") == SipInfo(provider="twilio")
    assert extract_sip([], provider="telnyx") == SipInfo(provider="telnyx")


def test_malformed_correlation_id_is_empty():
    info = SipInfo(sip_headers={"X-Correlation-Id": "bad value!"})
    assert info.correlation_id == ""


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        extract_sip({}, provider="nope")


def test_register_custom_provider():
    class Acme:
        name = "acme"

        def extract(self, payload):
            return SipInfo(sip_call_id=payload["id"])

        def forward_to_websocket(self, info, ws_url):
            return Forward(provider="acme", parameters={"id": info.sip_call_id or ""})

    register_provider(Acme())
    info = extract_sip({"id": "1"}, provider="ACME")
    assert info.provider == "acme"
    assert forward_to_websocket(info, ws_url="wss://x").parameters == {"id": "1"}
    assert get_provider("acme").name == "acme"


# --- forward_to_websocket --------------------------------------------------


def _params(twiml: str) -> dict[str, str]:
    root = ET.fromstring(twiml)
    return {p.get("name"): p.get("value") for p in root.iter("Parameter")}


def test_twilio_forward_twiml():
    info = extract_sip(
        {
            "CallSid": "CA1",
            "SipCallId": "sip@host",
            "SipHeader_X-Correlation-Id": TOKEN,
            "From": "+1",
        },
        provider="twilio",
    )
    fwd = forward_to_websocket(info, ws_url="wss://h/ws?a=1&b=2")
    root = ET.fromstring(fwd.twiml or "")
    assert root.find("Connect/Stream").get("url") == "wss://h/ws?a=1&b=2"
    assert _params(fwd.twiml or "") == {
        "SipCallId": "sip@host",
        "CallSid": "CA1",
        "SipHeader_X-Correlation-Id": TOKEN,
    }
    assert fwd.stream is None


def test_twilio_forward_drops_malformed_token_and_fallback_id():
    info = SipInfo(
        sip_call_id="CA1",
        carrier_call_id="CA1",
        sip_headers={"X-Correlation-Id": "bad value!", "X-Other": "o"},
        provider="twilio",
    )
    params = _params(forward_to_websocket(info, ws_url="wss://h").twiml or "")
    assert params == {"CallSid": "CA1", "SipHeader_X-Other": "o"}


def test_twilio_forward_round_trips_through_start_frame():
    form = {"CallSid": "CA1", "SipCallId": "sip@host", "SipHeader_X-Correlation-Id": TOKEN}
    fwd = forward_to_websocket(extract_sip(form, provider="twilio"), ws_url="wss://h")
    start = {"start": {"callSid": "CA1", "customParameters": _params(fwd.twiml or "")}}
    info = extract_sip(start, provider="twilio")
    assert info.sip_call_id == "sip@host"
    assert info.correlation_id == TOKEN


def test_telnyx_forward_stream():
    info = extract_sip(
        {
            "data": {
                "payload": {
                    "call_control_id": "v3:ctrl",
                    "sip_call_id": "sip@host",
                    "custom_headers": [{"name": "X-Correlation-Id", "value": TOKEN}],
                }
            }
        },
        provider="telnyx",
    )
    fwd = forward_to_websocket(info, ws_url="wss://h/ws")
    assert fwd.twiml is None
    assert fwd.stream is not None
    assert fwd.stream["stream_url"] == "wss://h/ws"
    assert fwd.stream["custom_parameters"] == [
        {"name": "X-Correlation-Id", "value": TOKEN},
        {"name": "SipCallId", "value": "sip@host"},
    ]
    # The websocket start frame carries them back as an object.
    start = {
        "start": {
            "call_control_id": "v3:ctrl",
            "custom_parameters": {p["name"]: p["value"] for p in fwd.stream["custom_parameters"]},
        }
    }
    back = extract_sip(start, provider="telnyx")
    assert back.sip_call_id == "sip@host" and back.correlation_id == TOKEN


def test_telnyx_forward_without_sip_data_has_no_custom_parameters():
    fwd = forward_to_websocket(SipInfo(provider="telnyx"), ws_url="wss://h")
    assert fwd.stream is not None and "custom_parameters" not in fwd.stream


def test_forward_needs_a_provider():
    with pytest.raises(ValueError, match="provider"):
        forward_to_websocket(SipInfo(), ws_url="wss://h")
    with pytest.raises(ValueError, match="cannot forward"):
        forward_to_websocket(SipInfo(provider="plivo"), ws_url="wss://h")
    fwd = forward_to_websocket(SipInfo(), ws_url="wss://h", provider="twilio")
    assert fwd.provider == "twilio"


# --- Observer.attach_sip and deprecations ----------------------------------

pipecat = pytest.importorskip("pipecat", reason="pipecat not installed")

from tuner_pipecat_sdk.observer import Observer  # noqa: E402
from tuner_pipecat_sdk.providers.twilio import build_sip_forwarding_twiml  # noqa: E402


def _observer():
    return Observer(
        api_key="k", workspace_id=1, agent_id="a", call_id="c", base_url="https://tuner.test"
    )


def test_attach_sip_from_info():
    obs = _observer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs.attach_sip(
            extract_sip(
                {
                    "start": {
                        "call_control_id": "v3:c",
                        "custom_parameters": {"X-Correlation-Id": TOKEN},
                    }
                },
                provider="telnyx",
            ),
            recipient="+15550001111",
        )
    assert obs._config.sip_call_id == "v3:c"
    assert obs._config.sip_headers == {"X-Correlation-Id": TOKEN}
    assert obs._config.recipient == "+15550001111"
    # A real SIP Call-ID replaces the carrier fallback; a later carrier ID does not.
    obs.attach_sip(SipInfo(sip_call_id="sip@host"))
    obs.attach_sip(SipInfo(sip_call_id="v3:d", carrier_call_id="v3:d"))
    assert obs._config.sip_call_id == "sip@host"


def test_attach_sip_explicit_values_win():
    obs = _observer()
    obs.attach_sip(SipInfo(sip_call_id="a@h"), sip_call_id="b@h", sip_headers={"X-A": "1"})
    assert obs._config.sip_call_id == "b@h"
    assert obs._config.sip_headers == {"X-A": "1"}


def test_attach_sip_drops_malformed_correlation_header():
    obs = _observer()
    obs.attach_sip(sip_headers={"X-Correlation-Id": "bad value!"})
    assert not obs._config.sip_headers


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("attach_sip_info", lambda o: o.attach_sip_info(sip_call_id="x")),
        ("attach_sip_from_dialin", lambda o: o.attach_sip_from_dialin({"call_id": "x"})),
        (
            "attach_sip_from_telephony",
            lambda o: o.attach_sip_from_telephony({"call_id": "x"}, provider="plivo"),
        ),
        (
            "attach_sip_from_context",
            lambda o: o.attach_sip_from_context(SipInfo(sip_call_id="x")),
        ),
    ],
)
def test_deprecated_observer_methods_warn_once(name, call):
    obs = _observer()
    with pytest.warns(DeprecationWarning, match=name):
        call(obs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call(obs)
    assert obs._config.sip_call_id == "x"


def test_deprecated_telephony_does_not_also_warn_for_attach_sip_info():
    obs = _observer()
    with pytest.warns(DeprecationWarning) as record:
        obs.attach_sip_from_telephony({"call_id": "x"}, provider="plivo")
    assert [str(w.message).split(" ")[0] for w in record] == ["attach_sip_from_telephony"]


def test_callable_extractor_warns():
    obs = _observer()
    with pytest.warns(DeprecationWarning) as record:
        obs.attach_sip_from_telephony({}, provider=lambda p: ("x", None))
    assert any("register_provider" in str(w.message) for w in record)


def test_build_sip_forwarding_twiml_warns_once():
    with pytest.warns(DeprecationWarning, match="forward_to_websocket"):
        build_sip_forwarding_twiml({"CallSid": "CA1"}, ws_url="wss://h")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_sip_forwarding_twiml({"CallSid": "CA1"}, ws_url="wss://h")


# --- outbound recipient ----------------------------------------------------


@pytest.mark.parametrize("provider", ["twilio", "telnyx"])
def test_recipient_round_trips_through_the_stream(provider):
    fwd = forward_to_websocket(
        SipInfo(recipient="sip:tok@tuner.test"), ws_url="wss://h", provider=provider
    )
    assert fwd.parameters == {"Recipient": "sip:tok@tuner.test"}
    if provider == "twilio":
        start = {"start": {"callSid": "CA1", "customParameters": _params(fwd.twiml or "")}}
    else:
        assert fwd.stream is not None
        start = {
            "start": {
                "call_control_id": "v3:c",
                "custom_parameters": fwd.stream["custom_parameters"],
            }
        }
    info = extract_sip(start, provider=provider)
    assert info.recipient == "sip:tok@tuner.test"
    assert "Recipient" not in info.sip_headers


def test_recipient_from_outbound_webhooks():
    twilio = extract_sip(
        {"CallSid": "CA1", "Direction": "outbound-api", "To": "+1555"}, provider="twilio"
    )
    assert twilio.recipient == "+1555"
    inbound = extract_sip({"CallSid": "CA1", "Direction": "inbound", "To": "+1"}, provider="twilio")
    assert inbound.recipient is None
    telnyx = extract_sip(
        {"data": {"payload": {"call_control_id": "c", "direction": "outgoing", "to": "+1555"}}},
        provider="telnyx",
    )
    assert telnyx.recipient == "+1555"


def test_attach_sip_reports_recipient_from_info():
    obs = _observer()
    obs.attach_sip(SipInfo(recipient="sip:tok@tuner.test"))
    assert obs._config.recipient == "sip:tok@tuner.test"
    obs.attach_sip(SipInfo(recipient="other"), recipient="explicit")
    assert obs._config.recipient == "explicit"


def test_custom_provider_end_to_end():
    class Acme:
        name = "acme2"

        def extract(self, payload):
            params = payload["start"]["params"]
            return SipInfo(
                sip_call_id=params.get("cid"),
                sip_headers={"X-Correlation-Id": params.get("tok", "")},
            )

        def forward_to_websocket(self, info, ws_url):
            params = {"cid": info.sip_call_id or "", "tok": info.correlation_id}
            return Forward(provider=self.name, parameters=params, stream={"url": ws_url})

    register_provider(Acme())
    webhook_info = SipInfo(
        sip_call_id="sip@host", sip_headers={"X-Correlation-Id": TOKEN}, provider="acme2"
    )
    fwd = forward_to_websocket(webhook_info, ws_url="wss://h")
    obs = _observer()
    obs.attach_sip(extract_sip({"start": {"params": fwd.parameters}}, provider="acme2"))
    assert obs._config.sip_call_id == "sip@host"
    assert obs._config.sip_headers == {"X-Correlation-Id": TOKEN}


def test_extract_accepts_pipecat_call_data_model():
    # Same shape as pipecat.runner.types.CallData (importing it needs fastapi).
    from pydantic import BaseModel, ConfigDict, Field

    class CallData(BaseModel):
        model_config = ConfigDict(populate_by_name=True, extra="allow")
        stream_id: str | None = None
        call_id: str | None = None
        to_number: str | None = Field(default=None, alias="to")
        body: dict = Field(default_factory=dict)

    data = CallData(call_id="CA1", stream_id="MZ1", body={"SipCallId": "abc@host"})
    info = extract_sip(data, provider="twilio")
    assert info.sip_call_id == "abc@host"
    assert info.carrier_call_id == "CA1"


def test_extract_warns_on_unreadable_payload():
    from loguru import logger

    seen: list[str] = []
    sink = logger.add(seen.append, level="WARNING")
    try:
        assert extract_sip(object(), provider="telnyx") == SipInfo(provider="telnyx")
    finally:
        logger.remove(sink)
    assert any("unreadable payload" in m for m in seen)

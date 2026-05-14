"""Tests for the typed Twilio integration helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from tuner_pipecat_sdk.observer import Observer
from tuner_pipecat_sdk.providers.twilio import (
    TwilioCallContext,
    build_sip_forwarding_twiml,
)


def _observer():
    return Observer(
        api_key="k",
        workspace_id=1,
        agent_id="a",
        call_id="c",
        base_url="https://tuner.test",
    )


# ---------------------------------------------------------------------------
# TwilioCallContext.from_call_data
# ---------------------------------------------------------------------------


def test_from_call_data_prefers_sip_call_id_in_body():
    ctx = TwilioCallContext.from_call_data(
        {
            "stream_id": "MZ-x",
            "call_id": "CA-twilio",
            "body": {
                "SipCallId": "trunk-id-abc",
                "Caller": "sip:+15551112222@trunk.example",
                "Called": "sip:+15553334444@trunk.example",
                "CallSid": "CA-twilio",
            },
        }
    )
    assert ctx.sip_call_id == "trunk-id-abc"
    assert ctx.call_sid == "CA-twilio"
    assert ctx.stream_sid == "MZ-x"
    assert ctx.from_number == "sip:+15551112222@trunk.example"
    assert ctx.to_number == "sip:+15553334444@trunk.example"
    assert ctx.raw_headers["SipCallId"] == "trunk-id-abc"


def test_from_call_data_alias_lookup_case_insensitive():
    ctx = TwilioCallContext.from_call_data(
        {"call_id": "CA-x", "body": {"x-sip-call-id": "alt-id"}}
    )
    assert ctx.sip_call_id == "alt-id"


def test_from_call_data_falls_back_to_callsid_when_body_empty():
    """Default TwiML (no <Parameter> tags) → only CallSid is available."""
    ctx = TwilioCallContext.from_call_data(
        {"stream_id": "MZ-x", "call_id": "CA-twilio", "body": {}}
    )
    assert ctx.sip_call_id == "CA-twilio"
    assert ctx.call_sid == "CA-twilio"
    assert ctx.raw_headers == {}
    assert ctx.from_number is None


def test_from_call_data_handles_missing_body_key():
    ctx = TwilioCallContext.from_call_data({"call_id": "CA-only"})
    assert ctx.sip_call_id == "CA-only"
    assert ctx.raw_headers == {}


def test_from_call_data_handles_malformed_payload():
    assert TwilioCallContext.from_call_data(None).call_sid == ""  # type: ignore[arg-type]
    assert TwilioCallContext.from_call_data({}).sip_call_id is None


def test_fallback_minimal_context():
    ctx = TwilioCallContext.fallback("CA-abc")
    assert ctx.call_sid == "CA-abc"
    assert ctx.sip_call_id == "CA-abc"
    assert ctx.raw_headers == {}


# ---------------------------------------------------------------------------
# build_sip_forwarding_twiml
# ---------------------------------------------------------------------------


def _parse(xml: str) -> ET.Element:
    return ET.fromstring(xml)


def test_twiml_forwards_default_fields_only_when_present():
    xml = build_sip_forwarding_twiml(
        {
            "CallSid": "CA-123",
            "Caller": "+15551112222",
            "SipCallId": "trunk-abc",
            "From": "+15551112222",
            # Direction intentionally omitted
        },
        ws_url="wss://example.com/ws",
    )
    root = _parse(xml)
    stream = root.find(".//Stream")
    assert stream is not None
    assert stream.get("url") == "wss://example.com/ws"

    params = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
    assert params["CallSid"] == "CA-123"
    assert params["SipCallId"] == "trunk-abc"
    assert params["Caller"] == "+15551112222"
    assert "Direction" not in params  # not on form → not forwarded


def test_twiml_forwards_sip_header_prefixed_fields_automatically():
    xml = build_sip_forwarding_twiml(
        {
            "CallSid": "CA-x",
            "SipHeader_X-Trace-Id": "trace-1",
            "SipHeader_X-Tenant": "acme",
            "Direction": "inbound",
        },
        ws_url="wss://x/ws",
    )
    params = {
        p.get("name"): p.get("value")
        for p in _parse(xml).findall(".//Parameter")
    }
    assert params["SipHeader_X-Trace-Id"] == "trace-1"
    assert params["SipHeader_X-Tenant"] == "acme"
    assert params["Direction"] == "inbound"


def test_twiml_extra_params_merged():
    xml = build_sip_forwarding_twiml(
        {"CallSid": "CA-x"},
        ws_url="wss://x/ws",
        extra_params={"X-Bot-Variant": "premium"},
    )
    params = {
        p.get("name"): p.get("value")
        for p in _parse(xml).findall(".//Parameter")
    }
    assert params["X-Bot-Variant"] == "premium"


def test_twiml_escapes_special_chars_in_values_and_url():
    """Quotes/ampersands in values and URLs must not break the XML."""
    xml = build_sip_forwarding_twiml(
        {
            "CallSid": "CA-x",
            "SipHeader_X-Note": 'has "quotes" & <brackets>',
        },
        ws_url="wss://x/ws?token=a&b=c",
    )
    root = _parse(xml)  # must parse without error
    stream = root.find(".//Stream")
    assert stream is not None
    assert stream.get("url") == "wss://x/ws?token=a&b=c"
    params = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
    assert params["SipHeader_X-Note"] == 'has "quotes" & <brackets>'


def test_twiml_pause_seconds_configurable():
    xml = build_sip_forwarding_twiml(
        {"CallSid": "CA-x"}, ws_url="wss://x/ws", pause_seconds=120
    )
    pause = _parse(xml).find(".//Pause")
    assert pause is not None
    assert pause.get("length") == "120"


def test_twiml_with_empty_form_still_produces_valid_twiml():
    """No SIP fields forwarded; <Stream> still opens correctly."""
    xml = build_sip_forwarding_twiml({}, ws_url="wss://x/ws")
    root = _parse(xml)
    assert root.tag == "Response"
    assert root.find(".//Stream") is not None
    assert root.findall(".//Parameter") == []


def test_twiml_custom_forwarded_fields():
    xml = build_sip_forwarding_twiml(
        {
            "CallSid": "CA-x",
            "Caller": "should-not-forward",
            "From": "+1",
        },
        ws_url="wss://x/ws",
        forwarded_fields=("CallSid", "From"),
    )
    params = {
        p.get("name"): p.get("value")
        for p in _parse(xml).findall(".//Parameter")
    }
    assert "Caller" not in params
    assert params["CallSid"] == "CA-x"
    assert params["From"] == "+1"


# ---------------------------------------------------------------------------
# Observer.attach_sip_from_context with TwilioCallContext
# ---------------------------------------------------------------------------


def test_observer_attaches_from_twilio_context():
    obs = _observer()
    ctx = TwilioCallContext.from_call_data(
        {
            "stream_id": "MZ-x",
            "call_id": "CA-y",
            "body": {"SipCallId": "abc", "Caller": "+1"},
        }
    )
    obs.attach_sip_from_context(ctx)
    assert obs._config.sip_call_id == "abc"
    assert obs._config.sip_headers is not None
    assert obs._config.sip_headers["SipCallId"] == "abc"
    assert obs._config.sip_headers["Caller"] == "+1"


def test_observer_attaches_from_twilio_fallback_context():
    obs = _observer()
    obs.attach_sip_from_context(TwilioCallContext.fallback("CA-z"))
    assert obs._config.sip_call_id == "CA-z"
    assert obs._config.sip_headers is None

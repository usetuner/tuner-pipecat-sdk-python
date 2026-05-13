"""SIP metadata capture: provider-agnostic flow into the final payload."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from tuner_pipecat_sdk.config import TunerConfig
from tuner_pipecat_sdk.models import CallPayload
from tuner_pipecat_sdk.observer import Observer


def _make_observer(**overrides):
    kwargs = dict(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
    )
    kwargs.update(overrides)
    return Observer(**kwargs)


# ---------------------------------------------------------------------------
# Construction-time and explicit setter
# ---------------------------------------------------------------------------


def test_config_accepts_sip_fields():
    cfg = TunerConfig(
        api_key="k",
        workspace_id=1,
        agent_id="a",
        call_id="c",
        sip_call_id="abc-123",
        sip_headers={"X-Custom": "v"},
    )
    assert cfg.sip_call_id == "abc-123"
    assert cfg.sip_headers == {"X-Custom": "v"}


def test_observer_accepts_sip_at_construction():
    obs = _make_observer(sip_call_id="cid-1", sip_headers={"X-A": "1"})
    assert obs._config.sip_call_id == "cid-1"
    assert obs._config.sip_headers == {"X-A": "1"}


def test_attach_sip_info_late_binding():
    obs = _make_observer()
    obs.attach_sip_info(sip_call_id="late-id", sip_headers={"X-Late": "y"})
    assert obs._config.sip_call_id == "late-id"
    assert obs._config.sip_headers == {"X-Late": "y"}


# ---------------------------------------------------------------------------
# attach_sip_from_dialin (Daily PSTN)
# ---------------------------------------------------------------------------


def test_attach_sip_from_dialin_dict():
    obs = _make_observer()
    obs.attach_sip_from_dialin(
        {"call_id": "daily-uuid", "sip_headers": {"X-Forwarded": "trunk-A"}}
    )
    assert obs._config.sip_call_id == "daily-uuid"
    assert obs._config.sip_headers == {"X-Forwarded": "trunk-A"}


def test_attach_sip_from_dialin_object():
    obs = _make_observer()
    settings = MagicMock(spec=["call_id", "sip_headers"])
    settings.call_id = "obj-uuid"
    settings.sip_headers = {"X-A": "B"}
    obs.attach_sip_from_dialin(settings)
    assert obs._config.sip_call_id == "obj-uuid"
    assert obs._config.sip_headers == {"X-A": "B"}


# ---------------------------------------------------------------------------
# attach_sip_from_telephony — the canonical Twilio/Telnyx/Plivo/Exotel path
# ---------------------------------------------------------------------------


def test_telephony_falls_back_to_native_callsid_when_no_customparams():
    """Twilio with default TwiML: body={} → fall back to CallSid."""
    obs = _make_observer()
    obs.attach_sip_from_telephony(
        {"stream_id": "MZ-x", "call_id": "CA-twilio", "body": {}}
    )
    assert obs._config.sip_call_id == "CA-twilio"
    assert obs._config.sip_headers is None


def test_telephony_prefers_sip_call_id_from_twilio_body():
    """When user's TwiML adds <Parameter name='SipCallId' .../>, SDK uses it."""
    obs = _make_observer()
    obs.attach_sip_from_telephony(
        {
            "stream_id": "MZ-x",
            "call_id": "CA-twilio",
            "body": {
                "SipCallId": "LEiKmt9tC1j9RQdnUzwLKeN67iC",
                "Caller": "sip:+15550001111@trunk.example",
                "CallSid": "CA-twilio",
            },
        }
    )
    assert obs._config.sip_call_id == "LEiKmt9tC1j9RQdnUzwLKeN67iC"
    assert obs._config.sip_headers == {
        "SipCallId": "LEiKmt9tC1j9RQdnUzwLKeN67iC",
        "Caller": "sip:+15550001111@trunk.example",
        "CallSid": "CA-twilio",
    }


def test_telephony_alias_lookup_is_case_insensitive():
    obs = _make_observer()
    obs.attach_sip_from_telephony(
        {"call_id": "native", "body": {"x-sip-call-id": "xyz-789"}}
    )
    assert obs._config.sip_call_id == "xyz-789"


def test_telephony_telnyx_call_control_id_fallback():
    obs = _make_observer()
    obs.attach_sip_from_telephony(
        {"stream_id": "s", "call_control_id": "ctrl-xyz", "customParameters": {}}
    )
    assert obs._config.sip_call_id == "ctrl-xyz"


def test_telephony_explicit_sip_headers_override_auto_derived():
    obs = _make_observer()
    obs.attach_sip_from_telephony(
        {"call_id": "id", "body": {"SipCallId": "real-id", "Caller": "+1"}},
        sip_headers={"X-Override": "yes"},
    )
    assert obs._config.sip_call_id == "real-id"
    assert obs._config.sip_headers == {"X-Override": "yes"}


def test_telephony_unknown_call_data_keys_yields_none():
    obs = _make_observer()
    obs.attach_sip_from_telephony({"stream_id": "s"})
    assert obs._config.sip_call_id is None
    assert obs._config.sip_headers is None


# ---------------------------------------------------------------------------
# End-to-end: SIP fields land in the final payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sip_fields_propagate_into_payload():
    obs = _make_observer()
    obs.attach_context(MagicMock(messages=[]))
    obs.attach_sip_info(sip_call_id="sip-001", sip_headers={"X-Caller": "us-east"})

    obs._acc.on_start(0)
    obs._acc.on_call_end(1_000_000_000)

    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        await obs._flush()
        payload: CallPayload = post_mock.call_args[0][1]
        assert payload.sip_call_id == "sip-001"
        assert payload.sip_headers == {"X-Caller": "us-east"}

        d = payload.to_dict()
        assert d["sip_call_id"] == "sip-001"
        assert d["sip_headers"] == {"X-Caller": "us-east"}


@pytest.mark.asyncio
async def test_payload_omits_sip_fields_when_unset():
    obs = _make_observer()
    obs.attach_context(MagicMock(messages=[]))
    obs._acc.on_start(0)
    obs._acc.on_call_end(1_000_000_000)

    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        await obs._flush()
        d = post_mock.call_args[0][1].to_dict()
        assert "sip_call_id" not in d
        assert "sip_headers" not in d

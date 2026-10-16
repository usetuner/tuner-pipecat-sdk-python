"""Header merge and carrier-ID fallback across repeated SIP attaches."""

from __future__ import annotations

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from tuner_pipecat_sdk.observer import Observer


def _observer(cls=Observer):
    return cls(
        api_key="k", workspace_id=1, agent_id="a", call_id="c", base_url="https://tuner.test"
    )


def _telnyx_webhook(**payload):
    return {"data": {"event_type": "call.initiated", "payload": payload}}


def test_later_attach_merges_headers():
    obs = _observer()
    obs.attach_sip_info(sip_headers={"X-Correlation-Id": "tok", "from": "+1"})
    obs.attach_sip_info(sip_headers={"X-Other": "o"})
    assert obs._config.sip_headers == {"X-Correlation-Id": "tok", "from": "+1", "X-Other": "o"}


def test_first_non_empty_value_wins_case_insensitively():
    obs = _observer()
    obs.attach_sip_info(sip_headers={"X-Correlation-Id": "first", "X-Empty": ""})
    obs.attach_sip_info(sip_headers={"x-correlation-id": "second", "X-Empty": "filled"})
    assert obs._config.sip_headers == {"X-Correlation-Id": "first", "X-Empty": "filled"}


def test_empty_headers_do_not_wipe_existing():
    obs = _observer()
    obs.attach_sip_info(sip_headers={"X-Correlation-Id": "tok"})
    obs.attach_sip_info(sip_headers={})
    assert obs._config.sip_headers == {"X-Correlation-Id": "tok"}


def test_sparse_telnyx_media_start_keeps_webhook_header():
    obs = _observer()
    obs.attach_sip_from_telephony(
        _telnyx_webhook(
            call_control_id="v3:ctrl",
            custom_headers=[{"name": "X-Correlation-Id", "value": "tok"}],
        ),
        provider="telnyx",
    )
    obs.attach_sip_from_telephony(
        {"stream_id": "s", "call_control_id": "v3:ctrl", "customParameters": {"a": "b"}},
        provider="telnyx",
    )
    assert obs._config.sip_headers["X-Correlation-Id"] == "tok"
    assert obs._config.sip_headers["a"] == "b"


def test_duplicate_attach_is_idempotent():
    obs = _observer()
    event = _telnyx_webhook(
        call_control_id="v3:ctrl",
        sip_call_id="real@host",
        custom_headers=[{"name": "X-Correlation-Id", "value": "tok"}],
    )
    obs.attach_sip_from_telephony(event, provider="telnyx")
    first = (obs._config.sip_call_id, dict(obs._config.sip_headers))
    obs.attach_sip_from_telephony(event, provider="telnyx")
    assert (obs._config.sip_call_id, obs._config.sip_headers) == first


def test_concurrent_calls_do_not_share_state():
    a, b = _observer(), _observer()
    a.attach_sip_info(sip_call_id="A", sip_headers={"X-Correlation-Id": "ta"})
    b.attach_sip_info(sip_call_id="B", sip_headers={"X-Correlation-Id": "tb"})
    assert a._config.sip_headers == {"X-Correlation-Id": "ta"}
    assert b._config.sip_headers == {"X-Correlation-Id": "tb"}
    assert (a._config.sip_call_id, b._config.sip_call_id) == ("A", "B")


def test_malformed_telnyx_payloads_do_not_raise():
    obs = _observer()
    for bad in ({}, {"data": None}, {"data": {"payload": {"custom_headers": "x"}}}):
        obs.attach_sip_from_telephony(bad, provider="telnyx")
    assert obs._config.sip_call_id is None


def test_subclass_attach_sip_info_hook_still_called():
    calls = []

    class Hooked(Observer):
        def attach_sip_info(self, sip_call_id=None, sip_headers=None):
            calls.append((sip_call_id, sip_headers))
            super().attach_sip_info(sip_call_id=sip_call_id, sip_headers=sip_headers)

    obs = _observer(Hooked)
    obs.attach_sip_from_telephony(
        _telnyx_webhook(call_control_id="v3:ctrl", sip_call_id="real@host"), provider="telnyx"
    )
    obs.attach_sip_from_telephony(
        {"call_id": "CA-1", "body": {"SipCallId": "x"}}, provider="twilio"
    )
    assert len(calls) == 2

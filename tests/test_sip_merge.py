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


# ---------------------------------------------------------------------------
# Carrier-ID fallback (CallSid / call_control_id are not SIP Call-IDs)
# ---------------------------------------------------------------------------


def test_twilio_real_sip_call_id_replaces_callsid_fallback():
    obs = _observer()
    obs.attach_sip_from_telephony({"call_id": "CA-1", "body": {}}, provider="twilio")
    assert obs._config.sip_call_id == "CA-1"
    obs.attach_sip_from_telephony(
        {"call_id": "CA-1", "body": {"SipCallId": "real@host"}}, provider="twilio"
    )
    assert obs._config.sip_call_id == "real@host"


def test_twilio_callsid_does_not_overwrite_real_sip_call_id():
    obs = _observer()
    obs.attach_sip_from_telephony(
        {"call_id": "CA-1", "body": {"SipCallId": "real@host"}}, provider="twilio"
    )
    obs.attach_sip_from_telephony({"call_id": "CA-1", "body": {}}, provider="twilio")
    assert obs._config.sip_call_id == "real@host"


def test_telnyx_sparse_media_start_keeps_real_sip_call_id():
    obs = _observer()
    obs.attach_sip_from_telephony(
        _telnyx_webhook(call_control_id="v3:ctrl", sip_call_id="real@host"), provider="telnyx"
    )
    obs.attach_sip_from_telephony(
        {"stream_id": "s", "call_control_id": "v3:ctrl"}, provider="telnyx"
    )
    assert obs._config.sip_call_id == "real@host"


def test_telnyx_real_id_replaces_control_id_but_not_another_real_id():
    obs = _observer()
    obs.attach_sip_from_telephony({"call_control_id": "v3:ctrl"}, provider="telnyx")
    assert obs._config.sip_call_id == "v3:ctrl"
    obs.attach_sip_from_telephony(
        _telnyx_webhook(call_control_id="v3:ctrl", sip_call_id="real-1"), provider="telnyx"
    )
    assert obs._config.sip_call_id == "real-1"
    obs.attach_sip_from_telephony(
        _telnyx_webhook(call_control_id="v3:ctrl", sip_call_id="real-2"), provider="telnyx"
    )
    assert obs._config.sip_call_id == "real-1"


def test_twilio_context_callsid_fallback_then_real():
    from tuner_pipecat_sdk.providers.twilio import TwilioCallContext

    obs = _observer()
    obs.attach_sip_from_context(TwilioCallContext.fallback("CA-1"))
    obs.attach_sip_from_context(
        TwilioCallContext.from_call_data({"call_id": "CA-1", "body": {"SipCallId": "real"}})
    )
    obs.attach_sip_from_context(TwilioCallContext.fallback("CA-1"))
    assert obs._config.sip_call_id == "real"


def test_custom_extractor_keeps_replace_behavior():
    obs = _observer()
    obs.attach_sip_from_telephony({}, provider=lambda p: ("one", None))
    obs.attach_sip_from_telephony({}, provider=lambda p: ("two", None))
    assert obs._config.sip_call_id == "two"


def test_explicit_attach_sip_info_still_replaces():
    obs = _observer()
    obs.attach_sip_info(sip_call_id="a")
    obs.attach_sip_info(sip_call_id="b")
    assert obs._config.sip_call_id == "b"


@pytest.mark.asyncio
async def test_header_only_report_serializes_without_sip_call_id():
    from unittest.mock import AsyncMock, MagicMock, patch

    obs = _observer()
    obs.attach_context(MagicMock(messages=[]))
    obs.attach_sip_info(sip_headers={"X-Correlation-Id": "tok"})
    obs._acc.on_start(0)
    obs._acc.on_call_end(1_000_000_000)
    with patch("tuner_pipecat_sdk._base.post_call", new_callable=AsyncMock) as post_mock:
        await obs._flush()
    d = post_mock.call_args[0][1].to_dict()
    assert d["sip_headers"] == {"X-Correlation-Id": "tok"}
    assert "sip_call_id" not in d


@pytest.mark.asyncio
async def test_flush_does_not_log_phone_numbers_or_sip_headers():
    from unittest.mock import AsyncMock, MagicMock, patch

    from loguru import logger

    import tuner_pipecat_sdk.client as client_mod

    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO")
    try:
        obs = _observer()
        obs.attach_context(MagicMock(messages=[]))
        obs.attach_sip_info(
            sip_call_id="sipid-secret", sip_headers={"X-Correlation-Id": "tok-secret"}
        )
        obs._config.recipient = "+15550001111"
        obs._acc.on_start(0)
        obs._acc.on_call_end(1_000_000_000)
        response = MagicMock(status_code=200)
        http = MagicMock()
        http.__aenter__ = AsyncMock(return_value=MagicMock(post=AsyncMock(return_value=response)))
        http.__aexit__ = AsyncMock(return_value=None)
        with patch.object(client_mod.httpx, "AsyncClient", return_value=http):
            await obs._flush()
    finally:
        logger.remove(sink)
    text = "".join(lines)
    assert "[tuner]" in text
    for secret in ("tok-secret", "sipid-secret", "+15550001111"):
        assert secret not in text

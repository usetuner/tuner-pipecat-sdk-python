"""Tests for the typed Jambonz integration helpers."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("pipecat", reason="pipecat not installed")

from tuner_pipecat_sdk.observer import Observer
from tuner_pipecat_sdk.providers.jambonz import (
    JambonzCallContext,
    JambonzPendingStore,
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
# JambonzCallContext.from_webhook — extraction priority
# ---------------------------------------------------------------------------


def test_from_webhook_prefers_x_cid():
    ctx = JambonzCallContext.from_webhook(
        {
            "call_sid": "jb-001",
            "from": "+15550001111",
            "to": "+15550002222",
            "direction": "inbound",
            "sip": {
                "headers": {
                    "X-CID": "livekit-cid-abc",
                    "Call-ID": "sip-tx-id",
                    "SipCallId": "trunk-id",
                },
                "call_id": "sip-call-id",
            },
        }
    )
    assert ctx.sip_call_id == "livekit-cid-abc"
    assert ctx.call_sid == "jb-001"
    assert ctx.from_number == "+15550001111"
    assert ctx.to_number == "+15550002222"
    assert ctx.direction == "inbound"
    assert ctx.raw_headers["from"] == "+15550001111"
    assert ctx.raw_headers["direction"] == "inbound"


def test_from_webhook_falls_back_through_priority_chain():
    """Call-ID → SipCallId → sip.call_id → call_sid."""
    ctx = JambonzCallContext.from_webhook(
        {
            "call_sid": "jb-002",
            "sip": {
                "headers": {"Call-ID": "sip-tx", "SipCallId": "trunk"},
                "call_id": "sip-layer",
            },
        }
    )
    assert ctx.sip_call_id == "sip-tx"

    ctx = JambonzCallContext.from_webhook(
        {
            "call_sid": "jb-003",
            "sip": {"headers": {"SipCallId": "trunk"}, "call_id": "sip-layer"},
        }
    )
    assert ctx.sip_call_id == "trunk"

    ctx = JambonzCallContext.from_webhook(
        {"call_sid": "jb-004", "sip": {"call_id": "sip-layer"}}
    )
    assert ctx.sip_call_id == "sip-layer"

    ctx = JambonzCallContext.from_webhook({"call_sid": "jb-005"})
    assert ctx.sip_call_id == "jb-005"


def test_from_webhook_accepts_list_shaped_headers():
    ctx = JambonzCallContext.from_webhook(
        {
            "call_sid": "jb",
            "sip": {
                "headers": [
                    {"name": "X-CID", "value": "abc"},
                    {"name": "Call-ID", "value": "sip-id"},
                ]
            },
        }
    )
    assert ctx.sip_call_id == "abc"
    assert ctx.raw_headers["Call-ID"] == "sip-id"


def test_from_webhook_handles_empty_payload():
    ctx = JambonzCallContext.from_webhook({})
    assert ctx.call_sid == ""
    assert ctx.sip_call_id is None
    assert ctx.raw_headers == {}


def test_fallback_minimal_context():
    ctx = JambonzCallContext.fallback("xyz-123")
    assert ctx.call_sid == "xyz-123"
    assert ctx.sip_call_id == "xyz-123"
    assert ctx.raw_headers == {}
    assert ctx.from_number is None


# ---------------------------------------------------------------------------
# JambonzPendingStore — park/wait_and_pop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_park_then_wait_and_pop_immediate():
    store = JambonzPendingStore()
    ctx = JambonzCallContext.from_webhook({"call_sid": "id-1", "from": "+1"})
    store.park(ctx)

    out = await store.wait_and_pop("id-1", timeout=0.1)
    assert out is ctx
    # Second pop is None — entry consumed.
    assert await store.wait_and_pop("id-1", timeout=0.05) is None


@pytest.mark.asyncio
async def test_wait_and_pop_blocks_until_park_arrives():
    """WS handler started waiting before webhook fired (defensive case)."""
    store = JambonzPendingStore()
    ctx = JambonzCallContext.fallback("id-2")

    async def delayed_park():
        await asyncio.sleep(0.05)
        store.park(ctx)

    parker = asyncio.create_task(delayed_park())
    out = await store.wait_and_pop("id-2", timeout=1.0)
    await parker

    assert out is ctx


@pytest.mark.asyncio
async def test_wait_and_pop_times_out_when_webhook_never_arrives():
    store = JambonzPendingStore()
    assert await store.wait_and_pop("missing", timeout=0.05) is None


@pytest.mark.asyncio
async def test_wait_and_pop_rejects_empty_call_sid():
    store = JambonzPendingStore()
    assert await store.wait_and_pop("", timeout=0.05) is None


@pytest.mark.asyncio
async def test_park_is_noop_for_empty_call_sid():
    store = JambonzPendingStore()
    store.park(JambonzCallContext.fallback(""))
    # No entry got created — wait should still time out.
    assert await store.wait_and_pop("", timeout=0.05) is None


@pytest.mark.asyncio
async def test_ttl_eviction_removes_unconsumed_entry():
    store = JambonzPendingStore(ttl_seconds=0.05)
    store.park(JambonzCallContext.fallback("id-3"))
    await asyncio.sleep(0.15)
    # Entry should have been evicted by the scheduled callback.
    assert await store.wait_and_pop("id-3", timeout=0.05) is None


# ---------------------------------------------------------------------------
# Observer.attach_sip_from_context
# ---------------------------------------------------------------------------


def test_attach_sip_from_context_wires_call_id_and_headers():
    obs = _observer()
    ctx = JambonzCallContext.from_webhook(
        {
            "call_sid": "jb-006",
            "from": "+15551112222",
            "sip": {"headers": {"X-CID": "cid-xyz"}},
        }
    )
    obs.attach_sip_from_context(ctx)
    assert obs._config.sip_call_id == "cid-xyz"
    assert obs._config.sip_headers is not None
    assert obs._config.sip_headers["X-CID"] == "cid-xyz"
    assert obs._config.sip_headers["from"] == "+15551112222"


def test_attach_sip_from_context_handles_fallback_context():
    """fallback() produces no headers — observer must accept that cleanly."""
    obs = _observer()
    obs.attach_sip_from_context(JambonzCallContext.fallback("jb-007"))
    assert obs._config.sip_call_id == "jb-007"
    assert obs._config.sip_headers is None


def test_attach_sip_from_context_duck_typed():
    """Any object with .sip_call_id and .raw_headers works (forward-compat)."""

    class MyCtx:
        sip_call_id = "custom-id"
        raw_headers = {"X-My-Header": "yes"}

    obs = _observer()
    obs.attach_sip_from_context(MyCtx())
    assert obs._config.sip_call_id == "custom-id"
    assert obs._config.sip_headers == {"X-My-Header": "yes"}

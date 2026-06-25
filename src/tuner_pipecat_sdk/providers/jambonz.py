"""Jambonz integration helpers.

Jambonz is an open-source SIP application server. Inbound calls deliver
SIP info on an HTTP **call-hook** webhook, then Jambonz opens a separate
audio WebSocket whose first frame is a thin JSON metadata blob carrying
only ``call_sid``. Customer code therefore has to bridge the two — park
the rich webhook info, then look it up when the WS arrives.

This module provides the two pieces a customer needs:

* :class:`JambonzCallContext` — typed view of the webhook payload.
* :class:`JambonzPendingStore` — TTL-bounded, await-aware bridge between
  the webhook and the WebSocket.

Typical wiring (in the customer's server)::

    from tuner_pipecat_sdk.providers.jambonz import (
        JambonzCallContext, JambonzPendingStore,
    )

    pending = JambonzPendingStore()

    @app.post("/")
    async def call_hook(request):
        data = await request.json()
        pending.park(JambonzCallContext.from_webhook(data))
        return _jambonz_verbs(...)

    @app.websocket("/ws")
    async def ws(websocket):
        ...  # read first frame to learn call_sid
        ctx = await pending.wait_and_pop(call_sid) or JambonzCallContext.fallback(call_sid)
        await run_bot(transport, sip_context=ctx)

The customer's bot then hands the context to Tuner with one call::

    observer.attach_sip_from_context(ctx)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


def _normalize_headers(raw: Any) -> dict[str, str]:
    """Flatten Jambonz's two header shapes (dict or ``[{name, value}]`` list)."""
    headers: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is None:
                continue
            headers[str(k)] = str(v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name")
                value = item.get("value")
                if name and value is not None:
                    headers[str(name)] = str(value)
    return headers


@dataclass
class JambonzCallContext:
    """Typed snapshot of a Jambonz call's identity + SIP headers.

    Built once from the call-hook webhook (``from_webhook``), then carried
    across the webhook → WebSocket boundary by :class:`JambonzPendingStore`.
    """

    call_sid: str
    sip_call_id: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    direction: str | None = None
    raw_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_webhook(cls, payload: dict[str, Any]) -> JambonzCallContext:
        """Extract a context from a raw Jambonz call-hook payload.

        SIP Call-ID resolution priority:

        1. ``sip.headers[X-CID]`` — canonical cross-system id for chains
           like LiveKit SIP → Jambonz, where Jambonz regenerates SIP's
           own ``Call-ID`` on each hop.
        2. ``sip.headers[Call-ID]`` — SIP-layer transaction id.
        3. ``sip.headers[SipCallId]`` — custom forwarded id.
        4. ``sip.call_id`` — Jambonz's own SIP-layer field.
        5. ``call_sid`` — Jambonz's stable call SID (last resort).
        """
        if not isinstance(payload, dict):
            return cls.fallback("")

        sip = payload.get("sip") if isinstance(payload.get("sip"), dict) else {}
        headers = _normalize_headers(sip.get("headers"))

        for k in ("from", "to", "direction", "callerName"):
            v = payload.get(k)
            if v:
                headers.setdefault(k, str(v))

        call_sid = str(payload.get("call_sid") or payload.get("callSid") or "")

        sip_call_id = (
            headers.get("X-CID")
            or headers.get("x-cid")
            or headers.get("Call-ID")
            or headers.get("call-id")
            or headers.get("SipCallId")
            or sip.get("call_id")
            or (call_sid or None)
        )

        return cls(
            call_sid=call_sid,
            sip_call_id=str(sip_call_id) if sip_call_id else None,
            from_number=payload.get("from"),
            to_number=payload.get("to"),
            direction=payload.get("direction"),
            raw_headers=headers,
        )

    @classmethod
    def fallback(cls, call_sid: str) -> JambonzCallContext:
        """Minimal context when the webhook was never seen.

        Use in the WebSocket handler if :meth:`JambonzPendingStore.wait_and_pop`
        returns ``None`` — rare in practice, since Jambonz only opens the
        WS after the webhook response.
        """
        return cls(
            call_sid=call_sid,
            sip_call_id=call_sid or None,
            raw_headers={},
        )


class JambonzPendingStore:
    """Bridge a Jambonz call-hook payload to the audio WebSocket handler.

    Customers instantiate one store at module scope. The webhook handler
    calls :meth:`park`; the WebSocket handler calls :meth:`wait_and_pop`.

    The store is awaitable: if the WebSocket somehow opens before the
    webhook completes (rare for Jambonz but defensive), ``wait_and_pop``
    blocks until the webhook arrives or the timeout fires. Entries that
    are never consumed are evicted ``ttl_seconds`` after being parked.

    Not designed for cross-process use — keep state in Redis (or similar)
    if you run multiple replicas.
    """

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self._store: dict[str, JambonzCallContext] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._ttl = float(ttl_seconds)

    def park(self, ctx: JambonzCallContext) -> None:
        """Store ``ctx`` and wake any WS handler waiting on its ``call_sid``.

        No-op if ``ctx.call_sid`` is empty. Must be called from an async
        context (a running event loop is required for TTL scheduling).
        """
        if not ctx.call_sid:
            return
        self._store[ctx.call_sid] = ctx
        event = self._events.get(ctx.call_sid)
        if event is not None:
            event.set()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — TTL eviction is best-effort and skipped.
            return
        loop.call_later(self._ttl, self._evict, ctx.call_sid)

    async def wait_and_pop(
        self,
        call_sid: str,
        timeout: float = 5.0,
    ) -> JambonzCallContext | None:
        """Retrieve and remove the context parked under ``call_sid``.

        Awaits up to ``timeout`` seconds for the webhook to arrive when
        the entry is not already present. Returns ``None`` on timeout or
        when ``call_sid`` is empty.
        """
        if not call_sid:
            return None
        if call_sid not in self._store:
            event = self._events.setdefault(call_sid, asyncio.Event())
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except TimeoutError:
                self._events.pop(call_sid, None)
                return None
        ctx = self._store.pop(call_sid, None)
        self._events.pop(call_sid, None)
        return ctx

    def _evict(self, call_sid: str) -> None:
        # Harmless no-op if already consumed by wait_and_pop.
        self._store.pop(call_sid, None)


__all__ = ["JambonzCallContext", "JambonzPendingStore"]

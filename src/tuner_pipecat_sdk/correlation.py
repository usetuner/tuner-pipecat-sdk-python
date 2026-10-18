"""Tuner's simulation correlation header: its name and the token shapes forwarded."""

from __future__ import annotations

import re

CORRELATION_HEADER = "X-Correlation-Id"

# Tuner mints a hex token; anything else is dropped rather than forwarded.
_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


def valid_correlation_id(value: str | None) -> str:
    """Return ``value`` if it is safe to forward, else ``""``. Never log the value."""
    if value and _TOKEN_RE.fullmatch(value):
        return value
    return ""


def is_correlation_header(name: str) -> bool:
    """True for ``X-Correlation-Id`` and Twilio's ``SipHeader_``-prefixed copy."""
    return name.lower().endswith("correlation-id")

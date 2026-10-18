"""valid_correlation_id: same rules as tuner-livekit-sdk's tuner.telephony."""

from __future__ import annotations

import pytest

from tuner_pipecat_sdk import CORRELATION_HEADER, valid_correlation_id


def test_header_name():
    assert CORRELATION_HEADER == "X-Correlation-Id"


@pytest.mark.parametrize("value", ["abc123", "A-b_c.9", "x" * 128])
def test_accepts_token_shapes(value):
    assert valid_correlation_id(value) == value


@pytest.mark.parametrize("value", [None, "", "x" * 129, "a b", "a\nb", "a;b", "a@b"])
def test_drops_everything_else(value):
    assert valid_correlation_id(value) == ""

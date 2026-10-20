"""valid_correlation_id: same rules as tuner-livekit-sdk's tuner.telephony."""

from __future__ import annotations

import pytest

from tuner_pipecat_sdk import CORRELATION_HEADER, valid_correlation_id


def test_header_name():
    assert CORRELATION_HEADER == "X-Correlation-Id"


@pytest.mark.parametrize(
    "value",
    [
        "0123456789abcdef0123456789ABCDEF",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_accepts_token_shapes(value):
    assert valid_correlation_id(value) == value


@pytest.mark.parametrize(
    "value", [None, "", "abc123", "A-b_c.9", "a" * 31, "a" * 33, "A" * 25, "A1" * 13, "a b"]
)
def test_drops_everything_else(value):
    assert valid_correlation_id(value) == ""


@pytest.mark.parametrize(
    "name", ["X-Correlation-Id", "correlation-id", "SipHeader_X-Correlation-Id"]
)
def test_correlation_header_names(name):
    from tuner_pipecat_sdk.correlation import is_correlation_header

    assert is_correlation_header(name)


def test_upstream_correlation_header_is_not_tuners():
    from tuner_pipecat_sdk import SipInfo
    from tuner_pipecat_sdk.correlation import is_correlation_header

    assert not is_correlation_header("X-Upstream-Correlation-Id")
    info = SipInfo(
        sip_headers={"X-Upstream-Correlation-Id": "c" * 32, "X-Correlation-Id": "d" * 32}
    )
    assert info.correlation_id == "d" * 32

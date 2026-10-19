"""Warn once per process when a deprecated name is used."""

from __future__ import annotations

import warnings

_warned: set[str] = set()


def warn_deprecated(old: str, new: str) -> None:
    if old in _warned:
        return
    _warned.add(old)
    warnings.warn(f"{old} is deprecated; use {new}", DeprecationWarning, stacklevel=3)

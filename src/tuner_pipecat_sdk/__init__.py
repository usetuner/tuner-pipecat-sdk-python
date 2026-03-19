"""Public package interface for `tuner_pipecat_sdk`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import TunerConfig

__all__ = ["FlowsObserver", "TunerConfig"]

if TYPE_CHECKING:
    from .observer import FlowsObserver


def __getattr__(name: str) -> Any:
    if name == "FlowsObserver":
        from .observer import FlowsObserver

        return FlowsObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

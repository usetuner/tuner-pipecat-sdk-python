"""Public package interface for `tuner_pipecat_sdk`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import TunerConfig

__all__ = ["TunerObserver", "TunerFlowsObserver", "TunerConfig"]

if TYPE_CHECKING:
    from .flows_observer import TunerFlowsObserver
    from .observer import TunerObserver


def __getattr__(name: str) -> Any:
    if name == "TunerObserver":
        from .observer import TunerObserver
        return TunerObserver
    if name == "TunerFlowsObserver":
        from .flows_observer import TunerFlowsObserver
        return TunerFlowsObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
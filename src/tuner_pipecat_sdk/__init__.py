"""Public package interface for `tuner_pipecat_sdk`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import TunerConfig

__all__ = ["Observer", "FlowsObserver", "TunerConfig"]

if TYPE_CHECKING:
    from .flows_observer import FlowsObserver
    from .observer import Observer


def __getattr__(name: str) -> Any:
    if name == "Observer":
        from .observer import Observer

        return Observer
    if name == "FlowsObserver":
        from .flows_observer import FlowsObserver

        return FlowsObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Required alongside __getattr__ so that dir() and IDE autocompletion
    # surface Observer and FlowsObserver even though they are lazily imported.
    # Return only __all__ — globals() would expose implementation details
    # like TYPE_CHECKING and Any, cluttering IDE autocompletion.
    return list(__all__)

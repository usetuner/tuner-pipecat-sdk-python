"""Tuner Observer — for plain pipecat pipelines (no pipecat-flows)."""

from __future__ import annotations

from pipecat.processors.aggregators.llm_context import LLMContext

from ._base import _BaseObserver


class Observer(_BaseObserver):
    """
    Drop-in observer for plain pipecat pipelines.

    Pipeline position (after TTS):
        transport.input() → stt → user_agg → llm → tts → Observer → transport.output()

    Usage::

        observer = Observer(api_key=..., workspace_id=..., agent_id=..., call_id=...)
        observer.attach_context(context)          # LLMContext instance
        observer.attach_turn_tracking_observer(turn_tracker)  # optional
    """

    def attach_context(self, context: LLMContext) -> None:
        """Read the transcript from an LLMContext at call end."""
        self._context_provider = lambda: context.messages

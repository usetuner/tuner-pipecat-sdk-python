"""TunerObserver — for plain pipecat pipelines (no pipecat-flows)."""

from __future__ import annotations

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from ._base import _BaseTunerObserver


class TunerObserver(_BaseTunerObserver):
    """
    Drop-in observer for plain pipecat pipelines.

    Pipeline position (after TTS):
        transport.input() → stt → user_agg → llm → tts → TunerObserver → transport.output()

    Usage::

        observer = TunerObserver(api_key=..., workspace_id=..., agent_id=..., call_id=...)
        observer.attach_context(context)          # OpenAILLMContext instance
        observer.attach_turn_tracking_observer(turn_tracker)  # optional
    """

    def attach_context(self, context: OpenAILLMContext) -> None:
        """Read the transcript from an OpenAILLMContext at call end."""
        self._context_provider = lambda: context.messages

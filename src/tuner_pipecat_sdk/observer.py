"""Tuner Observer — for plain pipecat pipelines."""

from __future__ import annotations

from pipecat.processors.aggregators.llm_context import LLMContext

from ._base import _BaseObserver


class Observer(_BaseObserver):
    """
    Drop-in observer for plain pipecat pipelines.

    This is a pipeline-level observer — pass it in ``PipelineTask(observers=[...])``,
    NOT as a processor in the ``Pipeline([...])`` list. It sees every frame at every
    processor boundary, so it captures frames an intermediate processor consumes (e.g.
    ``TranscriptionFrame``, swallowed by the user aggregator) and stays out of the audio path.

    Usage::

        observer = Observer(api_key=..., workspace_id=..., agent_id=..., call_id=...)
        observer.attach_context(context)                      # LLMContext instance
        observer.attach_turn_tracking_observer(turn_tracker)  # optional
        task = PipelineTask(
            pipeline,
            observers=[observer, observer.latency_observer, turn_tracker],
        )
    """

    def attach_context(self, context: LLMContext) -> None:
        """Read the transcript from an LLMContext at call end."""
        self._context_provider = lambda: context.messages

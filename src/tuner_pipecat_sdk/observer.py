"""Tuner Observer — for plain pipecat pipelines."""

from __future__ import annotations

from typing import Any

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
        observer.attach_turn_tracking_observer(turn_tracker)  # optional
        task = PipelineTask(
            pipeline,
            observers=[observer, observer.latency_observer, turn_tracker],
        )
    """

    def attach_context(self, context: Any) -> None:
        """Deprecated no-op. The transcript is now built live from the frame stream
        (acc.live_turns) and no longer reads the LLM context. Kept so existing integrations
        that call ``attach_context(context)`` keep working without change."""
        return None

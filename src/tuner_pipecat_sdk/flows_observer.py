"""TunerFlowsObserver — for pipecat-flows pipelines."""

from __future__ import annotations

from pipecat_flows import FlowManager

from ._base import _BaseTunerObserver


class TunerFlowsObserver(_BaseTunerObserver):
    """
    Drop-in observer for pipecat-flows pipelines.

    Pipeline position (after TTS):
        transport.input() → stt → user_agg → llm → tts → TunerFlowsObserver → transport.output()

    Usage::

        observer = TunerFlowsObserver(api_key=..., workspace_id=..., agent_id=..., call_id=...)
        observer.attach_flow_manager(flow_manager)    # must be called after FlowManager is ready
        observer.attach_turn_tracking_observer(turn_tracker)  # optional
    """

    def attach_flow_manager(self, flow_manager: FlowManager) -> None:
        """Read the transcript from the active FlowManager context at call end."""
        self._context_provider = lambda: flow_manager.get_current_context()

"""pipecat_flows_tuner — Tuner SDK for pipecat-flows.

Tracks node transitions, flow state, latency metrics, and the full
conversation transcript (via flow_manager.get_current_context()), then
ships the data to the Tuner API.

Usage::

    from pipecat_flows_tuner import FlowsObserver

    observer = FlowsObserver(
        api_key="...",
        workspace_id=42,
        agent_id="my-agent",
        call_id=str(uuid.uuid4()),
        debug=True,
    )

    # Wire up before running the pipeline
    observer.attach_flow_manager(flow_manager)

    pipeline = Pipeline([
        transport.input(), stt, user_agg, llm, tts,
        observer,
        transport.output(), assistant_agg,
    ])
"""

from .observer import FlowsObserver

__all__ = ["FlowsObserver"]

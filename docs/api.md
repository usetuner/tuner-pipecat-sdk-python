# API Reference

## `FlowsObserver`

`FlowsObserver` is a `FrameProcessor` that captures runtime signals and emits one payload per call.

Constructor:

```python
FlowsObserver(
    api_key: str,
    workspace_id: int,
    agent_id: str,
    call_id: str,
    call_type: str = "web_call",
    base_url: str = "http://localhost:8000",
    recording_url: str = "pipecat://no-recording",
    debug: bool = False,
    asr_model: str = "",
    llm_model: str = "",
    tts_model: str = "",
)
```

Methods:

- `attach_flow_manager(flow_manager) -> None`
- `attach_turn_tracking_observer(turn_tracker) -> None`
- `latency_observer -> UserBotLatencyObserver`

## `TunerConfig`

Validated configuration model used by the observer and HTTP client.

Validation rules:

- `api_key`, `agent_id`, `call_id` must be non-empty.
- `workspace_id` must be a positive integer.

## Models

Public payload and transcript models are available via:

- `tuner_pipecat_sdk.models.CallPayload`
- `tuner_pipecat_sdk.models.TranscriptSegment`
- `tuner_pipecat_sdk.models.ToolInfo`
- `tuner_pipecat_sdk.models.NodeInfo`

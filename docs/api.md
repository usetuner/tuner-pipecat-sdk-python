# API Reference

## `Observer`

`Observer` is a pipeline-level observer that captures runtime signals and emits one payload per call. Register it on `PipelineTask(observers=[...])`, not inside `Pipeline([...])`.

Constructor:

```python
Observer(
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
    recipient: str | None = None,
    sip_call_id: str | None = None,
    sip_headers: dict[str, str] | None = None,
    cost_calculator: Callable[[CallUsage], float] | None = None,
    disconnection_reason_resolver: Callable[[], str | None] | None = None,
    agent_version: int | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recipient` | `str \| None` | `None` | Phone number or SIP URL of the callee party (e.g. `+15551234567` or `sip:alice@example.com`). **Not collected automatically** — pass this when the callee identity is known to your application. |
| `sip_call_id` | `str \| None` | `None` | SIP provider call identifier. |
| `sip_headers` | `dict[str, str] \| None` | `None` | SIP INVITE headers as a flat dict. |
| `cost_calculator` | `Callable[[CallUsage], float] \| None` | `None` | Returns call cost in USD cents from usage data. |
| `disconnection_reason_resolver` | `Callable[[], str \| None] \| None` | `None` | Called at flush time to record why the call ended. |
| `agent_version` | `int \| None` | `None` | Deployment version number — overrides `APP_VERSION` env var. |

Methods:

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

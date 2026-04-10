# tuner-pipecat-sdk

`tuner-pipecat-sdk` is a lightweight observer SDK for [`pipecat`](https://github.com/pipecat-ai/pipecat) and [`pipecat-flows`](https://github.com/pipecat-ai/pipecat-flows).
It captures flow transitions, latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.


## Requirements

- Python **3.10–3.13**. 
- **Do not use Python 3.14** for installs yet: Pipecat pulls **`onnxruntime~=1.23.2`** and **`numba`** without 3.14 wheels → errors like *No matching distribution found for onnxruntime*.
- This SDK depends on **`pipecat-ai>=0.0.105`**.

## Installation

```bash
pip install tuner-pipecat-sdk
```


## Quick Start Example
---

## Plain Pipecat — `Observer`

Use `Observer` when your pipeline manages context directly via `OpenAILLMContext`.

```python
import uuid
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from tuner_pipecat_sdk import Observer

context = OpenAILLMContext(messages=[])

observer = Observer(
    api_key="YOUR_TUNER_API_KEY",
    workspace_id=42,
    agent_id="my-agent",
    call_id=str(uuid.uuid4()),
    base_url="https://api.usetuner.ai",
    asr_model="deepgram/nova-3",
    llm_model="gpt-4o-mini",
    tts_model="cartesia/sonic",
)

# Required: attach the LLM context before running the pipeline
observer.attach_context(context)
observer.attach_turn_tracking_observer(turn_tracker)
```

---

## Pipecat Flows — `FlowsObserver`

Use `FlowsObserver` when your pipeline is managed by `pipecat-flows` and a `FlowManager`.

```python
import uuid
from tuner_pipecat_sdk import FlowsObserver

observer = FlowsObserver(
    api_key="YOUR_TUNER_API_KEY",
    workspace_id=42,
    agent_id="my-agent",
    call_id=str(uuid.uuid4()),
    base_url="https://api.usetuner.ai",
    asr_model="deepgram/nova-3",
    llm_model="gpt-4o-mini",
    tts_model="cartesia/sonic",
)

# Required: attach the flow manager before running the pipeline
observer.attach_flow_manager(flow_manager)
observer.attach_turn_tracking_observer(turn_tracker)
```

---

## Pipeline Setup

Place the observer after TTS in your pipeline (same for both observer types):

```python
Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    observer,
    transport.output(),
    context_aggregator.assistant(),
])
```

Enable metrics on the pipeline task so latency and usage fields are populated:

```python
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.pipeline_params import PipelineParams
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver

turn_tracker = TurnTrackingObserver()

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        observers=[observer.latency_observer, turn_tracker],
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
)
```

Without these flags the observer will log warnings and LLM/TTS metric fields will be absent from the payload.
For more example check https://github.com/usetuner/tuner-pipecat-sdk-python/tree/main/examples

## Observer Parameters
Both `Observer` and `FlowsObserver` accept the same constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | — | Tuner API key |
| `workspace_id` | `int` | — | Tuner workspace ID |
| `agent_id` | `str` | — | Agent identifier |
| `call_id` | `str` | — | Unique call ID (use `uuid4()`) |
| `base_url` | `str` | `http://localhost:8000` | Tuner API base URL |
| `call_type` | `str` | `"web_call"` | Call type label |
| `recording_url` | `str` | `"pipecat://no-recording"` | Recording URL if available |
| `asr_model` | `str` | `""` | ASR model name (e.g. `deepgram/nova-3`) |
| `llm_model` | `str` | `""` | LLM model name (e.g. `gpt-4o-mini`) |
| `tts_model` | `str` | `""` | TTS model name (e.g. `cartesia/sonic`) |
| `debug` | `bool` | `False` | Log full transcript at flush |

## Disconnection Reason

Pass a `disconnection_reason_resolver` callable to the observer to record why a call ended.
The resolver is called at flush time and should return a string or `None`.

```python
from tuner_pipecat_sdk.models import DisconnectReason

observer = Observer(
    ...
    disconnection_reason_resolver=lambda: DisconnectReason.USER_HANGUP,
)
```

Use the built-in constants from `DisconnectReason` or pass a custom string:

| Constant | Value |
|----------|-------|
| `DisconnectReason.USER_HANGUP` | `"user_hangup"` |
| `DisconnectReason.AGENT_HANGUP` | `"agent_hangup"` |
| `DisconnectReason.ERROR` | `"error"` |
| `DisconnectReason.TIMEOUT` | `"timeout"` |
| `DisconnectReason.UNKNOWN` | `"unknown"` |

For dynamic resolution (e.g. when the reason is only known at call end):

```python
_reason = None

def resolve_reason() -> str | None:
    return _reason

observer = Observer(..., disconnection_reason_resolver=resolve_reason)

# Later, when you know the reason:
_reason = DisconnectReason.AGENT_HANGUP
```

## Public API

- `tuner_pipecat_sdk.Observer` — for plain pipecat pipelines
- `tuner_pipecat_sdk.FlowsObserver` — for pipecat-flows pipelines
- `tuner_pipecat_sdk.TunerConfig`

Payload and transcript schemas are available under `tuner_pipecat_sdk.models`.


## To build the project
folow the steps in setup_guide.md
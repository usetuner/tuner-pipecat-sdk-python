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

To use pipecat-flows, install with the `flows` extra:

```bash
pip install tuner-pipecat-sdk[flows]
```

## Quick Start Example
---

## Plain Pipecat — `Observer`

Use `Observer` when your pipeline manages context directly via `OpenAILLMContext`.

```python
import uuid
from pipecat.processors.aggregators.llm_context import LLMContext
from tuner_pipecat_sdk import Observer

context = LLMContext()

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
| `sip_call_id` | `str \| None` | `None` | SIP provider call identifier (see SIP section below) |
| `sip_headers` | `dict[str, str] \| None` | `None` | SIP INVITE headers as a flat dict |
| `debug` | `bool` | `False` | Log full transcript at flush |

## SIP / Telephony Calls

The observer can capture the SIP call identifier and SIP headers and forward
them to Tuner. This works with any Pipecat SIP/telephony provider — Daily
PSTN, Twilio, Telnyx, Plivo, Exotel, or a custom SIP trunk.

There are three equivalent ways to supply the info:

**1. At construction** (when you already know it):

```python
observer = Observer(
    api_key="...", workspace_id=42, agent_id="...", call_id="...",
    sip_call_id="CA-xxxxxxxx",
    sip_headers={"X-Caller-Region": "us-east", "From": "sip:+1...@trunk"},
)
```

**2. Late-bound via `attach_sip_info()`** (when the ID arrives in the
WebSocket "start" message — Twilio/Telnyx/Plivo/Exotel):

```python
from pipecat.runner.utils import parse_telephony_websocket

transport_type, call_data = await parse_telephony_websocket(websocket)
observer.attach_sip_info(
    sip_call_id=call_data.get("call_id") or call_data.get("call_control_id"),
)
```

Or use the provider-shape-aware helper:

```python
observer.attach_sip_from_telephony(call_data)
# Optionally pass headers obtained out-of-band:
observer.attach_sip_from_telephony(call_data, sip_headers={"X-Trunk": "A"})
```

**3. From Daily `DialinSettings`** (PSTN/SIP dial-in):

```python
from pipecat.runner.types import DialinSettings, DailyDialinRequest

req = DailyDialinRequest.model_validate(runner_args.body)
observer.attach_sip_from_dialin(req.dialin_settings)
# or pass the dict directly:
observer.attach_sip_from_dialin(runner_args.body["dialin_settings"])
```

Provider → call-id field mapping:

| Provider | Call ID source | SIP headers |
|----------|----------------|-------------|
| Daily PSTN/SIP | `DialinSettings.call_id` | `DialinSettings.sip_headers` |
| Twilio Media Streams | `call_data["call_id"]` (Twilio `callSid`) | not in WS protocol — pass out-of-band |
| Telnyx | `call_data["call_control_id"]` | not in WS protocol — pass out-of-band |
| Plivo | `call_data["call_id"]` | not in WS protocol — pass out-of-band |
| Exotel | `call_data["call_id"]` (`call_sid`) | not in WS protocol — pass out-of-band |
| Custom SIP trunk | whatever your stack exposes | whatever your stack exposes |

Both fields are optional. When omitted they are excluded from the payload so
non-SIP web calls remain unaffected.

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
# pipecat-flows-tuner

`pipecat-flows-tuner` is a lightweight observer SDK for [`pipecat-flows`](https://github.com/pipecat-ai/pipecat-flows).
It captures flow transitions, latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.



## Requirements

- Python **3.10–3.13**. **`python3.12` is verified** for `pip install -e .` (repo includes `.python-version` for `uv`).
- **Do not use Python 3.14** for installs yet: Pipecat pulls **`onnxruntime~=1.23.2`** and **`numba`** without 3.14 wheels → errors like *No matching distribution found for onnxruntime*.
- This SDK depends on **`pipecat-ai>=0.0.105`**.

## Installation

```bash
[placeholder for pypi command] feat/update-call-details-page
```


**Troubleshooting**

| Issue | What to do |
|-------|------------|
| `No matching distribution found for onnxruntime~=1.23.2` | **Python 3.14**: Pipecat pins `onnxruntime` versions that have no 3.14 wheels. Switch to **3.12 or 3.13** (new venv). |
| `Failed to build numba` / *Cannot install on Python version 3.14* | Same: use **Python 3.12 or 3.13**. |
| `No matching distribution found for pipecat-flows-tuner` (example + **pip**) | Install the SDK from **repo root** first (`pip install -e .`), then install the example. |

## Quick Start Example

```python
import uuid
from pipecat_flows_tuner import FlowsObserver

observer = FlowsObserver(
    api_key="YOUR_TUNER_API_KEY",
    workspace_id=42,
    agent_id="my-agent",
    call_id=str(uuid.uuid4()),
    base_url="https://app.usetuner.ai",
    asr_model="deepgram/nova-3",
    llm_model="gpt-4o-mini",
    tts_model="cartesia/sonic",
)

# Required: attach the flow manager before running the pipeline
observer.attach_flow_manager(flow_manager)
observer.attach_turn_tracking_observer(turn_tracker)
```


Place the observer after TTS in your pipeline:

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

## FlowsObserver Parameters

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

## Public API

- `pipecat_flows_tuner.FlowsObserver`
- `pipecat_flows_tuner.TunerConfig`

Payload and transcript schemas are available under `pipecat_flows_tuner.models`.



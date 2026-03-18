# pipecat-flows-tuner

`pipecat-flows-tuner` is a lightweight observer SDK for [`pipecat-flows`](https://github.com/pipecat-ai/pipecat-flows).
It captures flow transitions, latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.

## Why this SDK

- Small integration surface: one `FlowsObserver` in the pipeline.
- Flow-aware analytics: node transitions and state snapshots from `FlowManager`.
- Exact metrics from Pipecat: TTFB, processing latency, LLM token counts, TTS character counts via `MetricsFrame`.
- Typed payload models for downstream tooling.

## Requirements

- Python **3.10–3.13**. **`python3.12` is verified** for `pip install -e .` (repo includes `.python-version` for `uv`).
- **Do not use Python 3.14** for installs yet: Pipecat pulls **`onnxruntime~=1.23.2`** and **`numba`** without 3.14 wheels → errors like *No matching distribution found for onnxruntime*.
- This SDK depends on **`pipecat-ai>=0.0.105`**.

## Installation

This package is **not on PyPI**. Install from a clone of this repository.

```bash
git clone https://github.com/usetuner/tuner-pipecat-sdk-python.git
cd tuner-pipecat-sdk-python
```

Create a venv with **Python 3.12 or 3.13** (not the system 3.14 if that is your default):

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # macOS/Linux
# or: uv venv --python 3.12 && source .venv/bin/activate
```

**Into that venv:**

```bash
pip install -e .
# optional dev/test extras
pip install -e ".[dev]"
```

**Examples** under `examples/` depend on this SDK via a **local path** (`tool.uv.sources`). Use **`uv sync`** inside each example directory, or install the SDK from the repo root with `pip install -e .` first, then `pip install -e .` in the example (plain `pip` does not read `uv.sources`).

**Troubleshooting**

| Issue | What to do |
|-------|------------|
| `No matching distribution found for onnxruntime~=1.23.2` | **Python 3.14**: Pipecat pins `onnxruntime` versions that have no 3.14 wheels. Switch to **3.12 or 3.13** (new venv). |
| `Failed to build numba` / *Cannot install on Python version 3.14* | Same: use **Python 3.12 or 3.13**. |
| `No matching distribution found for pipecat-flows-tuner` (example + **pip**) | Install the SDK from **repo root** first (`pip install -e .`), then install the example. |

## Quick Start

```python
import uuid
from pipecat_flows_tuner import FlowsObserver

observer = FlowsObserver(
    api_key="YOUR_TUNER_API_KEY",
    workspace_id=42,
    agent_id="my-agent",
    call_id=str(uuid.uuid4()),
    base_url="https://app.tuner.ai",
    asr_model="deepgram/nova-3",
    llm_model="gpt-4o-mini",
    tts_model="cartesia/sonic",
)

# Required: attach the flow manager before running the pipeline
observer.attach_flow_manager(flow_manager)
observer.attach_turn_tracking_observer(turn_tracker)
```

### OpenTelemetry (optional)

- **Not required** for the Tuner API payload. With `pip install 'pipecat-flows-tuner[otel]'`, the observer can emit optional spans around `FunctionCall*` frames if you configure an exporter.
- Tool transcript timings use **`FlowManager._create_transition_func`** (`params.tool_call_id`), not FCIP frame order.

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

## Examples

See [`examples/`](examples/) for working bots built with this SDK:

| Example | Use case |
|---------|----------|
| [`pizza_order/`](examples/pizza_order/) | Pizza ordering with toppings, delivery/pickup branch |
| [`appointment_booking/`](examples/appointment_booking/) | Medical clinic receptionist, 7-node linear flow |
| [`customer_support/`](examples/customer_support/) | Multi-branch support agent with escalation |

Each example is self-contained. See **Installation** above for how the SDK is resolved. To run one:

```bash
cd examples/<example_name>
uv sync
cp .env.example .env   # if present; fill in API keys
uv run <script>.py
```

Then open http://localhost:7860 where applicable.


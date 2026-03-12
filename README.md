# pipecat-flows-tuner

`pipecat-flows-tuner` is a lightweight observer SDK for [`pipecat-flows`](https://github.com/pipecat-ai/pipecat-flows).
It captures flow transitions, latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.

## Why this SDK

- Small integration surface: one `FlowsObserver` in the pipeline.
- Flow-aware analytics: node transitions and state snapshots from `FlowManager`.
- Exact metrics from Pipecat: TTFB, processing latency, LLM token counts, TTS character counts via `MetricsFrame`.
- Typed payload models for downstream tooling.

## Installation

```bash
pip install pipecat-flows-tuner
```

For local development:

```bash
pip install -e ".[dev]"
```

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

task = PipelineTask(
    pipeline,
    params=PipelineParams(
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

Each example is self-contained. To run one:

```bash
cd examples/<example_name>
uv sync
cp .env.example .env   # fill in your API keys
uv run <example_name>.py
```

Then open http://localhost:7860 in your browser.


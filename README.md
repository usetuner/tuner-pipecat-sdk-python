# pipecat-flows-tuner

`pipecat-flows-tuner` is a lightweight observer SDK for `pipecat-flows`.
It captures flow transitions, latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.

## Why this SDK

- Small integration surface: one `FlowsObserver` in the pipeline.
- Flow-aware analytics: node transitions and state snapshots from `FlowManager`.
- Robust timing metrics: VAD, LLM-start, TTS-start, bot-start, bot-stop.
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
    base_url="https://your-tuner-api.example.com",
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

## Public API

- `pipecat_flows_tuner.FlowsObserver`
- `pipecat_flows_tuner.TunerConfig`

Payload and transcript schemas are available under `pipecat_flows_tuner.models`.

## Development

Run quality checks locally:

```bash
ruff check .
mypy src
pytest
```

## Documentation

- `docs/getting-started.md`
- `docs/integration.md`
- `docs/api.md`

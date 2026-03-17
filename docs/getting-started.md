# Getting Started

## Requirements

- Python 3.10+
- A `pipecat` bot that uses `pipecat-flows`
- Tuner API credentials (`api_key`, `workspace_id`, `agent_id`)

## Pipecat app working directory

Run the pipeline from your **Pipecat application project root** (the app that creates the pipeline, adds `FlowsObserver`, and calls `PipelineTask`). That is the working directory (`pwd`) for the process that runs this SDK.

Example: if your app is at `/path/to/pipecat-app`, start your server or script from there:

```bash
cd /path/to/pipecat-app
python -m your_app.main
# or
uv run your_app
```

Relative imports, config paths, and pipeline setup all resolve from this directory. The observer runs inside the pipeline process; it does not need a separate working directory.

## Install

```bash
pip install pipecat-flows-tuner
```

## Minimal Setup

```python
import uuid
from pipecat_flows_tuner import FlowsObserver

observer = FlowsObserver(
    api_key="YOUR_TUNER_API_KEY",
    workspace_id=42,
    agent_id="support-agent",
    call_id=str(uuid.uuid4()),
    base_url="https://your-tuner-api.example.com",
    asr_model="deepgram/nova-3",
    llm_model="gpt-4o-mini",
    tts_model="cartesia/sonic",
)

observer.attach_flow_manager(flow_manager)
observer.attach_context_aggregators(context_aggregator)
```

Set `asr_model`, `llm_model`, and `tts_model` to populate
`general_meta_data_raw.ai_models` in the payload.

## Pipeline Placement

Put `FlowsObserver` after TTS and before transport output:

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

## Metrics from Pipecat (required for full payload)

The SDK reads **usage and latency from Pipecat only** (no fallback). Ensure the task is created with metrics enabled:

```python
from pipecat.pipeline import PipelineTask
from pipecat.pipeline.pipeline_params import PipelineParams

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        observers=[observer.latency_observer],
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
)
```

Without these, `llm_token`, `tts_character_count`, and per-turn `ttfb_ms` / `llm_ms` / `tts_ms` will be absent (null/zero) in the payload.

## What Happens at Call End

When `EndFrame` or `CancelFrame` is observed:

1. The observer reads transcript context from the attached flow manager.
2. It builds a typed `CallPayload`.
3. It sends the payload to `POST /api/v1/public/call`.

Network failures are logged and swallowed so the media pipeline is not blocked.

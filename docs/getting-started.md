# Getting Started

## Requirements

- Python 3.10+
- A `pipecat` bot that uses `pipecat-flows`
- Tuner API credentials (`api_key`, `workspace_id`, `agent_id`)

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
)

observer.attach_flow_manager(flow_manager)
```

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

## What Happens at Call End

When `EndFrame` or `CancelFrame` is observed:

1. The observer reads transcript context from the attached flow manager.
2. It builds a typed `CallPayload`.
3. It sends the payload to `POST /api/v1/public/call`.

Network failures are logged and swallowed so the media pipeline is not blocked.

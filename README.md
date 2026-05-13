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

The observer can capture the SIP Call-ID and SIP headers and forward them to
Tuner. The SDK call is **the same one-liner across every provider** — what
differs is what your provider gives you (and what you have to configure on
their side). Pick your provider below.

The SDK call:

```python
# Bot has access to call_data from parse_telephony_websocket:
from pipecat.runner.utils import parse_telephony_websocket
_, call_data = await parse_telephony_websocket(websocket)
observer.attach_sip_from_telephony(call_data)

# Daily PSTN/SIP dial-in:
observer.attach_sip_from_dialin(runner_args.body["dialin_settings"])

# Anything else (you already have the values):
observer.attach_sip_info(sip_call_id="...", sip_headers={...})
```

### What the SDK does automatically

`attach_sip_from_telephony(call_data)` will:

1. Look inside any nested customParameters dict on `call_data` — `body`
   (Twilio), `customParameters` (Telnyx), `custom_parameters` (Exotel) —
   for a SIP-layer Call-ID under any case-insensitive alias: `SipCallId`,
   `sip_call_id`, `sip-call-id`, `X-Sip-Call-Id`.
2. Fall back to the provider's native call id (`call_id` for
   Twilio/Plivo/Exotel, `call_control_id` for Telnyx) when no SIP-layer key
   was forwarded.
3. Use those customParameters as `sip_headers` when you don't pass
   `sip_headers` explicitly.

For the SIP-layer Call-ID to actually appear, **your trunk/webhook must
forward it**. The per-provider configuration is below.

### Twilio Media Streams (most common case)

Twilio's WebSocket protocol drops the SIP-layer fields (`SipCallId`,
`Caller`, every `SipHeader_*`) by default — they only exist on the HTTP
voice webhook. To bridge them, your webhook must return TwiML with
`<Parameter>` tags:

```xml
<Response>
  <Connect>
    <Stream url="wss://your-host/ws">
      <Parameter name="SipCallId" value="{{SipCallId}}"/>
      <Parameter name="Caller"    value="{{Caller}}"/>
      <Parameter name="CallSid"   value="{{CallSid}}"/>
    </Stream>
  </Connect>
</Response>
```

**Important:** Pipecat's built-in dev runner
(`python bot.py -t twilio -x ...`) ships a hardcoded TwiML response with
**no** `<Parameter>` tags. You must run your own webhook server. Minimal
example:

```python
# twiml_server.py — run alongside your bot
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()

@app.post("/twiml")
async def twiml(request: Request) -> Response:
    form = await request.form()
    params = "".join(
        f'<Parameter name="{k}" value="{v}"/>'
        for k, v in form.items()
        if k in {"SipCallId", "Caller", "CallSid"} or k.startswith("SipHeader_")
    )
    host = request.headers.get("host", "")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws">{params}</Stream>
  </Connect>
</Response>"""
    return Response(content=xml, media_type="application/xml")

# Plus a WebSocket route at /ws that calls into your bot()
# (see examples/nova_clinic_pipecat for a full pattern)
```

Point Twilio's Voice URL at `https://your-host/twiml`.

> **Why this is required:** the SIP Call-ID exists on Twilio's HTTP webhook
> but not on the Media Streams WebSocket. Without `<Parameter>` tags, there
> is no path for it to reach the bot — the SDK will see only the Twilio
> `CallSid` and fall back to that.

### Telnyx, Plivo, Exotel

These providers deliver SIP info on the WebSocket directly. The bot side is
unchanged — `observer.attach_sip_from_telephony(call_data)` finds the native
call id automatically. To capture the actual SIP Call-ID, configure your
trunk/XML to forward it as a customParameter:

* **Telnyx** — add `<CustomParameter name="SipCallId" value="{{sip_call_id}}"/>`
  to your `<Stream>` element.
* **Plivo** — add `extraHeaders="SipCallId={{sip_call_id}}"` to `<Stream>`.
* **Exotel** — add `custom_parameters: SipCallId=<%CallSid%>` in the App
  Bazaar Voicebot applet.

Fallback when nothing is forwarded:

| Provider | Fallback id     | Native key on `call_data` |
|----------|-----------------|---------------------------|
| Telnyx   | call_control_id | `call_control_id`         |
| Plivo    | callId          | `call_id`                 |
| Exotel   | call_sid        | `call_id`                 |

### Daily PSTN/SIP dial-in

Daily delivers SIP info in the dial-in webhook payload — no XML required.
Pass it straight to the SDK:

```python
observer.attach_sip_from_dialin(runner_args.body["dialin_settings"])
```

`DialinSettings.call_id` and `DialinSettings.sip_headers` reach Tuner
as-is.

### Custom SIP trunk / direct termination

If you terminate SIP yourself (Asterisk, FreeSWITCH, Kamailio, …), pass
the values straight in:

```python
observer.attach_sip_info(
    sip_call_id=my_sip_session.call_id,
    sip_headers=my_sip_session.headers,
)
```

### Debugging checklist

Both `sip_call_id` and `sip_headers` are optional in the final payload —
they are omitted entirely when unset, so non-SIP web calls remain
backward-compatible. If you expect SIP fields but see them missing:

| Symptom in the final payload | Likely cause | Fix |
|------------------------------|--------------|-----|
| `sip_call_id` is the Twilio `CA…` CallSid | TwiML did not include `<Parameter name="SipCallId" .../>` | Update your webhook TwiML (Twilio section above) |
| `sip_call_id` is `null` for a Twilio call | `attach_sip_from_telephony()` never ran | Call it after `parse_telephony_websocket` returns |
| `sip_call_id` is `null` for Daily PSTN | `attach_sip_from_dialin()` not wired | Call it from `runner_args.body["dialin_settings"]` |
| `sip_headers` is `null` but `sip_call_id` is set | Headers were not in customParameters | Pass them explicitly via `attach_sip_info(sip_headers=...)` |

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
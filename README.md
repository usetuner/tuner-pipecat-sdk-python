# tuner-pipecat-sdk

`tuner-pipecat-sdk` is a lightweight observer SDK for [`pipecat`](https://github.com/pipecat-ai/pipecat) and [`pipecat-flows`](https://github.com/pipecat-ai/pipecat-flows).
It captures flow transitions, latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.


## Requirements

- Python **3.10–3.13**. 
- **Do not use Python 3.14** for installs yet: Pipecat pulls **`onnxruntime~=1.23.2`** and **`numba`** without 3.14 wheels → errors like *No matching distribution found for onnxruntime*.
- This SDK depends on **`pipecat-ai>=1.0.0`**.

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
| `agent_version` | `int \| None` | `None` | Deployment version number — overrides `APP_VERSION` env var |

## Agent Version Tracking

Set `APP_VERSION` in your environment to track which version of your agent handled each call:

```bash
APP_VERSION=42 python agent.py start
```

Tuner reads it automatically — bump the number on every deployment.

**Using GitHub Actions or CircleCI?** No setup needed — Tuner reads the build number from `GITHUB_RUN_NUMBER` / `CIRCLE_BUILD_NUM` automatically.

**Manual override** — takes priority over all env vars:

```python
Observer(
    ...
    agent_version=42,
)
```

`agent_version` is omitted from the payload entirely when no value is found.

---

## SIP / Telephony Calls

The observer can capture the SIP Call-ID and SIP headers and forward them to
Tuner. The SDK call is **one line per provider** — your server passes its
raw payload straight through and the SDK handles all SIP-field extraction
internally.

The SDK call:

```python
# Any built-in provider — pass the raw payload your server already has:
observer.attach_sip_from_telephony(payload, provider="twilio")
observer.attach_sip_from_telephony(payload, provider="telnyx")
observer.attach_sip_from_telephony(payload, provider="plivo")
observer.attach_sip_from_telephony(payload, provider="exotel")
observer.attach_sip_from_telephony(payload, provider="jambonz")

# Unlisted provider — pass your own extractor:
def my_extractor(payload):
    return payload["my_id"], payload.get("my_headers")
observer.attach_sip_from_telephony(payload, provider=my_extractor)

# Daily PSTN/SIP dial-in:
observer.attach_sip_from_dialin(runner_args.body["dialin_settings"])

# Anything else (you already have the values):
observer.attach_sip_info(sip_call_id="...", sip_headers={...})
```

`provider=` is **required** for `attach_sip_from_telephony`. Built-in
strings: `"twilio"`, `"telnyx"`, `"plivo"`, `"exotel"`, `"jambonz"`. An
unknown string raises `ValueError` — pass a callable instead for any
provider not in the list.

### What the SDK does automatically

The built-in extractors look for a SIP-layer Call-ID in the
provider-specific location, then fall back to the provider's native call
id when no SIP-layer key was forwarded:

| Provider | Expected `payload`                       | SIP-layer source                          | Native fallback           |
|----------|------------------------------------------|-------------------------------------------|---------------------------|
| Twilio   | `parse_telephony_websocket` output       | `body[SipCallId]` (case-insensitive)      | `call_id` (CallSid)       |
| Telnyx   | Call Control event **or** WS `call_data` | `data.payload.sip_call_id` / `sip_headers` list / `customParameters` | `call_control_id`         |
| Plivo    | `parse_telephony_websocket` output       | `customParameters` / `extra_headers`      | `call_id`                 |
| Exotel   | `parse_telephony_websocket` output       | `custom_parameters`                       | `call_id`                 |
| Jambonz  | call-hook webhook JSON body              | `sip.headers[X-CID]` → `Call-ID` → `SipCallId` → `sip.call_id` | `call_sid` |

Headers in the same location populate `sip_headers` automatically;
override with the `sip_headers=` kwarg when you want to send something
different.

For the SIP-layer Call-ID to actually appear, **your trunk/webhook must
forward it**. The per-provider configuration is below.

### Twilio Media Streams (most common case)

Twilio's HTTP voice webhook receives SIP context (`SipCallId`, `Caller`,
every `SipHeader_*`) but the Media Streams WebSocket drops those fields
by default. The SDK ships two helpers so your integration stays short and
correct:

* `build_sip_forwarding_twiml(form, ws_url=...)` — returns the TwiML
  response that forwards every SIP-relevant field as a `<Parameter>`
  tag. Handles XML escaping and `SipHeader_*` auto-forwarding.
* `TwilioCallContext` — typed dataclass with `.sip_call_id`,
  `.from_number`, `.to_number`, `.stream_sid`, `.raw_headers`. Built via
  `TwilioCallContext.from_call_data(call_data)` where `call_data` is the
  output of Pipecat's `parse_telephony_websocket`.

Server wiring (in your application):

```python
from tuner_pipecat_sdk.providers.twilio import (
    TwilioCallContext,
    build_sip_forwarding_twiml,
)

@app.post("/twiml")
async def twiml(request):
    form = await request.form()
    xml = build_sip_forwarding_twiml(form, ws_url="wss://your-host/ws")
    return Response(content=xml, media_type="application/xml")

@app.websocket("/ws")
async def ws(websocket):
    _, call_data = await parse_telephony_websocket(websocket)
    ctx = TwilioCallContext.from_call_data(call_data)
    await run_bot(transport, sip_context=ctx)
```

Bot — single line:

```python
observer.attach_sip_from_context(ctx)
```

Point Twilio's Voice URL at `https://your-host/twiml`. See
`examples/nova_clinic_pipecat/twilio_server.py` for the full pattern.

> **Why this matters:** Pipecat's built-in dev runner ships a hardcoded
> TwiML response with no `<Parameter>` tags, so the SIP Call-ID never
> reaches the bot — Tuner sees only the Twilio `CallSid` and falls back
> to it. You must run your own webhook server using
> `build_sip_forwarding_twiml` (or equivalent) to get the real SIP id.

### Telnyx, Plivo, Exotel

Bot side stays one line:

```python
observer.attach_sip_from_telephony(payload, provider="telnyx")  # or "plivo" / "exotel"
```

To capture the actual SIP Call-ID, configure your trunk/XML to forward it
as a customParameter:

* **Telnyx** — add `<CustomParameter name="SipCallId" value="{{sip_call_id}}"/>`
  to your `<Stream>` element. With **Call Control** (JSON webhook + answer
  command), cache the inbound event payload and pass it through verbatim
  — the SDK reads `sip_headers` / `custom_headers` lists directly.
* **Plivo** — add `extraHeaders="SipCallId={{sip_call_id}}"` to `<Stream>`.
* **Exotel** — add `custom_parameters: SipCallId=<%CallSid%>` in the App
  Bazaar Voicebot applet.

Fallback when nothing is forwarded:

| Provider | Fallback id     | Native key on `call_data` |
|----------|-----------------|---------------------------|
| Telnyx   | call_control_id | `call_control_id`         |
| Plivo    | callId          | `call_id`                 |
| Exotel   | call_sid        | `call_id`                 |

### Jambonz

Jambonz delivers SIP info on the **call-hook webhook** (the JSON `POST` to
your application URL), not the WebSocket. The SDK ships two helpers so
your integration is short and correct:

* `JambonzCallContext` — typed dataclass with `.sip_call_id`,
  `.from_number`, `.to_number`, `.direction`, `.raw_headers`. Built via
  `JambonzCallContext.from_webhook(payload)`. SIP Call-ID resolution
  priority: `X-CID` → `Call-ID` → `SipCallId` → `sip.call_id` → `call_sid`.
* `JambonzPendingStore` — bridges webhook → WebSocket with a 30s TTL
  (configurable) and `asyncio.Event` await semantics, so the WS handler
  never races the webhook.

Server wiring (in your application):

```python
from tuner_pipecat_sdk.providers.jambonz import (
    JambonzCallContext,
    JambonzPendingStore,
)

pending = JambonzPendingStore()

@app.post("/")
async def call_hook(request):
    data = await request.json()
    pending.park(JambonzCallContext.from_webhook(data))
    return _jambonz_verbs(...)

@app.websocket("/ws")
async def ws(websocket):
    ...  # read first frame to learn call_sid
    ctx = await pending.wait_and_pop(call_sid) or JambonzCallContext.fallback(call_sid)
    await run_bot(transport, sip_context=ctx)
```

Bot — single line:

```python
observer.attach_sip_from_context(ctx)
```

`attach_sip_from_context` is duck-typed: any object exposing
`sip_call_id` and `raw_headers` attributes works, so you can pass your
own context type if you've wrapped Jambonz's payload further.

See `examples/nova_clinic_pipecat/jambonz_server.py` for the full
pattern.

SIP Call-ID resolution priority: `X-CID` → `Call-ID` → `SipCallId` →
`sip.call_id` → `call_sid`. The `X-CID` header is the canonical
cross-system id for chains like LiveKit SIP → Jambonz, where Jambonz
regenerates the SIP `Call-ID` on each hop. See
`examples/nova_clinic_pipecat/jambonz_server.py` for a full pattern.

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
| `sip_call_id` is `null` for a Twilio call | `attach_sip_from_telephony(..., provider="twilio")` never ran | Call it after `parse_telephony_websocket` returns |
| `sip_call_id` is the Jambonz `call_sid` | No `X-CID` / `Call-ID` header in the webhook | Configure your upstream trunk to add it, or use a custom callable extractor |
| `sip_call_id` is `null` for Daily PSTN | `attach_sip_from_dialin()` not wired | Call it from `runner_args.body["dialin_settings"]` |
| `sip_headers` is `null` but `sip_call_id` is set | Headers were not in customParameters | Pass them explicitly via `attach_sip_info(sip_headers=...)` |
| `ValueError: unknown provider …` | Misspelled built-in name | Use `"twilio"`/`"telnyx"`/`"plivo"`/`"exotel"`/`"jambonz"` or pass a callable |

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
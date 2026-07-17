# tuner-pipecat-sdk

`tuner-pipecat-sdk` is a lightweight observer SDK for [`pipecat`](https://github.com/pipecat-ai/pipecat).
It captures latency signals, transcript segments, and usage metadata,
then sends a structured `CallPayload` to the Tuner API when a call ends.


## Requirements

- Python **3.11–3.13**. 
- **Do not use Python 3.14** for installs yet: Pipecat pulls **`onnxruntime~=1.23.2`** and **`numba`** without 3.14 wheels → errors like *No matching distribution found for onnxruntime*.
- This SDK depends on **`pipecat-ai>=1.0.0`**.
- LangChain/LangGraph support (`Observer.wrap_graph`/`wrap_chain`) is fully optional —
  see [LangChain / LangGraph observability](#langchain--langgraph-observability).

## Installation

```bash
pip install tuner-pipecat-sdk
```

## Quick Start Example
---

## Plain Pipecat — `Observer`

```python
import uuid
from tuner_pipecat_sdk import Observer

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
```

---

## Pipeline Setup

`Observer` is a **pipeline-level observer** — register it on the `PipelineTask`
(`observers=[...]`), not as a processor inside `Pipeline([...])`. It then sees every frame at
every processor boundary and stays out of the audio path:

```python
Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

Enable metrics on the pipeline task so latency and usage fields are populated, and add the
observer to `observers=[...]`:

```python
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver

turn_tracker = TurnTrackingObserver()
observer.attach_turn_tracking_observer(turn_tracker)

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
    observers=[observer, observer.latency_observer, turn_tracker],
)
```

Without these flags the observer will log warnings and LLM/TTS metric fields will be absent from the payload.
For more example check https://github.com/usetuner/tuner-pipecat-sdk-python/tree/main/examples

## What Happens at Call End

When Pipecat emits `EndFrame` or `CancelFrame`, the observer:

1. Builds a typed `CallPayload` from everything captured during the call.
2. Sends it to the Tuner API in the background — this does not block your pipeline from shutting down.

If the request fails (network error, timeout, bad API key, Tuner API down), the SDK logs the failure and moves on. It never raises and never blocks or crashes your pipeline. That also means a failed delivery is easy to miss — if a call isn't showing up in Tuner, check your logs for `[tuner] failed to deliver call`.

## Observer Parameters
`Observer` accepts the following constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | — | Tuner API key |
| `workspace_id` | `int` | — | Tuner workspace ID |
| `agent_id` | `str` | — | Agent identifier |
| `call_id` | `str` | — | Unique call ID (use `uuid4()`) |
| `base_url` | `str` | `https://api.usetuner.ai` | Tuner API base URL |
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

## Cost Tracking

Pass a `cost_calculator` callable to compute call cost from usage data. The result is included in the payload sent to the Tuner API.

```python
from tuner_pipecat_sdk import Observer, CallUsage

def calculate_cost(usage: CallUsage) -> float:
    llm_cost  = (usage.llm_prompt_tokens     or 0) * 0.000_003
    llm_cost += (usage.llm_completion_tokens or 0) * 0.000_015
    tts_cost  = (usage.tts_characters        or 0) * 0.000_030
    stt_cost  = usage.stt_audio_seconds            * 0.000_006
    return llm_cost + tts_cost + stt_cost

observer = Observer(
    ...,
    cost_calculator=calculate_cost,
)
```

`CallUsage` fields:

| Field | Type | Description |
|---|---|---|
| `llm_prompt_tokens` | `int \| None` | Input tokens sent to the LLM |
| `llm_completion_tokens` | `int \| None` | Output tokens generated by the LLM |
| `llm_total_tokens` | `int \| None` | Total LLM tokens (prompt + completion) |
| `tts_characters` | `int \| None` | Characters sent to TTS |
| `stt_audio_seconds` | `int` | Call wall-clock duration in seconds (used as STT duration proxy) |

Token and character counts are `None` when no metrics were collected for that service during the call.

The callable must return the cost in **USD cents** (e.g. `0.5` = half a cent, `1.0` = one cent).

## LangChain / LangGraph observability

If your agent's "LLM step" is a LangChain runnable or LangGraph graph — driven through
pipecat's built-in `LangchainProcessor`, or your own custom processor — install the
optional `langchain` extra and wrap it with `Observer.wrap_graph()` / `Observer.wrap_chain()`:

```bash
pip install tuner-pipecat-sdk[langchain]
```

```python
from pipecat.processors.frameworks.langchain import LangchainProcessor
from tuner_pipecat_sdk import Observer

observer = Observer(api_key=..., workspace_id=..., agent_id=..., call_id=...)

wrapped_graph = observer.wrap_graph(my_compiled_graph)   # or observer.wrap_chain(my_chain)
lc = LangchainProcessor(chain=wrapped_graph)

pipeline = Pipeline([
    transport.input(),
    stt,
    user_aggregator,
    lc,                 # in place of a native LLM service
    tts,
    transport.output(),
    assistant_aggregator,
])
```

This captures tool calls, LangGraph node transitions, and LLM token usage from inside
the wrapped runnable via [`tuner-langchain`](https://github.com/usetuner/tuner-langchain)'s
callback handler, and merges them into the same `transcript_with_tool_calls` timeline
used by plain pipecat pipelines — independent of whatever pipecat frames the driving
processor does or doesn't emit.

Runnable examples: [`examples/pizza_order_langchain/`](examples/pizza_order_langchain/)
(`wrap_chain()`, a raw LangChain tool-calling chat model) and
[`examples/nova_clinic_langgraph/`](examples/nova_clinic_langgraph/) (`wrap_graph()`, a
LangGraph `create_react_agent` — demonstrates node-transition tracking).

**This extra is entirely optional.** If `tuner-langchain` isn't installed, calling
`wrap_graph`/`wrap_chain` raises a clear `ImportError` with install instructions;
everything else about the SDK behaves identically to a plain pipecat pipeline.

**Known limitations (inherited from pipecat, not fixed by this integration):**

- Pipecat's built-in `LangchainProcessor` only forwards the *latest* user message to
  the chain/graph (no full conversation history) — multi-turn memory has to be handled
  by your LangChain/LangGraph state itself (e.g. LangGraph checkpointing).
- `LangchainProcessor` is a bare `FrameProcessor`, not a pipecat `LLMService`, so it
  never emits pipecat's own latency/processing metrics for the LLM step — the
  `llm_node_ttft` field would otherwise always be absent for LangChain-driven turns.
  Instead, it's populated from LangChain's own `on_llm_start`→`on_llm_end` call
  duration (via the same session hook as usage/cost below). This is a **total call
  duration, not a strict time-to-first-token** — if a turn makes multiple LLM calls
  (e.g. a tool-calling round-trip), only the *last* call's duration is reported, not
  a sum across the turn.

## Public API

- `tuner_pipecat_sdk.Observer` — for pipecat pipelines
  - `Observer.wrap_graph(graph, capture=None)` / `Observer.wrap_chain(chain, capture=None)` —
    optional, requires the `langchain` extra (see above)
- `tuner_pipecat_sdk.TunerConfig`
- `tuner_pipecat_sdk.CallUsage` — usage data type for `cost_calculator`

Payload and transcript schemas are available under `tuner_pipecat_sdk.models`, including
`NodeInfo` (LangGraph node transitions) and `ToolInfo` (tool calls/results, shared
between native and LangChain-sourced rows).


## Building this project

```bash
pip install -e ".[dev]"   # test, lint, type-check, and publish tooling
pytest                     # run tests
ruff check .               # lint
```

Full environment setup (Python version, examples, publishing to PyPI): see [setup_guide.md](setup_guide.md).

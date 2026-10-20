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

## Trace view (OpenTelemetry)

Pipecat already emits OpenTelemetry spans for a conversation — STT, LLM, TTS, tool calls,
turn boundaries. The SDK forwards them to Tuner, which shows them as a trace tree on the call
details page: what ran, in what order, and how long each step took.

### Setup

Two installs and one flag.

```bash
pip install 'pipecat-ai[tracing]'          # lets Pipecat emit spans at all
pip install 'tuner-pipecat-sdk[traces]'    # lets this SDK forward them to Tuner
```

```python
task = PipelineTask(
    pipeline,
    params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    observers=[observer, observer.latency_observer, turn_tracker],
    enable_tracing=True,        # <-- the one thing you must set yourself
    enable_turn_tracking=True,  # recommended: gives the trace its turn structure
)
```

`enable_tracing=True` is Pipecat's own switch for emitting spans. You have to set it because
you build the `PipelineTask` — the `Observer` is constructed before the task exists and is
handed *to* it, so the SDK has no way to reach that argument.

Everything else is automatic. You do **not** need `setup_tracing(...)`, a `TracerProvider`,
an exporter, or any `OTEL_*` environment variable — the SDK builds all of that from the API
key and base URL you already pass to `Observer`, and tags every span with the call id so the
trace lands on the right call.

### Turning it off

```python
Observer(..., forward_traces=False)
```

Leaves the rest of the SDK untouched — calls are still reported to Tuner, only the spans stop
being forwarded. Note this is *our* switch; Pipecat's `enable_tracing` above is a separate
one that controls whether Pipecat produces spans at all.

### Notes

- **Already exporting traces elsewhere?** Your setup is preserved. If you call Pipecat's
  `setup_tracing(...)`, the SDK adds Tuner as an extra destination instead of replacing it.
- **No `traces` extra installed?** Trace forwarding becomes a no-op with a debug log. It can
  never fail a call.
- **Use the HTTP exporter if you configure one yourself.** Tuner accepts OTLP over HTTP with
  protobuf; gRPC is not supported. Pipecat's own tracing example uses the gRPC exporter, so
  this is the one line to change if you started from it.

### What is stored

The span tree — names, timings, parent relationships, status — plus metrics and
configuration: model names, time-to-first-byte, token counts, turn latency, endpointing
settings, and tool names and outcomes.

No conversation content. Transcripts, LLM inputs and outputs, TTS text, system prompts, and
tool call arguments and results are all dropped on arrival and never stored. The filter is an
allowlist, so anything not explicitly approved is discarded by default.

Roughly 30 KB per call, for a five-minute call with twenty turns.

## End-of-Utterance (EOU) Delay

Wire the user context aggregator to capture **how long the pipeline took to decide the user
was done talking** — the gap between the user going silent and the turn actually closing.
This lands on every user row's `eou_delay`, `eou_confidence`, and `eou_reason` metadata fields.

```python
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
observer.attach_user_aggregator(user_aggregator)
```

`eou_reason` tells you which signal closed the turn, and `eou_delay` is calculated differently depending on that signal: for VAD-driven turns the SDK measures it directly from the frame stream, while for model-driven turns it's the turn-detection model's own reported figure.

| `eou_reason` | What closed the turn | `eou_delay` measures | Source |
|---|---|---|---|
| `silence_timeout` | Local VAD silence threshold | ms from the last VAD-detected speech-end to the turn closing | Measured by the SDK from the frame stream (wall-clock) |
| `model_verdict` | A turn-completion model (e.g. Smart Turn) decided the turn was done | the model's own end-to-end decision time; `eou_confidence` carries its probability | Reported by pipecat `(TurnMetricsData.e2e_processing_time_ms)` — the model's internal timing, not a wall-clock measurement |
| `stt_endpoint` | Server-side STT (e.g. Deepgram Flux) decided end-of-turn on its own clock | not populated — the decision happens server-side, independent of local VAD | — |



All three fields are simply omitted from a row's metadata when nothing decided that turn's
end (e.g. a pipeline that doesn't use Pipecat's turn-strategy stack yet)

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
| `extra_metadata` | `dict \| None` | `None` | Arbitrary key-value data merged into every call's metadata (see [Extra Metadata](#extra-metadata)) |
| `forward_traces` | `bool` | `True` | Forward Pipecat's OpenTelemetry spans to Tuner (see [Trace view](#trace-view-opentelemetry)) |

## Extra Metadata

Attach arbitrary key-value data to every call record:

```python
Observer(
    api_key=..., workspace_id=..., agent_id=..., call_id=...,
    extra_metadata={
        "env": "production",
        "region": "us-east-1",
        "deployment": "v2.3.1",
    },
)
```

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

Tuner links a simulation call to your agent's call through the SIP data on the
inbound INVITE: the `X-Correlation-Id` header and the SIP Call-ID. Your call
webhook sees them, but the media websocket does not, so the SDK gives you three
steps, the same for every carrier:

| Step | API |
|---|---|
| Read | `extract_sip(payload, *, provider) -> SipInfo(sip_call_id, sip_headers, carrier_call_id)` |
| Forward | `forward_to_websocket(info, *, ws_url) -> Forward` (`.twiml` for Twilio, `.stream` for Telnyx) |
| Attach | `observer.attach_sip(info=None, *, sip_call_id=None, sip_headers=None, recipient=None)` |
| Add a carrier | `register_provider(provider)` |

Built-in providers: `"twilio"`, `"telnyx"`, `"plivo"`, `"exotel"`, `"jambonz"`,
`"daily"`. Forwarding is built in for Twilio and Telnyx.

### Twilio Media Streams

```python
from tuner_pipecat_sdk import extract_sip, forward_to_websocket

@app.post("/twiml")
async def twiml(request):
    info = extract_sip(await request.form(), provider="twilio")
    fwd = forward_to_websocket(info, ws_url="wss://your-host/ws")
    return Response(content=fwd.twiml, media_type="application/xml")
```

`fwd.twiml` is `<Connect><Stream url>` with one `<Parameter>` per SIP field:
`SipCallId`, `CallSid` and every `SipHeader_*`. In the websocket handler, read
them from the `start` frame (`start.customParameters`):

```python
start = ...  # the websocket "start" message, or parse_telephony_websocket's call_data
observer.attach_sip(extract_sip(start, provider="twilio"))
```

> Pipecat's built-in dev runner returns TwiML with no `<Parameter>` tags, so
> the SIP data never reaches the bot. Serve `fwd.twiml` from your own webhook.

### Telnyx Call Control

Answer the call, then start the stream with `fwd.stream` as the
`streaming_start` body. It carries `X-Correlation-Id` and `SipCallId` as
`custom_parameters`:

```python
info = extract_sip(webhook_json, provider="telnyx")      # call.initiated
fwd = forward_to_websocket(info, ws_url="wss://your-host/ws")
await telnyx_post(f"/v2/calls/{call_control_id}/actions/answer", {})
await telnyx_post(f"/v2/calls/{call_control_id}/actions/streaming_start", fwd.stream)
```

Telnyx sends them back on the websocket `start` frame as `start.custom_parameters`.
Pass that raw frame: Pipecat's `parse_telephony_websocket` drops `custom_parameters`
from its Telnyx `call_data`:

```python
observer.attach_sip(extract_sip(start_message, provider="telnyx"))
```

`fwd.stream` defaults to inbound-track, bidirectional RTP PCMU; override any
field with `{**fwd.stream, "stream_track": "both_tracks"}`. `answer` itself
rejects `custom_parameters` (422, code 10000), so it takes two commands.

TeXML works too: put `<Parameter name="X-Correlation-Id" value="..."/>` in `<Stream>`.

### Plivo, Exotel, Jambonz, Daily

```python
observer.attach_sip(extract_sip(call_data, provider="plivo"))     # or "exotel"
observer.attach_sip(extract_sip(call_hook_json, provider="jambonz"))
observer.attach_sip(extract_sip(runner_args.body["dialin_settings"], provider="daily"))
```

For Plivo add `extraHeaders="SipCallId={{sip_call_id}}"` to `<Stream>`; for
Exotel add `custom_parameters: SipCallId=<%CallSid%>` in the Voicebot applet.
Jambonz delivers SIP data on the call-hook webhook, not the websocket;
`JambonzPendingStore` bridges the two (see
`examples/nova_clinic_pipecat/jambonz_server.py`).

### Custom SIP trunk / your own values

```python
observer.attach_sip(sip_call_id=my_call.call_id, sip_headers=my_call.headers)
```

### Add a carrier

```python
from tuner_pipecat_sdk import Forward, SipInfo, register_provider

class AcmeProvider:
    name = "acme"

    def extract(self, payload) -> SipInfo:
        return SipInfo(sip_call_id=payload["sip_call_id"], sip_headers=payload["headers"])

    # Optional: lets forward_to_websocket(info, ws_url=..., provider="acme") work.
    def forward_to_websocket(self, info: SipInfo, ws_url: str) -> Forward: ...

register_provider(AcmeProvider())
```

### Merge rules

`attach_sip` is safe to call more than once. `sip_headers` merge: the first
non-empty value for a header name wins (case-insensitive). A carrier ID
(`CallSid`, `call_control_id`) only fills `sip_call_id` when it is empty; a real
SIP Call-ID replaces it but never another real one. An explicit `sip_call_id=`
always wins. Twilio `SipHeader_X-Foo` is also reported as `X-Foo`. A correlation
header that is not 32 hex or 26 base32 is dropped; see
`valid_correlation_id`.

### Deprecated

These still work and emit a `FutureWarning` once:

| Deprecated | Use |
|---|---|
| `observer.attach_sip_from_telephony(payload, provider=...)` | `observer.attach_sip(extract_sip(payload, provider=...))` |
| `attach_sip_from_telephony(payload, provider=<callable>)` | `register_provider(...)` |
| `observer.attach_sip_from_context(ctx)` | `observer.attach_sip(extract_sip(payload, provider=...))` |
| `observer.attach_sip_from_dialin(settings)` | `observer.attach_sip(extract_sip(settings, provider="daily"))` |
| `observer.attach_sip_info(sip_call_id=, sip_headers=)` | `observer.attach_sip(sip_call_id=, sip_headers=)` |
| `providers.twilio.build_sip_forwarding_twiml(form, ws_url=)` | `forward_to_websocket(extract_sip(form, provider="twilio"), ws_url=).twiml` |

### Debugging checklist

| Symptom in the final payload | Likely cause | Fix |
|------------------------------|--------------|-----|
| `sip_call_id` is the Twilio `CA…` CallSid | TwiML had no `SipCallId` `<Parameter>` | Return `forward_to_websocket(...).twiml` |
| No `X-Correlation-Id` on a Telnyx call | Stream started without `custom_parameters` | Send `fwd.stream` as the `streaming_start` body |
| `sip_call_id` is `null` | `attach_sip` never ran | Call it with the websocket `start` frame |
| `ValueError: Unknown provider …` | Misspelled name | Use a built-in name or `register_provider` |

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

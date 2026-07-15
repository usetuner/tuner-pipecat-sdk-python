# Nova Clinic Bot — LangGraph

The same clinic-receptionist flow as [`nova_clinic_pipecat/`](../nova_clinic_pipecat/)
(check availability → book → look up → cancel), with its LLM step replaced by a
hand-built LangGraph `StateGraph` (router → booking/cancellation/general →
tools), instrumented with `Observer.wrap_graph()` instead of a native
`OpenAILLMService`. WebRTC only — the telephony server variants
(Twilio/Telnyx/Jambonz) from `nova_clinic_pipecat/` are not replicated here;
this example is focused on the LangGraph integration itself.

**What it demonstrates (in addition to what `nova_clinic_pipecat/` already does):**
- Building the LLM step as a hand-built multi-node `StateGraph` — not the
  `create_react_agent` prebuilt, so each intent gets its own system prompt —
  run through pipecat's `LangchainProcessor`; see the module docstring at the
  top of `nova_clinic_langgraph.py` for why
- `Observer.wrap_graph()` capturing **node transitions** (not just tool calls) —
  its standout capability over `wrap_chain()` — see the root
  [README's LangChain / LangGraph section](../../README.md) for how the
  integration works
- Tools defined with LangChain's `@tool` decorator instead of pipecat's
  `FunctionSchema`/`llm.register_function()`
- Per-call graph + checkpointer construction, so conversation memory is scoped
  to a single call with no manual state reset needed between calls

## Conversation flow

```
greet
  ├─► booking: ask name/reason/date → check_availability → book_appointment
  └─► cancellation: ask name → get_appointment → confirm → cancel_appointment
           └─► (either path) → confirm details + end_call
```

## Prerequisites

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/)

## Setup

1. From this directory:

   ```bash
   uv sync
   ```

2. Create a `.env` file (see `.env.example`):

   ```env
   OPENAI_API_KEY=your_openai_key

   # Optional — Tuner observability
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=nova-clinic-langgraph
   TUNER_BASE_URL=https://api.usetuner.ai
   ```

## Run

```bash
uv run nova_clinic_langgraph.py
```

Open http://localhost:7860 in your browser and click **Connect**.

## Services used

| Role | Service |
|------|---------|
| STT  | OpenAI gpt-4o-transcribe |
| LLM  | LangGraph `StateGraph`, hand-built multi-node router (OpenAI GPT-4o-mini) |
| TTS  | OpenAI tts-1 (alloy voice) |
| Transport | SmallWebRTC (default) |

# Pizza Order Bot — LangChain

The same [Pizza Order Bot](../pizza_order/) flow, with its LLM step replaced by a
raw LangChain tool-calling chat model, instrumented with `Observer.wrap_chain()`
instead of a native `OpenAILLMService`.

**What it demonstrates (in addition to everything `pizza_order/` already does):**
- Building the LLM step as a chat model with tools bound, orchestrated by a manual
  tool-calling loop (`build_agent_chain()`) run through pipecat's `LangchainProcessor`
  — LangChain 1.x removed `AgentExecutor`/`create_tool_calling_agent`, so this is the
  simplest shape `wrap_chain()` targets (tool calls only, no graph nodes); see the
  note at the top of `pizza_order_langchain.py` for why
- `Observer.wrap_chain()` capturing tool calls and LLM token usage from the
  LangChain side — see the root [README's LangChain / LangGraph section](../../README.md)
  for how the integration works
- Tools defined with LangChain's `@tool` decorator instead of pipecat's
  `FunctionSchema`/`llm.register_function()`

## Conversation flow

```
greet + present menu
   └─► choose_pizza
           └─► choose_size
                 └─► confirm_order
                         ├─► (confirmed)  → thank + end_call
                         └─► (cancelled)  → apologise + end_call
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
   DEEPGRAM_API_KEY=your_deepgram_key
   OPENAI_API_KEY=your_openai_key

   # Optional — Tuner observability
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=pizza-order-langchain-bot
   TUNER_BASE_URL=https://api.usetuner.ai
   ```

## Run

```bash
uv run pizza_order_langchain.py
```

Open http://localhost:7860 in your browser and click **Connect**.

## Services used

| Role | Service |
|------|---------|
| STT  | Deepgram Nova-3 |
| LLM  | LangChain tool-calling chat model, manual loop (OpenAI GPT-4o-mini) |
| TTS  | Deepgram Aura-2 |
| Transport | SmallWebRTC (default) |

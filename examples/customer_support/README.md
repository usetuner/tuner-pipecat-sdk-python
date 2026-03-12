# Customer Support Bot

A multi-turn customer support bot built with Pipecat Flows and the `pipecat-flows-tuner` SDK.

**What it demonstrates:**
- Multi-node flow: greeting → issue category → context collection → resolution → farewell
- State accumulated across nodes (`name`, `category`, `description`, `resolved`)
- Conditional farewell branch — resolved in-call vs. escalated to a human agent
- Full call observability via `FlowsObserver` (latency, transcript, node transitions)

## Flow diagram

```
greeting
   └─► issue_category
           └─► collect_context
                   └─► resolution
                           ├─► farewell (resolved)
                           └─► farewell (escalated)
```

## Prerequisites

- Python 3.10+
- `uv` package manager

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file:

   ```env
   CARTESIA_API_KEY=your_cartesia_key
   DEEPGRAM_API_KEY=your_deepgram_key
   OPENAI_API_KEY=your_openai_key

   # Optional — Tuner observability (defaults to local dev server)
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=customer-support-bot
   TUNER_BASE_URL=http://localhost:8000
   ```

## Run

```bash
uv run customer_support.py
```

Open http://localhost:7860 in your browser and click **Connect**.

## Services used

| Role | Service |
|------|---------|
| STT  | Deepgram Nova-3 |
| LLM  | OpenAI GPT-4o-mini |
| TTS  | Cartesia Sonic |
| Transport | SmallWebRTC (default) |

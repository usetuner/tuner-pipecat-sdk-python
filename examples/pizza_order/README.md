# Pizza Order Bot

A pizza ordering bot built with Pipecat and the `tuner-pipecat-sdk` `Observer`.

**What it demonstrates:**
- LLM tool-calling order flow: choose pizza → choose size → confirm → end call
- Running price calculation (pizza price + size surcharge)
- Confirmation with a cancellation branch
- Full call observability via the plain `Observer`
- Per-call cost reporting via a `cost_calculator`

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
- SDK install paths (PyPI, `pip` vs examples): see the **repository root README**.

## Setup

1. From this directory:

   ```bash
   uv sync
   ```

2. Create a `.env` file:

   ```env
   DEEPGRAM_API_KEY=your_deepgram_key
   OPENAI_API_KEY=your_openai_key

   # Optional — Tuner observability (defaults to local dev server)
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=pizza-order-bot
   TUNER_BASE_URL=http://localhost:8000
   ```

## Run

```bash
uv run pizza_order.py
```

Open http://localhost:7860 in your browser and click **Connect**.

## Services used

| Role | Service |
|------|---------|
| STT  | Deepgram Nova-3 |
| LLM  | OpenAI GPT-4o-mini |
| TTS  | Deepgram Aura-2 |
| Transport | SmallWebRTC (default) |

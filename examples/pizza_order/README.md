# Pizza Order Bot

A full pizza ordering bot built with Pipecat Flows and the `pipecat-flows-tuner` SDK.

**What it demonstrates:**
- 7-node ordering flow: pizza → toppings → size → fulfilment → (address) → confirm → farewell
- Delivery vs. pickup branching — address node is skipped for pickup orders
- Running price calculation accumulated across nodes
- Confirmation with cancellation branch
- Full call observability via `FlowsObserver`

## Flow diagram

```
greeting (choose pizza)
   └─► toppings
           └─► size
                 └─► fulfilment
                         ├─► address ──┐  (delivery only)
                         │             ▼
                         └─────────► confirm
                                         ├─► farewell (confirmed)
                                         └─► farewell (cancelled)
```

## Prerequisites

- Python 3.10+, [`uv`](https://docs.astral.sh/uv/)
- SDK install paths (PyPI, `pip` vs examples): see the **repository root README**.

## Setup

1. From this directory:

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
| TTS  | Cartesia Sonic |
| Transport | SmallWebRTC (default) |

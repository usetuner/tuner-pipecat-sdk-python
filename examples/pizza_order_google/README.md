# Pizza Order Bot — Google Stack

The same [Pizza Order Bot](../pizza_order/) flow, running entirely on Google Cloud
services (Speech-to-Text, Gemini, Text-to-Speech) instead of Deepgram/OpenAI.

**What it demonstrates (in addition to everything `pizza_order/` already does):**
- Google Cloud Speech-to-Text V2 for STT
- `GoogleLLMService` (Gemini 2.5 Flash) for the LLM step
- Google Cloud TTS (Chirp 3 HD voice) for TTS

## Conversation flow

Same as [`pizza_order/`](../pizza_order/) — see its README for the flow diagram.

## Prerequisites

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/)
- A Google Cloud project with Speech-to-Text, Text-to-Speech, and the Gemini API
  enabled, and credentials available locally.

## Setup

1. From this directory:

   ```bash
   uv sync
   ```

2. Create a `.env` file:

   ```env
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   GOOGLE_API_KEY=your_gemini_api_key
   GOOGLE_TTS_VOICE=en-US-Chirp3-HD-Charon
   GOOGLE_LLM_MODEL=gemini-2.5-flash

   # Optional — Tuner observability
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=pizzeria-bot
   TUNER_BASE_URL=https://api.usetuner.ai   # override the staging default below for your own workspace
   ```

## Run

```bash
uv run pizza_order_google.py
```

Open http://localhost:7860 in your browser and click **Connect**.

## Services used

| Role | Service |
|------|---------|
| STT  | Google Cloud Speech-to-Text V2 |
| LLM  | Google Gemini 2.5 Flash |
| TTS  | Google Cloud TTS (Chirp 3 HD) |
| Transport | SmallWebRTC (default) |

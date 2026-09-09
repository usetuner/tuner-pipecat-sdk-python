# Pizza Order Bot — Speechmatics

Same pizza ordering flow as [`pizza_order`](../pizza_order), with the audio
stack swapped for **Speechmatics** (both STT and TTS). Run it alongside
`pizza_order` to A/B Speechmatics against the Deepgram audio stack on an
identical flow.

## Services used

| Role | Service |
|------|---------|
| STT  | `SpeechmaticsSTTService` (Agent STT preview `linden-1`, or standard RT) |
| TTS  | `SpeechmaticsTTSService` (preview HTTP API, 16 kHz, voice `sarah`) |
| LLM  | `OpenAILLMService` (gpt-4o-mini) |
| VAD  | Silero (Pipecat-side endpointing) |

Turn detection is left to Pipecat's Silero VAD
(`TurnDetectionMode.EXTERNAL`) so the pipeline behaves exactly like the
Deepgram version. To let Speechmatics do the endpointing instead, switch to
`TurnDetectionMode.ADAPTIVE` or `SMART_TURN` and set
`end_of_utterance_silence_trigger`.

Speechmatics TTS is a **preview** service: it is HTTP (one request per
utterance, so time-to-first-byte is higher than a streaming socket like
Deepgram Aura-2) and 16 kHz only, which the output transport resamples. It
needs a shared `aiohttp.ClientSession`, created in `run_bot` and closed when
the pipeline finishes.

## Switching STT engines

`SPEECHMATICS_STT_MODE` in `.env` picks the engine:

| Mode | Endpoint | Model | Operating point |
|------|----------|-------|-----------------|
| `agent` (default) | `wss://preview.rt.speechmatics.com/v2/agent` | `linden-1` | not sent |
| `rt` | `wss://eu2.rt.speechmatics.com/v2` | not sent | `enhanced` |

Agent STT needs **both** the `/v2/agent` path and `model: "linden-1"` in the
transcription config — miss either and you silently get the standard RT API.
The mode sets both together, and logs which one it picked at startup:

```
[STT] Speechmatics mode=agent model=linden-1 url=wss://preview.rt.speechmatics.com/v2/agent
```

`SPEECHMATICS_RT_URL` and `SPEECHMATICS_STT_MODEL` override the mode's
defaults (a different region, or a newer model build) — leave them blank to
take the defaults above.

## Setup

1. From this directory:

   ```bash
   uv sync
   ```

2. Create a `.env` file (see `.env.example`):

   ```env
   SPEECHMATICS_API_KEY=your_speechmatics_key   # STT + TTS
   OPENAI_API_KEY=your_openai_key

   # STT engine: "agent" (linden-1 preview) or "rt"
   SPEECHMATICS_STT_MODE=agent

   # Optional — Tuner observability
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=pizza-order-bot-speechmatics
   TUNER_AGENT_VERSION=1   # integer; blank falls back to APP_VERSION / CI build number
   TUNER_BASE_URL=https://api.usetuner.ai
   ```

   Get a Speechmatics key from the [Speechmatics portal](https://portal.speechmatics.com/).
   Every optional var may be left blank — a blank value falls back to the
   default. In `rt` mode, set `SPEECHMATICS_RT_URL` if your account is not on
   the EU2 region. Voice names go straight into the TTS URL path, so see the
   Speechmatics TTS docs for the current list.

## Run

```bash
uv run pizza_order_speechmatics.py
```

Open http://localhost:7860 and click **Connect**.

# Pizza Order Bot

A pizza ordering bot built with Pipecat and the `tuner-pipecat-sdk` `Observer`.

**What it demonstrates:**
- LLM tool-calling order flow: choose pizza → choose size → confirm → end call
- Running price calculation (pizza price + size surcharge)
- Confirmation with a cancellation branch
- Full call observability via the plain `Observer`
- Per-call cost reporting via a `cost_calculator`
- Call recording: capturing the audio and uploading it to Tuner's presigned target

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

   # Optional — Tuner observability
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=pizza-order-bot
   TUNER_BASE_URL=https://api.usetuner.ai

   # Optional — set to false to run without uploading call audio
   TUNER_RECORD_CALLS=true
   ```

## Call recording

The call's audio is uploaded to Tuner by default. `TUNER_RECORD_CALLS=false`
turns it off; everything else about the bot is unchanged.

The SDK never uploads a recording. `recording_file_extension="wav"` opts the
call into a presigned S3 POST that the Tuner API issues when the call is
submitted, and the SDK hands that target to `on_recording_upload` and stops
there — the upload is the agent's to make, so it never sits on the call's
teardown path. `CallRecording` in [`pizza_order.py`](pizza_order.py) is that
other half:

| Piece | What it does |
|-------|--------------|
| `observer.audio_buffer()` | The pipecat `AudioBufferProcessor` for this call, or `None` when recording is off. Goes in the pipeline **after** `transport.output()`, where it sees both directions of audio. |
| `audio_buffer.start_recording()` | Called from `on_client_connected`. Pipecat's recorder starts **idle** and buffers nothing until this runs — skip it and the call ends with an empty buffer, `on_audio_data` never fires, and there is nothing to upload. |
| `on_audio_data` | Fires once at the end of the call with the whole recording as **raw PCM**. `to_wav()` adds the RIFF header — without it the upload succeeds and the file will not play. |
| `on_recording_upload` | Fires with the presigned target (`url`, `fields`, `key`, `content_type`, `max_bytes`) once the call has been submitted. |
| `wait_for_upload()` | Runs after `runner.run(task)` and does the POST. The observer flushes the call in a background task, so the target does not exist yet when the pipeline ends — returning any earlier would drop the recording. |

The two events arrive independently and in either order, so each half records
what it has and the upload waits for both. A call that produced no audio, or
one Tuner issued no target for, logs a warning and uploads nothing.

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

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEPGRAM_API_KEY` | — | STT and TTS |
| `OPENAI_API_KEY` | — | LLM |
| `TUNER_API_KEY` / `TUNER_WORKSPACE_ID` / `TUNER_AGENT_ID` / `TUNER_BASE_URL` | `dev` / — / `pizza-order-bot` / `https://api.usetuner.ai` | Tuner observability |
| `TUNER_RECORD_CALLS` | `true` | Record the call and upload the audio to Tuner |
| `DEEPGRAM_VOICE` | `aura-2-thalia-en` | TTS voice |

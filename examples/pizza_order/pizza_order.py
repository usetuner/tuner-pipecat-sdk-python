#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Pizzeria ordering bot built with Pipecat and the Tuner Observer.

Also shows the call-recording hand-off: the SDK opens a presigned upload
target for the call, and this bot captures the audio and uploads it (see
``CallRecording`` below). Set ``TUNER_RECORD_CALLS=false`` to turn it off.

Requirements:
- DEEPGRAM_API_KEY
- OPENAI_API_KEY

Run the example:
uv run pizza_order.py
"""

import asyncio
import os
import uuid

import httpx
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from tuner_pipecat_sdk import CallUsage, Observer, to_wav

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

MENU = {
    "margherita": 10.99,
    "pepperoni": 12.99,
    "veggie": 11.99,
    "bbq chicken": 13.99,
}

SIZE_SURCHARGE = {"small": 0.0, "medium": 2.0, "large": 4.0}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

AGENT_INSTRUCTIONS = f"""
You are a friendly cashier at 'Pipecat Pizza'.
Keep responses short and conversational. This is a voice call — never use
emojis, formatting, bullet points, or special characters in your responses.

## Today's menu
{chr(10).join(f"- {name} (${price:.2f})" for name, price in MENU.items())}

## Sizes
- small (no extra charge)
- medium (+$2.00)
- large (+$4.00)

## How to take an order
1. Greet the customer warmly and present today's menu, then ask which pizza
   they would like.
2. Once they pick a pizza, call the choose_pizza function with their choice.
3. Ask what size they want, then call the choose_size function.
4. Read back the full order — pizza, size, and total price — and ask the
   customer to confirm. Call the confirm_order function with their answer.
5. If confirmed, thank them enthusiastically and tell them their order is being
   prepared. If not confirmed, apologise politely and tell them they can call
   back anytime.
6. After your final closing line, call the end_call function to end the call.

## Important rules
- Only accept pizzas from the menu above.
- Never confirm an order without calling confirm_order first.
- Always read back the pizza, size, and total before asking to confirm.
"""


# ---------------------------------------------------------------------------
# Debug logging processor
# ---------------------------------------------------------------------------


class DebugLogProcessor(FrameProcessor):
    """Logs transcriptions and LLM response text as they flow through the pipeline."""

    def __init__(self):
        super().__init__()
        self._bot_response_buf = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[TRANSCRIPT] User said: '{frame.text}'")

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._bot_response_buf = []

        elif isinstance(frame, TextFrame) and self._bot_response_buf is not None:
            self._bot_response_buf.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._bot_response_buf:
                full = "".join(self._bot_response_buf)
                logger.info(f"[BOT RESPONSE] {full}")
            self._bot_response_buf = []

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Call recording upload
# ---------------------------------------------------------------------------

# The flush that issues the target POSTs the call first, so allow for that
# round trip; the audio is already in memory by then and only needs handing
# over; the upload itself is a multi-megabyte POST.
TARGET_WAIT_SECONDS = 20.0
AUDIO_WAIT_SECONDS = 10.0
UPLOAD_TIMEOUT_SECONDS = 60.0


class CallRecording:
    """Captures the call's audio and POSTs it to Tuner's presigned target.

    The SDK never uploads a recording. Passing ``recording_file_extension`` to
    the Observer makes the Tuner API issue a presigned S3 POST for the call and
    hand it to ``on_recording_upload``; the upload is ours to make, so it stays
    off the call's teardown path and can be retried, deferred or skipped.

    The two halves arrive independently and in no fixed order — pipecat
    dispatches ``on_audio_data`` as its own task when the pipeline ends, and
    the observer emits the target from the flush task it starts on the same
    frame — so each half only records what it has, and the upload runs after
    the pipeline, in :meth:`wait_for_upload`.
    """

    def __init__(self) -> None:
        self._wav: bytes | None = None
        self._target: dict | None = None
        self._audio_ready = asyncio.Event()
        self._target_ready = asyncio.Event()

    def attach(self, observer: Observer, audio_buffer) -> None:
        """Register both halves of the hand-off."""

        @audio_buffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            # `audio` is raw PCM with no RIFF header. to_wav() is what makes
            # the uploaded file playable — posting these bytes as-is to a
            # target opened for "wav" succeeds and produces a file that no
            # player will open.
            self._wav = to_wav(audio, sample_rate, num_channels)
            logger.info(
                f"[RECORDING] captured {len(self._wav)} bytes ({sample_rate} Hz, {num_channels}ch)"
            )
            self._audio_ready.set()

        @observer.event_handler("on_recording_upload")
        async def on_recording_upload(observer, target):
            # Awaited inline inside the observer's flush: take the target and
            # return immediately, so the upload never runs inside a task
            # nobody awaits, where the process exiting would truncate it.
            logger.info(f"[RECORDING] upload target issued (key={target.get('key', '?')})")
            self._target = target
            self._target_ready.set()

    async def wait_for_upload(self) -> None:
        """Upload the recording, once both halves have arrived.

        Call this after ``runner.run(task)``. That returning does not mean the
        call has been submitted — the observer flushes in a background task,
        and the target only exists once that POST comes back — so returning
        from the bot any earlier would drop the recording on the floor.
        """
        try:
            await asyncio.wait_for(self._target_ready.wait(), timeout=TARGET_WAIT_SECONDS)
        except TimeoutError:
            logger.warning("[RECORDING] no upload target arrived — nothing to upload")
            return

        try:
            await asyncio.wait_for(self._audio_ready.wait(), timeout=AUDIO_WAIT_SECONDS)
        except TimeoutError:
            logger.warning("[RECORDING] the call produced no audio — skipping upload")
            return

        await self._upload(self._target, self._wav or b"")

    async def _upload(self, target: dict, wav: bytes) -> None:
        max_bytes = target.get("max_bytes")
        if max_bytes and len(wav) > max_bytes:
            logger.error(
                f"[RECORDING] {len(wav)} bytes exceeds the target's "
                f"{max_bytes}-byte limit — skipping upload"
            )
            return

        # A presigned S3 POST: the policy fields go in the form, the file last.
        # The target expires (the deadline is inside the base64 `policy`), so
        # anything that queues it for later has to upload inside that window.
        key = target.get("key") or "recording.wav"
        files = {
            "file": (
                os.path.basename(key),
                wav,
                target.get("content_type") or "audio/wav",
            )
        }
        try:
            async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT_SECONDS) as client:
                response = await client.post(target["url"], data=target["fields"], files=files)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(f"[RECORDING] upload failed: {exc}")
            return

        logger.info(f"[RECORDING] uploaded {len(wav)} bytes to {key}")


# ---------------------------------------------------------------------------
# Order state + tools
# ---------------------------------------------------------------------------

# Single-call demo: keep the in-progress order in module state.
_order: dict = {}


async def choose_pizza(params):
    """Record which pizza the customer wants to order."""
    pizza = params.arguments.get("pizza", "").lower()
    price = MENU.get(pizza, 0.0)
    logger.info(f"[ORDER] Pizza chosen: {pizza} (${price:.2f})")
    _order["pizza"] = pizza
    _order["price"] = price
    await params.result_callback({"pizza": pizza, "price": price})


async def choose_size(params):
    """Record the pizza size the customer wants."""
    size = params.arguments.get("size", "").lower()
    surcharge = SIZE_SURCHARGE.get(size, 0.0)
    total = _order.get("price", 0.0) + surcharge
    logger.info(f"[ORDER] Size chosen: {size} | total=${total:.2f}")
    _order["size"] = size
    _order["total"] = total
    await params.result_callback({"size": size, "total": total})


async def confirm_order(params):
    """Confirm or cancel the order."""
    confirmed = bool(params.arguments.get("confirmed"))
    logger.info(f"[ORDER] Confirmed: {confirmed} | order={_order}")
    await params.result_callback({"confirmed": confirmed})


async def end_call(params):
    logger.info("[END CALL] Reason: agent_hangup")
    await params.result_callback({"status": "ending", "reason": "agent_hangup"})
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


def build_tools() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="choose_pizza",
                description="Record which pizza the customer wants to order.",
                properties={"pizza": {"type": "string", "enum": list(MENU.keys())}},
                required=["pizza"],
            ),
            FunctionSchema(
                name="choose_size",
                description="Record the pizza size the customer wants.",
                properties={"size": {"type": "string", "enum": list(SIZE_SURCHARGE.keys())}},
                required=["size"],
            ),
            FunctionSchema(
                name="confirm_order",
                description="Confirm or cancel the order.",
                properties={"confirmed": {"type": "boolean"}},
                required=["confirmed"],
            ),
            FunctionSchema(
                name="end_call",
                description=(
                    "End the call once you have delivered your final spoken response "
                    "to the customer."
                ),
                properties={},
                required=[],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice=os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en"),
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(system_instruction=AGENT_INSTRUCTIONS),
    )

    llm.register_function("choose_pizza", choose_pizza)
    llm.register_function("choose_size", choose_size)
    llm.register_function("confirm_order", confirm_order)
    llm.register_function("end_call", end_call)

    context = LLMContext(tools=build_tools())
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    debug_logger = DebugLogProcessor()
    turn_tracker = TurnTrackingObserver()

    record_calls = os.getenv("TUNER_RECORD_CALLS", "true").lower() in ("1", "true", "yes")

    def calculate_cost(usage: CallUsage) -> float:
        # OpenAI gpt-4o-mini pricing (per token)
        llm_cost = (usage.llm_prompt_tokens or 0) * 0.000_000_150
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_000_600
        # Deepgram Aura-2 TTS: $0.030 per 1K characters
        tts_cost = (usage.tts_characters or 0) * 0.000_030
        # Deepgram Nova-3 STT: $0.0043 per audio minute
        stt_cost = usage.stt_audio_seconds * (0.0043 / 60)
        return (llm_cost + tts_cost + stt_cost) * 100

    observer = Observer(
        api_key=os.getenv("TUNER_API_KEY", "dev"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID")),
        agent_id=os.getenv("TUNER_AGENT_ID", "pizzeria-bot"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "https://api.usetuner.ai"),
        asr_model=os.getenv("TUNER_ASR_MODEL", "deepgram/nova-3"),
        llm_model=os.getenv("TUNER_LLM_MODEL", "gpt-4o-mini"),
        tts_model=os.getenv("TUNER_TTS_MODEL", "deepgram/aura-2-thalia-en"),
        cost_calculator=calculate_cost,
        # Opts the call into a presigned upload target. The SDK stops there —
        # CallRecording does the uploading.
        recording_file_extension="wav" if record_calls else None,
        debug=True,
    )
    observer.attach_turn_tracking_observer(turn_tracker)

    # None unless recording is on. The SDK owns the buffer's configuration
    # (two channels, caller on the left, nothing emitted until the call ends),
    # which is what lets it describe the layout to Tuner. It does not start
    # the recorder — on_client_connected does that below.
    audio_buffer = observer.audio_buffer()
    recording = CallRecording() if audio_buffer else None
    if recording:
        recording.attach(observer, audio_buffer)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            debug_logger,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            # After transport.output(): frames travel downstream past it in
            # both directions, so one processor captures the caller's audio
            # and the agent's.
            *([audio_buffer] if audio_buffer else []),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[observer, observer.latency_observer, turn_tracker],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("[BOT] Client connected — starting pizzeria flow")
        _order.clear()
        if audio_buffer:
            # Pipecat's recorder starts idle and buffers nothing until this is
            # called. Without it the call ends with an empty buffer, so
            # on_audio_data never fires and there is nothing to upload.
            await audio_buffer.start_recording()
        context.add_message(
            {
                "role": "assistant",
                "content": (
                    "Greet the customer warmly, present the menu, "
                    "and ask which pizza they would like."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("[BOT] Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)

    if recording:
        await recording.wait_for_upload()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

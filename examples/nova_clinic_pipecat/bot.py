"""
Nova Clinic Voice Assistant — Pipecat + Tuner Demo
"""

import json
import os
import random
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, LLMRunFrame
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from tuner_pipecat_sdk import CallUsage, Observer

load_dotenv()

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# TUNER_TEST_STRATEGY = "vad_only" | "stt_endpointing" | "smart_turn"
_STRATEGY = os.getenv("TUNER_TEST_STRATEGY", "smart_turn")

def _build_user_turn_strategies() -> UserTurnStrategies:
    if _STRATEGY == "vad_only":
        stop_strategy = SpeechTimeoutUserTurnStopStrategy(wait_for_transcript=False)

    elif _STRATEGY == "stt_endpointing":
        stop_strategy = SpeechTimeoutUserTurnStopStrategy(wait_for_transcript=True)
    else:
        stop_strategy = TurnAnalyzerUserTurnStopStrategy(
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
            wait_for_transcript=True,
        )

    logger.info(
        "[tuner-test] stop strategy resolved: {} wait_for_transcript={} user_speech_timeout={}",
        type(stop_strategy).__name__,
        getattr(stop_strategy, "_wait_for_transcript", "n/a"),
        getattr(stop_strategy, "_user_speech_timeout", "n/a"),
    )

    return UserTurnStrategies(
        start=[VADUserTurnStartStrategy()],
        stop=[stop_strategy],
    )

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

AGENT_INSTRUCTIONS = """
You are Aria, a friendly and professional voice assistant for Nova Clinic.
You help patients book, reschedule, and cancel appointments over the phone.

## Your personality
- Warm, calm, and professional — like a real front desk receptionist
- Concise — this is a voice call, keep responses short and clear
- Never use formatting, bullet points, asterisks, or emojis in your responses
- Always confirm details back to the patient before taking action

## What you can do
- Check available appointment slots
- Book new appointments
- Look up existing appointments
- Cancel appointments

## Available doctors
- Dr. Sarah Patel — General Practice
- Dr. James Lee — General Practice

## How to handle a booking request
1. Ask for the patient's full name if not provided
2. Ask what the appointment is for (brief reason)
3. Ask for their preferred date
4. Check availability using the check_availability function
5. Offer available slots and confirm their choice
6. Book the appointment using the book_appointment function
7. If booking fails due to a conflict, apologize and offer the next available slot
8. Confirm the final booking details clearly

## How to handle a cancellation
1. Ask for the patient's full name
2. Look up their existing appointment using the get_appointment function
3. Confirm they want to cancel
4. Use cancel_appointment function

## Important rules
- Never make up availability — always use the check_availability function
- Never confirm a booking without calling the book_appointment function
- If a function fails, tell the patient politely and offer an alternative
- Always read back the confirmed appointment details at the end

## When to end the call
- Once you have completed the patient's request and read back all confirmed details, ask if there is anything else you can help with
- If the patient says no or says goodbye, say a warm closing like "Have a great day, goodbye!" and immediately call the end_call function
- Do not wait or ask further questions after the patient has said goodbye
"""


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_data.json")

with open(_DATA_PATH) as f:
    _DB = json.load(f)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def check_availability(params):
    """Check available appointment slots for a doctor on a given date."""
    date = params.arguments.get("date")
    doctor_name = params.arguments.get("doctor_name")
    logger.info(f"Checking availability for {doctor_name} on {date}")

    doctor_id = None
    for doc in _DB["doctors"]:
        if doc["name"].lower() == doctor_name.lower():
            doctor_id = doc["id"]
            break

    logger.info(f"Found doctor_id: {doctor_id}")

    if doctor_id:
        logger.info(
            f"Dates available for {doctor_id}: {list(_DB['availability'].get(doctor_id, {}).keys())}"
        )

    if not doctor_id:
        result = {
            "success": False,
            "error": f"Doctor '{doctor_name}' not found. Available doctors are: "
            + ", ".join(d["name"] for d in _DB["doctors"]),
        }
        await params.result_callback(result)
        return

    slots = _DB["availability"].get(doctor_id, {}).get(date, [])

    if not slots:
        result = {
            "success": True,
            "available": False,
            "message": f"No availability for {doctor_name} on {date}. Try another date or doctor.",
        }
        await params.result_callback(result)
        return

    result = {
        "success": True,
        "available": True,
        "doctor": doctor_name,
        "date": date,
        "slots": slots,
    }
    await params.result_callback(result)


async def book_appointment(params):
    """Book an appointment for a patient."""
    patient_name = params.arguments.get("patient_name")
    date = params.arguments.get("date")
    time = params.arguments.get("time")
    doctor_name = params.arguments.get("doctor_name")
    reason = params.arguments.get("reason")

    logger.info(f"Booking appointment for {patient_name} with {doctor_name} on {date} at {time}")

    # Designed-in failure: 40% chance of slot conflict
    if random.random() < 0.4:
        logger.warning("Slot conflict triggered (designed-in failure)")
        result = {
            "success": False,
            "error": "slot_conflict",
            "message": f"Sorry, the {time} slot with {doctor_name} on {date} was just taken. Please offer the next available slot.",
        }
        await params.result_callback(result)
        return

    appointment_id = f"APT-{random.randint(1000, 9999)}"

    result = {
        "success": True,
        "appointment_id": appointment_id,
        "patient_name": patient_name,
        "doctor": doctor_name,
        "date": date,
        "time": time,
        "reason": reason,
        "message": f"Appointment confirmed. Booking reference: {appointment_id}.",
    }
    await params.result_callback(result)


async def get_appointment(params):
    """Look up an existing appointment for a patient."""
    patient_name = params.arguments.get("patient_name")
    logger.info(f"Looking up appointment for {patient_name}")

    appointment = _DB["existing_appointments"].get(patient_name.lower())

    if not appointment:
        result = {
            "success": False,
            "error": "not_found",
            "message": f"No existing appointment found for {patient_name}.",
        }
        await params.result_callback(result)
        return

    result = {"success": True, "appointment": appointment}
    await params.result_callback(result)


async def cancel_appointment(params):
    """Cancel an existing appointment."""
    appointment_id = params.arguments.get("appointment_id")
    logger.info(f"Cancelling appointment {appointment_id}")

    result = {
        "success": True,
        "appointment_id": appointment_id,
        "message": f"Appointment {appointment_id} has been successfully cancelled.",
    }
    await params.result_callback(result)


async def end_call(params):
    logger.info("[END CALL] Reason: agent_hangup")
    await params.result_callback({"status": "ending", "reason": "agent_hangup"})
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    sip_call_data: dict | None = None,
    sip_provider: str | None = None,
    sip_context: object | None = None,
):
    logger.info("Starting Nova Clinic assistant")

    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    #stt = DeepgramFluxSTTService(
    #    api_key=os.getenv("DEEPGRAM_API_KEY"),
    #    settings=DeepgramFluxSTTService.Settings(
    #        eot_threshold=0.7,      # end-of-turn confidence needed to fire END_OF_TURN
    #        eot_timeout_ms=3000,
    #    ),
    #)
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), voice="alloy")

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="check_availability",
                description="Check available appointment slots for a doctor on a given date",
                properties={
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "doctor_name": {"type": "string", "description": "Doctor's full name"},
                },
                required=["date", "doctor_name"],
            ),
            FunctionSchema(
                name="book_appointment",
                description="Book an appointment for a patient",
                properties={
                    "patient_name": {"type": "string", "description": "Patient's full name"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Time in HH:MM format"},
                    "doctor_name": {"type": "string", "description": "Doctor's full name"},
                    "reason": {"type": "string", "description": "Reason for visit"},
                },
                required=["patient_name", "date", "time", "doctor_name", "reason"],
            ),
            FunctionSchema(
                name="get_appointment",
                description="Look up an existing appointment for a patient",
                properties={
                    "patient_name": {"type": "string", "description": "Patient's full name"},
                },
                required=["patient_name"],
            ),
            FunctionSchema(
                name="cancel_appointment",
                description="Cancel an existing appointment",
                properties={
                    "appointment_id": {"type": "string", "description": "Appointment ID"},
                },
                required=["appointment_id"],
            ),
            FunctionSchema(
                name="end_call",
                description=(
                    "End the call once you have fully served the user "
                    "and delivered your final spoken response."
                ),
                properties={},
                required=[],
            ),
        ]
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            system_instruction=AGENT_INSTRUCTIONS,
        ),
    )

    llm.register_function("check_availability", check_availability)
    llm.register_function("book_appointment", book_appointment)
    llm.register_function("get_appointment", get_appointment)
    llm.register_function("cancel_appointment", cancel_appointment)
    llm.register_function("end_call", end_call)

    context = LLMContext(tools=tools)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            #user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_strategies=_build_user_turn_strategies(),
        ),
    )


    def calculate_cost(usage: CallUsage) -> float:
        # OpenAI gpt-4o-mini pricing (per token / per character / per second)
        llm_cost  = (usage.llm_prompt_tokens     or 0) * 0.000_000_150
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_000_600
        # OpenAI TTS-1: $15 per 1M characters
        tts_cost  = (usage.tts_characters        or 0) * 0.000_015
        # OpenAI gpt-4o-transcribe: $6 per 1M audio seconds
        stt_cost  = usage.stt_audio_seconds            * 0.000_006
        return (llm_cost + tts_cost + stt_cost) * 100

    turn_tracker = TurnTrackingObserver()
    observer = Observer(
        api_key=os.getenv("TUNER_API_KEY"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID", "0")),
        agent_id=os.getenv("TUNER_AGENT_ID", "nova-clinic-pipecat"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "https://api.usetuner.ai"),
        asr_model="openai/gpt-4o-transcribe",
        llm_model="gpt-5-mini-2025-08-07",
        tts_model="openai/tts-1",
        cost_calculator=calculate_cost,
        debug=True,
    )
    observer.attach_turn_tracking_observer(turn_tracker)
    observer.attach_user_aggregator(user_aggregator)

    # Forward SIP-layer Call-ID + headers when the call arrived over SIP.
    # Preferred path: a typed provider context (e.g. JambonzCallContext) the
    # server built and handed in. Legacy path: ``sip_call_data + sip_provider``
    # for providers without a typed context yet. See README
    # "SIP / Telephony Calls".
    if sip_context is not None:
        observer.attach_sip_from_context(sip_context)
    elif sip_call_data is not None and sip_provider is not None:
        observer.attach_sip_from_telephony(sip_call_data, provider=sip_provider)
    elif isinstance(getattr(runner_args, "body", None), dict):
        dialin = runner_args.body.get("dialin_settings")
        if dialin:
            observer.attach_sip_from_dialin(dialin)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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
        logger.info("Client connected")
        context.add_message(
            {
                "role": "assistant",
                "content": "Greet the caller warmly and ask how you can help them today.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Pipecat dev-runner entrypoint — used for non-telephony transports (webrtc).

    Telephony providers (Twilio, Telnyx, …) have their own server scripts that
    build the transport and call ``run_bot`` directly. Keeping this entrypoint
    means ``python bot.py -t webrtc`` still works for local development.
    """
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

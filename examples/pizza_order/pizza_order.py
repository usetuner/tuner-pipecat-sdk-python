#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Pizzeria ordering bot built with Pipecat Flows.

Requirements:
- DEEPGRAM_API_KEY
- OPENAI_API_KEY

Run the example:
uv run pizza_order.py
"""

import json
import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
)
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
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.transports.base_transport import BaseTransport, TransportParams

from openai import AsyncOpenAI

from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)
from pipecat_flows_tuner import FlowsObserver

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# Debug logging processor — sits in the pipeline to intercept live frames
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
# Helpers
# ---------------------------------------------------------------------------

MENU = {
    "margherita": 10.99,
    "pepperoni": 12.99,
    "veggie": 11.99,
    "bbq chicken": 13.99,
}

TOPPINGS = {
    "mushrooms": 1.50,
    "extra cheese": 1.50,
    "jalapeños": 1.25,
    "peppers": 1.00,
    "olives": 1.00,
    "onions": 0.75,
}

# Zip codes we deliver to, mapped to zone label (used for ETA estimation)
DELIVERY_ZONES: dict[str, str] = {
    "10001": "zone-a",  # Manhattan core — 20 min
    "10002": "zone-a",
    "10003": "zone-a",
    "10011": "zone-b",  # West Village / Chelsea — 30 min
    "10014": "zone-b",
    "10036": "zone-b",
    "10019": "zone-b",
    "11201": "zone-c",  # Brooklyn — 45 min
    "11211": "zone-c",
    "11222": "zone-c",
}

ZONE_ETA: dict[str, str] = {
    "zone-a": "approximately 20 minutes",
    "zone-b": "approximately 30 minutes",
    "zone-c": "approximately 45 minutes",
}

# Time slots available for pre-scheduled orders (every 30 min, next 4 hours)
AVAILABLE_SLOTS = [
    "12:00", "12:30", "13:00", "13:30", "14:00",
    "14:30", "15:00", "15:30", "16:00", "16:30",
    "17:00", "17:30", "18:00", "18:30", "19:00",
    "19:30", "20:00", "20:30", "21:00", "21:30",
]


def log_node(node: dict, label: str) -> None:
    logger.debug(f"[NODE CREATED] {label}:\n{json.dumps(node, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Flow nodes
# ---------------------------------------------------------------------------


def create_greeting_node() -> NodeConfig:
    choose_pizza_func = FlowsFunctionSchema(
        name="choose_pizza",
        description="Record which pizza the customer wants to order.",
        required=["pizza"],
        properties={"pizza": {"type": "string", "enum": list(MENU.keys())}},
        handler=handle_choose_pizza,
    )

    node = {
        "name": "greeting",
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly pizzeria cashier at 'Pipecat Pizza'. "
                    "Keep responses short and conversational. "
                    "Your responses will be converted to audio — no emojis or special characters."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Greet the customer warmly and present today's menu: "
                    f"{', '.join(f'{k} (${v:.2f})' for k, v in MENU.items())}. "
                    "Ask them which pizza they'd like."
                ),
            }
        ],
        "functions": [choose_pizza_func],
    }
    log_node(node, "greeting")
    return node


async def handle_choose_pizza(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    pizza = args["pizza"].lower()
    price = MENU.get(pizza, 0.0)
    logger.info(f"[ORDER] Pizza chosen: {pizza} (${price:.2f})")
    flow_manager.state["pizza"] = pizza
    flow_manager.state["price"] = price
    return {"pizza": pizza, "price": price}, create_size_node()


def create_size_node() -> NodeConfig:
    choose_size_func = FlowsFunctionSchema(
        name="choose_size",
        description="Record the pizza size the customer wants.",
        required=["size"],
        properties={"size": {"type": "string", "enum": ["small", "medium", "large"]}},
        handler=handle_choose_size,
    )

    node = NodeConfig(
        name="size",
        task_messages=[
            {
                "role": "system",
                "content": "Ask the customer what size they want: small ($0 extra), medium (+$2), or large (+$4).",
            }
        ],
        functions=[choose_size_func],
    )
    log_node(dict(node), "size")
    return node


async def handle_choose_size(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    size = args["size"].lower()
    surcharge = {"small": 0.0, "medium": 2.0, "large": 4.0}.get(size, 0.0)
    total = flow_manager.state["price"] + surcharge
    logger.info(f"[ORDER] Size chosen: {size} | total=${total:.2f}")
    flow_manager.state["size"] = size
    flow_manager.state["total"] = total
    flow_manager.state["toppings"] = []
    return {"size": size, "total": total}, create_ai_pairing_node()


# ---------------------------------------------------------------------------
# AI drink pairing node — makes a real async OpenAI call to verify timing
# ---------------------------------------------------------------------------


def create_ai_pairing_node() -> NodeConfig:
    suggest_func = FlowsFunctionSchema(
        name="suggest_drink_pairing",
        description="Fetch an AI-generated drink pairing suggestion for the chosen pizza.",
        properties={},
        required=[],
        handler=handle_suggest_drink_pairing,
    )
    node = NodeConfig(
        name="ai_pairing",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Call `suggest_drink_pairing` immediately to fetch a personalized drink "
                    "recommendation for the customer's pizza. Then share it conversationally "
                    "before moving on to toppings."
                ),
            }
        ],
        functions=[suggest_func],
    )
    log_node(dict(node), "ai_pairing")
    return node


async def handle_suggest_drink_pairing(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    pizza = flow_manager.state.get("pizza", "pizza")
    size = flow_manager.state.get("size", "")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Suggest a drink that pairs well with a {size} {pizza} pizza. "
                    "One short sentence only."
                ),
            }
        ],
        max_tokens=40,
    )
    suggestion = response.choices[0].message.content.strip()
    logger.info(f"[AI PAIRING] {suggestion}")
    flow_manager.state["drink_suggestion"] = suggestion
    return {"drink_pairing": suggestion}, create_toppings_node()


# ---------------------------------------------------------------------------
# Toppings node — multiple function calls: add_topping (stays) + finish_toppings
# ---------------------------------------------------------------------------


def create_toppings_node() -> NodeConfig:
    topping_list = ", ".join(f"{k} (+${v:.2f})" for k, v in TOPPINGS.items())

    add_topping_func = FlowsFunctionSchema(
        name="add_topping",
        description="Add one extra topping to the pizza. Can be called multiple times.",
        required=["topping"],
        properties={"topping": {"type": "string", "enum": list(TOPPINGS.keys())}},
        handler=handle_add_topping,
    )
    finish_toppings_func = FlowsFunctionSchema(
        name="finish_toppings",
        description="Customer is done selecting toppings (even if none were added). Proceed to delivery options.",
        properties={},
        required=[],
        handler=handle_finish_toppings,
    )

    node = NodeConfig(
        name="toppings",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Ask the customer if they'd like any extra toppings. Available toppings: {topping_list}. "
                    "They can add multiple toppings by calling add_topping for each one. "
                    "When they're done (or if they don't want any), call finish_toppings."
                ),
            }
        ],
        functions=[add_topping_func, finish_toppings_func],
    )
    log_node(dict(node), "toppings")
    return node


async def handle_add_topping(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    topping = args["topping"].lower()
    price = TOPPINGS.get(topping, 0.0)
    toppings = flow_manager.state.setdefault("toppings", [])
    if topping not in toppings:
        toppings.append(topping)
        flow_manager.state["total"] = flow_manager.state.get("total", 0.0) + price
    logger.info(f"[ORDER] Topping added: {topping} (+${price:.2f}) | toppings={toppings}")
    # Return None as next node to stay on the same node
    return {"topping": topping, "toppings": toppings, "new_total": flow_manager.state["total"]}, None


async def handle_finish_toppings(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    toppings = flow_manager.state.get("toppings", [])
    total = flow_manager.state.get("total", 0.0)
    logger.info(f"[ORDER] Toppings finalised: {toppings} | total=${total:.2f}")
    return {"toppings": toppings, "total": total}, create_delivery_node()


# ---------------------------------------------------------------------------
# Delivery / pickup branching node
# ---------------------------------------------------------------------------


def create_delivery_node() -> NodeConfig:
    pickup_func = FlowsFunctionSchema(
        name="choose_pickup",
        description="Customer wants to pick up the order in-store.",
        properties={},
        required=[],
        handler=handle_choose_pickup,
    )
    delivery_func = FlowsFunctionSchema(
        name="choose_delivery",
        description="Customer wants the order delivered to their address.",
        properties={},
        required=[],
        handler=handle_choose_delivery,
    )

    node = NodeConfig(
        name="delivery_or_pickup",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the customer whether they'd like to pick up their order in-store (free) "
                    "or have it delivered (flat $3 delivery fee). "
                    "Call choose_pickup or choose_delivery based on their answer."
                ),
            }
        ],
        functions=[pickup_func, delivery_func],
    )
    log_node(dict(node), "delivery_or_pickup")
    return node


async def handle_choose_pickup(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    logger.info("[ORDER] Fulfilment: pickup")
    flow_manager.state["fulfilment"] = "pickup"
    return {"fulfilment": "pickup"}, create_schedule_node()


async def handle_choose_delivery(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    fee = 3.0
    flow_manager.state["fulfilment"] = "delivery"
    flow_manager.state["total"] = flow_manager.state.get("total", 0.0) + fee
    logger.info(f"[ORDER] Fulfilment: delivery | +${fee:.2f} fee | total=${flow_manager.state['total']:.2f}")
    return {"fulfilment": "delivery", "delivery_fee": fee}, create_address_node()


# ---------------------------------------------------------------------------
# Address collection node
# ---------------------------------------------------------------------------


def create_address_node(error: str | None = None) -> NodeConfig:
    set_address_func = FlowsFunctionSchema(
        name="set_delivery_address",
        description="Record the customer's delivery address.",
        required=["street", "city", "zip_code"],
        properties={
            "street": {"type": "string", "description": "Street name and number"},
            "city": {"type": "string", "description": "City name"},
            "zip_code": {"type": "string", "description": "Postal/ZIP code"},
        },
        handler=handle_set_address,
    )

    if error:
        content = (
            f"{error} Please ask the customer for a different delivery address "
            "and collect the street, city, and zip code again."
        )
    else:
        content = (
            "Ask the customer for their delivery address. "
            "Collect the street, city, and zip code, then call set_delivery_address."
        )

    node = NodeConfig(
        name="address",
        task_messages=[{"role": "system", "content": content}],
        functions=[set_address_func],
    )
    log_node(dict(node), "address")
    return node


async def handle_set_address(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    street = args["street"]
    city = args["city"]
    zip_code = args["zip_code"].strip()
    zone = DELIVERY_ZONES.get(zip_code)

    if zone is None:
        logger.warning(f"[ORDER] Zip {zip_code!r} outside delivery zone — staying on address node")
        error_msg = (
            f"Sorry, we don't deliver to zip code {zip_code}. "
            f"We currently serve: {', '.join(sorted(DELIVERY_ZONES.keys()))}."
        )
        return {"error": "out_of_zone", "zip_code": zip_code}, create_address_node(error=error_msg)

    address = {"street": street, "city": city, "zip_code": zip_code}
    eta = ZONE_ETA[zone]
    flow_manager.state["address"] = address
    flow_manager.state["delivery_zone"] = zone
    flow_manager.state["eta"] = eta
    logger.info(f"[ORDER] Delivery address: {address} | zone={zone} | ETA={eta}")
    return {"address": address, "zone": zone, "eta": eta}, create_schedule_node()


# ---------------------------------------------------------------------------
# Schedule node — ASAP or pick a time slot
# ---------------------------------------------------------------------------


def create_schedule_node() -> NodeConfig:
    slots_str = ", ".join(AVAILABLE_SLOTS)

    asap_func = FlowsFunctionSchema(
        name="order_asap",
        description="Customer wants the order as soon as possible.",
        properties={},
        required=[],
        handler=handle_order_asap,
    )
    schedule_func = FlowsFunctionSchema(
        name="schedule_order",
        description="Customer wants to pre-schedule the order for a specific time slot.",
        required=["time_slot"],
        properties={
            "time_slot": {
                "type": "string",
                "enum": AVAILABLE_SLOTS,
                "description": "Time slot in HH:MM format (24-hour)",
            }
        },
        handler=handle_schedule_order,
    )

    node = NodeConfig(
        name="schedule",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the customer when they'd like their order: as soon as possible, "
                    f"or scheduled for a specific time. Available slots today: {slots_str}. "
                    "Call order_asap for ASAP, or schedule_order with the chosen time slot."
                ),
            }
        ],
        functions=[asap_func, schedule_func],
    )
    log_node(dict(node), "schedule")
    return node


async def handle_order_asap(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    logger.info("[ORDER] Timing: ASAP")
    flow_manager.state["scheduled_time"] = "ASAP"
    return {"scheduled_time": "ASAP"}, create_confirm_node()


async def handle_schedule_order(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    slot = args["time_slot"]
    logger.info(f"[ORDER] Timing: scheduled for {slot}")
    flow_manager.state["scheduled_time"] = slot
    return {"scheduled_time": slot}, create_confirm_node()


def create_confirm_node() -> NodeConfig:
    confirm_func = FlowsFunctionSchema(
        name="confirm_order",
        description="Confirm or cancel the order.",
        required=["confirmed"],
        properties={"confirmed": {"type": "boolean"}},
        handler=handle_confirm,
    )

    node = NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Read back the full order summary to the customer and ask them to confirm. "
                    "Include: pizza name, size, any extra toppings, fulfilment method (pickup or delivery address), "
                    "scheduled time (or ASAP), and the total price. Then ask for confirmation."
                ),
            }
        ],
        functions=[confirm_func],
    )
    log_node(dict(node), "confirm")
    return node


async def handle_confirm(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    confirmed = args["confirmed"]
    logger.info(f"[ORDER] Confirmed: {confirmed} | state={flow_manager.state}")
    if confirmed:
        return {"confirmed": True}, create_farewell_node(confirmed=True)
    else:
        return {"confirmed": False}, create_farewell_node(confirmed=False)


def create_farewell_node(confirmed: bool = True) -> NodeConfig:
    if confirmed:
        content = (
            "Thank the customer enthusiastically, tell them their order is being prepared, "
            "and wish them a great meal. Then end the conversation."
        )
    else:
        content = "Apologise politely, tell them they can call back anytime, and end the conversation."

    node = NodeConfig(
        name="farewell",
        task_messages=[{"role": "system", "content": content}],
        post_actions=[{"type": "end_conversation"}],
    )
    log_node(dict(node), "farewell")
    return node


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    debug_logger = DebugLogProcessor()
    turn_tracker = TurnTrackingObserver()

    observer = FlowsObserver(
        api_key=os.getenv("TUNER_API_KEY", "dev"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID")),
        agent_id=os.getenv("TUNER_AGENT_ID", "pizzeria-bot"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "http://localhost:8000"),
        asr_model=os.getenv("TUNER_ASR_MODEL", "deepgram/nova-3"),
        llm_model=os.getenv("TUNER_LLM_MODEL", "gpt-4o-mini"),
        tts_model=os.getenv("TUNER_TTS_MODEL", "deepgram/aura-2"),
        debug=True,
    )
    observer.attach_turn_tracking_observer(turn_tracker)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            debug_logger,                       # intercepts TranscriptionFrame + bot text
            context_aggregator.user(),
            llm,
            tts,
            observer,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            observers=[observer.latency_observer, turn_tracker],
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )
    observer.attach_flow_manager(flow_manager)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("[BOT] Client connected — starting pizzeria flow")
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("[BOT] Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Pizzeria ordering bot built with Pipecat, a raw LangChain tool-calling loop,
and the Tuner Observer's `wrap_chain()` LangChain bridge.

Same conversation flow and audio stack as ``examples/pizza_order`` -- the only
difference is the LLM step is a LangChain chat model with tools bound (via
pipecat's `LangchainProcessor`) instead of a native `OpenAILLMService`,
instrumented with `observer.wrap_chain()` so Tuner still captures tool calls
and LLM usage.

Note: LangChain 1.x removed `AgentExecutor`/`create_tool_calling_agent` from
`langchain.agents` (the only agent-building API left there, `create_agent`, is
itself LangGraph-based). This example instead builds the simplest possible
"raw" LangChain agent: a chat model with tools bound, orchestrated by hand in
a manual tool-calling loop -- exactly the shape `wrap_chain()` targets (tool
calls only, no graph nodes).

Requirements:
- DEEPGRAM_API_KEY
- OPENAI_API_KEY

Run the example:
uv run pizza_order_langchain.py
"""

import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
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
from pipecat.processors.frameworks.langchain import LangchainProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from tuner_pipecat_sdk import CallUsage, Observer

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


class EndCallWatcher(FrameProcessor):
    """Ends the call once the bot has finished voicing its final response.

    LangChain tools (unlike pipecat's ``llm.register_function()`` tools) have no
    ``FunctionCallParams`` to push an ``EndTaskFrame`` directly from inside the
    tool call. The ``end_call`` tool below just sets a flag; this processor
    watches for it and ends the call after the goodbye has been fully spoken.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame) and _call_state.get("should_end"):
            _call_state["should_end"] = False
            await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


# ---------------------------------------------------------------------------
# Order state + LangChain tools
# ---------------------------------------------------------------------------

# Single-call demo: keep the in-progress order in module state (same convention
# as examples/pizza_order/pizza_order.py).
_order: dict = {}
_call_state: dict = {"should_end": False}


@tool
def choose_pizza(pizza: str) -> dict:
    """Record which pizza the customer wants to order.

    Args:
        pizza: One of: margherita, pepperoni, veggie, bbq chicken.
    """
    pizza = pizza.lower()
    price = MENU.get(pizza, 0.0)
    logger.info(f"[ORDER] Pizza chosen: {pizza} (${price:.2f})")
    _order["pizza"] = pizza
    _order["price"] = price
    return {"pizza": pizza, "price": price}


@tool
def choose_size(size: str) -> dict:
    """Record the pizza size the customer wants.

    Args:
        size: One of: small, medium, large.
    """
    size = size.lower()
    surcharge = SIZE_SURCHARGE.get(size, 0.0)
    total = _order.get("price", 0.0) + surcharge
    logger.info(f"[ORDER] Size chosen: {size} | total=${total:.2f}")
    _order["size"] = size
    _order["total"] = total
    return {"size": size, "total": total}


@tool
def confirm_order(confirmed: bool) -> dict:
    """Confirm or cancel the order.

    Args:
        confirmed: True if the customer confirmed the order, False if they cancelled.
    """
    logger.info(f"[ORDER] Confirmed: {confirmed} | order={_order}")
    return {"confirmed": confirmed}


@tool
def end_call() -> dict:
    """Call this once you have delivered your final spoken response to the customer."""
    logger.info("[END CALL] Reason: agent_hangup")
    _call_state["should_end"] = True
    return {"status": "ending", "reason": "agent_hangup"}


TOOLS = [choose_pizza, choose_size, confirm_order, end_call]


# ---------------------------------------------------------------------------
# LangChain agent: a chat model with tools bound, orchestrated by hand
# ---------------------------------------------------------------------------

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Single-call demo: per-session message history, same module-level convention
# as ``_order``/``_call_state`` above.
_histories: dict[str, list[BaseMessage]] = {}


def _history_for(session_id: str) -> list[BaseMessage]:
    return _histories.setdefault(session_id, [])


def build_agent_chain() -> RunnableLambda:
    """Build the LangChain "chain" fed to ``observer.wrap_chain()``.

    Wrapped in a single ``RunnableLambda`` (rather than calling the model/tools
    as bare Python objects) so LangChain's callback manager establishes one
    root run for the whole turn -- every nested ``model.ainvoke()``/
    ``tool.ainvoke()`` call below is then correctly attributed as a child of
    that run, which is what lets tuner-langchain's callback handler resolve a
    root invocation for each tool call. Without this, each nested call would
    look like an unrelated, parent-less invocation and tool calls would
    silently go uncaptured.
    """
    model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")).bind_tools(TOOLS)

    async def _run_turn(input: dict, config: RunnableConfig) -> str:
        session_id = (config.get("configurable") or {}).get("session_id", "default")
        history = _history_for(session_id)
        history.append(HumanMessage(content=input["input"]))

        while True:
            messages = [SystemMessage(content=AGENT_INSTRUCTIONS), *history]
            response = await model.ainvoke(messages, config=config)
            history.append(response)

            if not response.tool_calls:
                return response.content

            for call in response.tool_calls:
                result = await TOOLS_BY_NAME[call["name"]].ainvoke(call["args"], config=config)
                history.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return RunnableLambda(_run_turn)


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice=os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en"),
    )

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
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID", "0")),
        agent_id=os.getenv("TUNER_AGENT_ID", "pizzeria-langchain-bot"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "https://api.usetuner.ai"),
        asr_model=os.getenv("TUNER_ASR_MODEL", "deepgram/nova-3"),
        llm_model=os.getenv("TUNER_LLM_MODEL", "gpt-4o-mini"),
        tts_model=os.getenv("TUNER_TTS_MODEL", "deepgram/aura-2-thalia-en"),
        cost_calculator=calculate_cost,
        debug=True,
    )
    turn_tracker = TurnTrackingObserver()
    observer.attach_turn_tracking_observer(turn_tracker)

    # Build the LangChain agent fresh per call, wrap it for Tuner observability
    # (tool calls + LLM usage), then hand the wrapped runnable to pipecat's
    # LangchainProcessor -- the drop-in replacement for a native LLMService.
    agent_chain = build_agent_chain()
    wrapped_agent = observer.wrap_chain(agent_chain)
    lc_processor = LangchainProcessor(chain=wrapped_agent, transcript_key="input")

    # LangchainProcessor sets config={"configurable": {"session_id": participant_id}}
    # on every call -- this is what the agent chain keys its per-call message
    # history on above, so multi-turn context works without extra wiring.
    lc_processor.set_participant_id("webrtc-demo-session")

    # No tools schema needed here -- pipecat doesn't need to know about the
    # tools, LangChain resolves and executes them internally.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    debug_logger = DebugLogProcessor()
    end_call_watcher = EndCallWatcher()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            debug_logger,
            context_aggregator.user(),
            lc_processor,
            tts,
            end_call_watcher,
            transport.output(),
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
        _call_state["should_end"] = False
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


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

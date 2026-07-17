"""
Nova Clinic Voice Assistant — Pipecat + LangGraph (multi-node) + Tuner Demo

Same conversation flow, mock data, and tools as ``examples/nova_clinic_pipecat``,
but the LLM step is a hand-built LangGraph ``StateGraph`` -- not the
``create_react_agent`` prebuilt -- so each intent gets its own system prompt:

    router  -> booking / cancellation / general  -> tools -> (back to caller node)

instrumented with ``observer.wrap_graph()`` so Tuner captures tool calls, node
transitions, and LLM usage. WebRTC only -- no telephony server variants.

## Why a router instead of one shared prompt

``create_react_agent`` gives every turn the exact same system prompt. That's
fine when one persona handles everything, but it means the "how to book" and
"how to cancel" instructions are always in context together, and there's no
way to give the booking flow a different tone/tool-set than the cancellation
flow. Splitting into nodes lets each flow's prompt stay focused on exactly
one job, which is closer to how production voice agents are actually built.

## How this avoids getting stuck or corrupting history

- ``messages`` uses LangGraph's ``add_messages`` reducer and a fixed
  ``thread_id`` checkpointer (exactly like the original example), so the full
  transcript -- across router decisions, flow switches, and tool calls -- is
  one single growing list. Nothing is duplicated or dropped when the active
  node changes.
- System prompts are injected *at call time* (``[SystemMessage(...)] +
  state["messages"]``) and are never appended back into the persisted
  ``messages`` list. If we *did* persist them, every flow switch would leave
  a stale system message sitting in history forever, and turn 10 would be
  reasoning with 4 contradictory system prompts stacked on top of each other.
  This is the most common bug in hand-rolled multi-prompt graphs.
- Every node has exactly one deterministic conditional edge: a flow node
  either produced a tool call (-> ``tools``) or a final answer (-> ``END``).
  There's no path that leaves the graph without an edge, so nothing can hang.
- The router is "sticky": it's told the current active flow and is
  instructed to keep it unless the flow just completed or the caller clearly
  asked for something else. Without this, a routing LLM call every turn can
  flip-flop on ambiguous input (e.g. "yes", "large") and mid-flow.
- ``recursion_limit`` is set explicitly on every graph invocation as a hard
  circuit breaker, on top of the logical guarantees above.
"""

import json
import os
import random
import uuid
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import BotStoppedSpeakingFrame, EndTaskFrame, Frame, LLMRunFrame
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
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pydantic import BaseModel, Field

from tuner_pipecat_sdk import CallUsage, Observer

load_dotenv()

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# Per-node prompts
# ---------------------------------------------------------------------------

PERSONA = """
You are Aria, a friendly and professional voice assistant for Nova Clinic.
- Warm, calm, and professional -- like a real front desk receptionist
- Concise -- this is a voice call, keep responses short and clear
- Never use formatting, bullet points, asterisks, or emojis in your responses
"""

GENERAL_PROMPT = (
    PERSONA
    + """
Right now you are handling general conversation: greetings, small talk, or
figuring out what the caller needs. You are NOT currently inside a booking or
cancellation flow.

- If this is the start of the call, greet the caller warmly and ask how you
  can help.
- If the caller's request is ambiguous, ask one short clarifying question.
- If the caller says goodbye or there's nothing else to help with, say a warm
  closing like "Have a great day, goodbye!" and immediately call end_call.
- Do not attempt to check availability, book, look up, or cancel anything
  yourself here -- that will be handled once your reply routes the caller
  into the right flow.
"""
)

BOOKING_PROMPT = (
    PERSONA
    + """
You are handling a NEW APPOINTMENT BOOKING for the caller. Available doctors:
- Dr. Sarah Patel -- General Practice
- Dr. James Lee -- General Practice

Steps:
1. Ask for the patient's full name if not already provided.
2. Ask what the appointment is for (brief reason).
3. Ask for their preferred date.
4. Call check_availability -- never make up availability yourself.
5. Offer the available slots and confirm their choice.
6. Call book_appointment to confirm the booking. Never confirm a booking
   without calling this function.
7. If booking fails due to a conflict, apologize and offer the next
   available slot, then try again.
8. Once booked, clearly read back the confirmed appointment details, then
   ask if there's anything else you can help with.
9. If the caller says goodbye at any point, say a warm closing and
   immediately call end_call.
"""
)

CANCELLATION_PROMPT = (
    PERSONA
    + """
You are handling an APPOINTMENT CANCELLATION for the caller.

Steps:
1. Ask for the patient's full name if not already provided.
2. Call get_appointment to look up their existing appointment. Never assume
   appointment details -- always look them up.
3. Read back the appointment you found and confirm they want to cancel it.
4. Call cancel_appointment to cancel it. Never confirm a cancellation
   without calling this function.
5. If no appointment is found, apologize and ask if there's anything else
   you can help with.
6. Once cancelled, confirm it clearly, then ask if there's anything else you
   can help with.
7. If the caller says goodbye at any point, say a warm closing and
   immediately call end_call.
"""
)

ROUTER_PROMPT = """
You are a silent internal router for a clinic voice assistant. You never
speak to the caller -- you only decide which specialist flow should handle
the NEXT response, based on the conversation so far.

The current active flow is: {active_flow}

Rules:
- If the active flow is "booking" or "cancellation" and the caller is
  continuing that same task (giving details, confirming, answering a
  follow-up question), keep the SAME flow.
- Switch to "booking" if the caller wants to schedule a new appointment.
- Switch to "cancellation" if the caller wants to cancel an existing
  appointment.
- Switch to "general" once a booking or cancellation has just been
  completed and the assistant is asking "anything else?" -- stay in
  "general" until the caller states a new concrete request.
- Use "general" for greetings, small talk, unclear requests, or goodbyes.

Pick exactly one flow.
"""


class RouteDecision(BaseModel):
    flow: Literal["booking", "cancellation", "general"] = Field(
        description="Which flow should handle the caller's next turn."
    )


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_data.json")

with open(_DATA_PATH) as f:
    _DB = json.load(f)


# ---------------------------------------------------------------------------
# End-call signalling
# ---------------------------------------------------------------------------

_call_state: dict = {"should_end": False}


class EndCallWatcher(FrameProcessor):
    """Ends the call once the bot has finished voicing its final response.

    LangGraph tools (unlike pipecat's ``llm.register_function()`` tools) have no
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
# Tools (unchanged from the single-node version)
# ---------------------------------------------------------------------------


@tool
def check_availability(date: str, doctor_name: str) -> dict:
    """Check available appointment slots for a doctor on a given date.

    Args:
        date: Date in YYYY-MM-DD format.
        doctor_name: Doctor's full name.
    """
    logger.info(f"Checking availability for {doctor_name} on {date}")

    doctor_id = None
    for doc in _DB["doctors"]:
        if doc["name"].lower() == doctor_name.lower():
            doctor_id = doc["id"]
            break

    if not doctor_id:
        return {
            "success": False,
            "error": f"Doctor '{doctor_name}' not found. Available doctors are: "
            + ", ".join(d["name"] for d in _DB["doctors"]),
        }

    slots = _DB["availability"].get(doctor_id, {}).get(date, [])

    if not slots:
        return {
            "success": True,
            "available": False,
            "message": f"No availability for {doctor_name} on {date}. Try another date or doctor.",
        }

    return {
        "success": True,
        "available": True,
        "doctor": doctor_name,
        "date": date,
        "slots": slots,
    }


@tool
def book_appointment(
    patient_name: str, date: str, time: str, doctor_name: str, reason: str
) -> dict:
    """Book an appointment for a patient.

    Args:
        patient_name: Patient's full name.
        date: Date in YYYY-MM-DD format.
        time: Time in HH:MM format.
        doctor_name: Doctor's full name.
        reason: Reason for visit.
    """
    logger.info(f"Booking appointment for {patient_name} with {doctor_name} on {date} at {time}")

    # Designed-in failure: 40% chance of slot conflict
    if random.random() < 0.4:
        logger.warning("Slot conflict triggered (designed-in failure)")
        return {
            "success": False,
            "error": "slot_conflict",
            "message": (
                f"Sorry, the {time} slot with {doctor_name} on {date} was just taken. "
                "Please offer the next available slot."
            ),
        }

    appointment_id = f"APT-{random.randint(1000, 9999)}"
    return {
        "success": True,
        "appointment_id": appointment_id,
        "patient_name": patient_name,
        "doctor": doctor_name,
        "date": date,
        "time": time,
        "reason": reason,
        "message": f"Appointment confirmed. Booking reference: {appointment_id}.",
    }


@tool
def get_appointment(patient_name: str) -> dict:
    """Look up an existing appointment for a patient.

    Args:
        patient_name: Patient's full name.
    """
    logger.info(f"Looking up appointment for {patient_name}")
    appointment = _DB["existing_appointments"].get(patient_name.lower())

    if not appointment:
        return {
            "success": False,
            "error": "not_found",
            "message": f"No existing appointment found for {patient_name}.",
        }

    return {"success": True, "appointment": appointment}


@tool
def cancel_appointment(appointment_id: str) -> dict:
    """Cancel an existing appointment.

    Args:
        appointment_id: The appointment ID to cancel.
    """
    logger.info(f"Cancelling appointment {appointment_id}")
    return {
        "success": True,
        "appointment_id": appointment_id,
        "message": f"Appointment {appointment_id} has been successfully cancelled.",
    }


@tool
def end_call() -> dict:
    """End the call once you have fully served the patient and delivered
    your final spoken response."""
    logger.info("[END CALL] Reason: agent_hangup")
    _call_state["should_end"] = True
    return {"status": "ending", "reason": "agent_hangup"}


ALL_TOOLS = [check_availability, book_appointment, get_appointment, cancel_appointment, end_call]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    active_flow: Literal["booking", "cancellation", "general"]


_FLOW_NODES = {"booking": "booking", "cancellation": "cancellation", "general": "general"}

# Only these nodes ever produce text the caller should hear. "router" makes a
# structured-output (JSON) call and "tools" just executes functions -- if
# either of those leaks into the token stream, the caller hears raw JSON or
# tool output spoken aloud instead of a real reply.
_CALLER_FACING_NODES = frozenset(_FLOW_NODES.values())


# ---------------------------------------------------------------------------
# LangGraph <-> LangchainProcessor bridge
# ---------------------------------------------------------------------------


class _GraphInputBridge:
    """Adapts pipecat's ``LangchainProcessor`` to a custom LangGraph ``StateGraph``.

    Two mismatches to bridge:
    1. ``LangchainProcessor`` always calls ``.astream({"input": text}, config=...)``,
       but this graph's state expects ``{"messages": [...]}``.
    2. ``LangchainProcessor`` only ever sets ``configurable.session_id`` (never
       ``thread_id``), but the graph's checkpointer needs ``thread_id`` for
       cross-turn memory -- so a fixed thread_id is injected here. A
       ``recursion_limit`` is also injected as a hard ceiling on any single
       turn's router -> flow -> tools -> flow loop, on top of the graph's own
       deterministic edges.

    Also selects ``stream_mode="messages"`` explicitly: the graph's default
    stream mode yields whole-state dicts, not raw text tokens, which
    ``LangchainProcessor`` can't turn into ``TextFrame``s. ``stream_mode="messages"``
    yields ``(AIMessageChunk, metadata)`` tuples instead -- exactly what
    ``LangchainProcessor.__get_token_value`` expects.

    IMPORTANT: ``stream_mode="messages"`` streams tokens from *every* chat-model
    call inside the graph, not just the flow nodes -- including the internal
    ``router`` node's structured-output call. Unlike tool messages (which are a
    different message type and get discarded automatically), the router's
    output is a normal ``AIMessageChunk`` with real text content (its JSON
    decision), so without an explicit filter it gets forwarded straight into
    TTS and the caller hears ``{"flow":"booking"}`` spoken out loud before the
    actual reply. ``metadata["langgraph_node"]`` tells us which node produced
    each chunk, so we only forward chunks from nodes in
    ``_CALLER_FACING_NODES`` -- this is an allowlist, not a blocklist, so
    adding another internal-only node later (a summarizer, a second router
    pass, etc.) is safe by default rather than requiring you to remember to
    exclude it.

    Built as a plain adapter (not an LCEL ``|`` composition) so it never shows
    up as its own spurious node in Tuner's tracked node transitions -- only
    the graph's real nodes ("router", "booking", "cancellation", "general",
    "tools") do.
    """

    def __init__(self, graph, thread_id: str) -> None:
        self._graph = graph
        self._thread_id = thread_id

    def _with_thread_id(self, config: dict | None) -> dict:
        merged = dict(config or {})
        configurable = dict(merged.get("configurable") or {})
        configurable.setdefault("thread_id", self._thread_id)
        merged["configurable"] = configurable
        merged.setdefault("recursion_limit", 25)
        return merged

    async def astream(self, input, config=None, **kwargs):
        messages = {"messages": [HumanMessage(content=input["input"])]}
        async for message_chunk, metadata in self._graph.astream(
            messages, self._with_thread_id(config), stream_mode="messages", **kwargs
        ):
            if metadata.get("langgraph_node") not in _CALLER_FACING_NODES:
                continue
            yield message_chunk

    async def ainvoke(self, input, config=None, **kwargs):
        messages = {"messages": [HumanMessage(content=input["input"])]}
        return await self._graph.ainvoke(messages, self._with_thread_id(config), **kwargs)


def build_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    router_llm = llm.with_structured_output(RouteDecision)
    booking_llm = llm.bind_tools([check_availability, book_appointment, end_call])
    cancellation_llm = llm.bind_tools([get_appointment, cancel_appointment, end_call])
    general_llm = llm.bind_tools([end_call])

    async def router_node(state: ConversationState) -> dict:
        active_flow = state.get("active_flow", "general")
        prompt = ROUTER_PROMPT.format(active_flow=active_flow)
        # Structured-output call: used purely to pick a flow, never added to
        # the persisted transcript, so it can't pollute history or leak into
        # what the caller hears.
        decision = await router_llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        return {"active_flow": decision.flow}

    async def booking_node(state: ConversationState) -> dict:
        response = await booking_llm.ainvoke(
            [SystemMessage(content=BOOKING_PROMPT)] + state["messages"]
        )
        return {"messages": [response], "active_flow": "booking"}

    async def cancellation_node(state: ConversationState) -> dict:
        response = await cancellation_llm.ainvoke(
            [SystemMessage(content=CANCELLATION_PROMPT)] + state["messages"]
        )
        return {"messages": [response], "active_flow": "cancellation"}

    async def general_node(state: ConversationState) -> dict:
        response = await general_llm.ainvoke(
            [SystemMessage(content=GENERAL_PROMPT)] + state["messages"]
        )
        return {"messages": [response], "active_flow": "general"}

    def route_after_router(state: ConversationState) -> str:
        return _FLOW_NODES[state.get("active_flow", "general")]

    def tools_or_end(state: ConversationState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def route_after_tools(state: ConversationState) -> str:
        # active_flow was set by whichever flow node ran right before "tools"
        # and is untouched by tool execution, so this always routes back to
        # the node that made the call.
        return _FLOW_NODES[state.get("active_flow", "general")]

    graph = StateGraph(ConversationState)
    graph.add_node("router", router_node)
    graph.add_node("booking", booking_node)
    graph.add_node("cancellation", cancellation_node)
    graph.add_node("general", general_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, _FLOW_NODES)
    graph.add_conditional_edges("booking", tools_or_end, {"tools": "tools", END: END})
    graph.add_conditional_edges("cancellation", tools_or_end, {"tools": "tools", END: END})
    graph.add_conditional_edges("general", tools_or_end, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", route_after_tools, _FLOW_NODES)

    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Nova Clinic multi-node LangGraph assistant")

    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), voice="alloy")

    def calculate_cost(usage: CallUsage) -> float:
        # OpenAI gpt-4o-mini pricing (per token / per character / per second)
        llm_cost = (usage.llm_prompt_tokens or 0) * 0.000_000_150
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_000_600
        # OpenAI TTS-1: $15 per 1M characters
        tts_cost = (usage.tts_characters or 0) * 0.000_015
        # OpenAI gpt-4o-transcribe: $6 per 1M audio seconds
        stt_cost = usage.stt_audio_seconds * 0.000_006
        return (llm_cost + tts_cost + stt_cost) * 100

    turn_tracker = TurnTrackingObserver()
    observer = Observer(
        api_key=os.getenv("TUNER_API_KEY"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID", "0")),
        agent_id=os.getenv("TUNER_AGENT_ID", "nova-clinic-langgraph"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "https://api.usetuner.ai"),
        asr_model="openai/gpt-4o-transcribe",
        llm_model="gpt-4o-mini",
        tts_model="openai/tts-1",
        cost_calculator=calculate_cost,
        debug=True,
    )
    observer.attach_turn_tracking_observer(turn_tracker)

    # Built fresh per call (unlike a module-level singleton) so each call gets
    # its own checkpointer -- no cross-call state leakage, no manual reset needed.
    graph = build_graph()
    bridge = _GraphInputBridge(graph, thread_id="webrtc-demo-session")
    # wrap_graph() is called on the bridge, not the raw graph, because LangchainProcessor
    # calls .astream()/.ainvoke() on whatever it's given -- and only the bridge accepts
    # LangchainProcessor's {"input": text} shape. wrap_graph() works with anything
    # exposing that interface (it injects its callback into the `config` dict and
    # otherwise delegates every attribute untouched), and _GraphInputBridge.astream()
    # forwards that same `config` -- callback included -- straight into the real
    # graph's own .astream() call, so tool calls/node transitions are still captured
    # on `graph` itself, not just at the bridge boundary.
    wrapped_graph = observer.wrap_graph(bridge)
    lc_processor = LangchainProcessor(chain=wrapped_graph, transcript_key="input")
    lc_processor.set_participant_id("webrtc-demo-session")

    # No tools schema needed here -- pipecat doesn't need to know about the
    # tools, LangGraph resolves and executes them internally.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    end_call_watcher = EndCallWatcher()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
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
        logger.info("Client connected")
        _call_state["should_end"] = False
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
    """Pipecat dev-runner entrypoint (webrtc only -- no telephony transports)."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

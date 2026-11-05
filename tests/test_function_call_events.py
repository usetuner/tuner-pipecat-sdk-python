"""Tool rows from pipecat's FunctionCallObserver (pipecat >= 1.9).

Each call gets one row when it goes in progress and one result row when it settles, whatever
settles it, timed at the moment the observer reports.
"""

import asyncio
from types import SimpleNamespace

import pytest


def _event(r, kind, ms, **fields):
    event = {
        "function_name": "add_drug",
        "tool_call_id": "call_1",
        "arguments": None,
        "result": None,
        "error": None,
        **fields,
    }
    return SimpleNamespace(kind=kind, timestamp=r.base / 1_000_000_000 + ms / 1000, **event)


def _tool_rows(rows):
    return [
        (row.role, row.start_ms, row.tool.result if row.role == "agent_result" else None)
        for row in rows
        if row.role in ("agent_function", "agent_result")
    ]


@pytest.mark.parametrize(
    ("kind", "fields", "result"),
    [
        ("function_call_completed", {"result": {"added": True}}, {"added": True}),
        ("function_call_failed", {"error": "drug not found"}, {"error": "drug not found"}),
        ("function_call_timed_out", {}, {"timed_out": True}),
        ("function_call_cancelled", {}, {"cancelled": True}),
    ],
)
def test_a_call_gets_one_row_in_progress_and_one_as_it_settles(
    replay, tuner_config, kind, fields, result
):
    r = replay()
    r.acc.on_function_call_event(_event(r, "function_call_started", 900))
    r.acc.on_function_call_event(
        _event(r, "function_call_in_progress", 1000, arguments={"name": "carvedilol"})
    )
    r.acc.on_function_call_event(_event(r, kind, 1400, **fields))
    assert _tool_rows(r.rows(tuner_config)) == [
        ("agent_function", 1000, None),
        ("agent_result", 1400, result),
    ]


async def test_observer_reports_each_call_once_from_its_first_push():
    pytest.importorskip("pipecat.observers.function_call_observer")
    from unittest.mock import Mock

    from pipecat.frames.frames import (
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        FunctionCallResultProperties,
    )
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from tuner_pipecat_sdk.observer import Observer

    if "first_push" not in FramePushed.__dataclass_fields__:
        pytest.skip("pipecat < 1.12 has no first_push")

    observer = Observer(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
        forward_traces=False,
    )
    observer._acc.call_start_abs_ns = 1

    async def push(frame, first_push=True):
        await observer.on_push_frame(
            FramePushed(
                source=Mock(),
                destination=Mock(),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=first_push,
            )
        )

    result = FunctionCallResultFrame("add_drug", "call_1", {}, {"added": True})
    await push(FunctionCallInProgressFrame("add_drug", "call_1", {}))
    await push(
        FunctionCallResultFrame(
            "add_drug",
            "call_1",
            {},
            {"progress": "searching"},
            properties=FunctionCallResultProperties(is_final=False),
        )
    )
    await push(result)
    await push(result, first_push=False)  # the same frame, pushed again further along
    await asyncio.sleep(0)  # FunctionCallObserver runs its event handler as a task

    turns = [(t.kind, t.result) for t in observer._acc.live_turns]
    assert turns == [("agent_function", None), ("agent_result", {"added": True})]

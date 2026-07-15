"""LangChain/LangGraph observability bridge — optional, isolated in this module.

Everything that touches ``tuner_langchain`` lives here, including the ``Observer``-
facing ``LangchainIntegration`` helper, so the rest of the SDK (``_base.py``,
``accumulator.py``) never needs to import or type against ``tuner_langchain`` at
all. If it isn't installed, ``_LANGCHAIN_AVAILABLE`` is False and every public entry
point in this module fails loudly and early (``LangchainIntegration.wrap_graph``/
``wrap_chain``, called from ``Observer.wrap_graph``/``wrap_chain``, raise a clear
``ImportError``); nothing else in the SDK is affected.

Consumes tuner-langchain's public surface (``wrap_graph``, ``wrap_chain``,
``TunerAccumulator``, ``segments_from_invocation``) as-is, plus one private,
undocumented hook (``_WrappedRunnable._attach_session``) used to report LLM token
usage. Every touch point on that hook is defensive: a shape mismatch logs a warning
and degrades (LLM usage/cost read as zero for LangChain turns) rather than crashing
the pipeline.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from loguru import logger

from .models import NodeInfo, ToolInfo, TranscriptSegment

if TYPE_CHECKING:
    from tuner_langchain import CaptureConfig

    from .accumulator import CallAccumulator

try:
    from tuner_langchain import TunerAccumulator
    from tuner_langchain.graph_wrapper import wrap_chain as _lg_wrap_chain
    from tuner_langchain.graph_wrapper import wrap_graph as _lg_wrap_graph
    from tuner_langchain.segment_builder import segments_from_invocation

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    TunerAccumulator = None  # type: ignore[assignment,misc]
    _lg_wrap_graph = None  # type: ignore[assignment]
    _lg_wrap_chain = None  # type: ignore[assignment]
    segments_from_invocation = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False


class _AccumulatorUsageSink:
    """Defensive adapter binding tuner-langchain's private usage hook to a pipecat
    ``CallAccumulator``.

    tuner-langchain's ``TunerBaseHandler.on_llm_end()`` calls
    ``self._session._record_llm_usage(prompt, completion)`` on whatever object was
    attached via ``_WrappedRunnable._attach_session(session)``, and reads
    ``session._start_ns`` to convert its own absolute timestamps into relative ms.
    Both are underscore-prefixed and undocumented — see the module docstring for
    why every use of them here is guarded rather than assumed to work.

    Concurrency note: these forwarding calls assume tuner-langchain invokes
    ``on_llm_end()`` on the ``Observer``'s own asyncio event-loop thread, the same
    assumption the rest of the SDK makes (``CallAccumulator`` has no locking, by
    design, since pipecat observers run on a single event loop). If tuner-langchain
    ever delivered callbacks from a different thread (e.g. via a thread-pool bridge
    for sync callback paths), these counter updates would race.
    """

    def __init__(self, acc: CallAccumulator) -> None:
        self._acc = acc

    @property
    def _start_ns(self) -> int:
        return self._acc.call_start_abs_ns

    @property
    def _last_llm_duration_ms(self) -> int | None:
        return self._acc._pending_external_llm_duration_ms

    @_last_llm_duration_ms.setter
    def _last_llm_duration_ms(self, value: int) -> None:
        # tuner-langchain's TunerBaseHandler.on_llm_end() writes this directly as a
        # plain attribute assignment (session._last_llm_duration_ms = ms) on every
        # LLM call inside the wrapped runnable. Forwarding it immediately here (same
        # "push as it happens" pattern as _record_llm_usage below) is what lets
        # on_latency_breakdown() surface it as llm_node_ttft -- see accumulator.py's
        # record_external_llm_duration_ms().
        self._acc.record_external_llm_duration_ms(value)

    def _record_llm_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._acc.record_external_llm_usage(prompt_tokens, completion_tokens)

    def attach(self, wrapped_runnable: Any) -> None:
        """Best-effort: bind self as the wrapped runnable's usage-reporting session.

        Never raises. If tuner-langchain's private ``_attach_session`` hook has
        changed shape, LLM usage/cost for LangChain-driven turns on this call
        silently reads as zero — everything else (transcript, tool calls, node
        transitions) keeps working normally. Caught broadly, deliberately: the
        hook is undocumented and private, so any exception from it (not just
        ``AttributeError``/``TypeError`` for a missing/reshaped hook) is an
        unknown-future-shape-change failure, not something to let propagate out
        of ``wrap_graph()``/``wrap_chain()``.
        """
        try:
            wrapped_runnable._attach_session(self)
        except Exception as exc:
            logger.warning(
                "[tuner] could not attach LangChain usage sink — tuner-langchain's "
                "_attach_session hook may have changed shape. LLM usage/cost for "
                "LangChain-driven turns will be unavailable this call: {}",
                exc,
            )


def _segment_dict_to_transcript_segment(d: dict[str, Any]) -> TranscriptSegment:
    """Convert one of tuner-langchain's ``segments_from_invocation()`` dicts into
    our own typed ``TranscriptSegment``, reusing ``ToolInfo``/``NodeInfo`` as-is."""
    tool_dict = d.get("tool")
    tool_info = None
    duration_ms = d.get("duration_ms")
    if tool_dict is not None:
        tool_dict = dict(tool_dict)
        # ToolInfo has no duration_ms field; lift it onto the segment instead of
        # silently dropping it (pydantic v2 ignores unknown kwargs by default).
        popped = tool_dict.pop("duration_ms", None)
        if duration_ms is None:
            duration_ms = popped
        tool_info = ToolInfo(**tool_dict)

    node_dict = d.get("node")
    node_info = NodeInfo(**node_dict) if node_dict is not None else None

    return TranscriptSegment(
        role=d["role"],
        text=d.get("text"),
        start_ms=d.get("start_ms") or 0,
        end_ms=d.get("end_ms"),
        metadata=d.get("metadata") or {},
        duration_ms=duration_ms,
        tool=tool_info,
        node=node_info,
    )


def merge_langchain_segments(
    segments: list[TranscriptSegment],
    lg_accumulator: Any,
    call_start_abs_ns: int,
) -> list[TranscriptSegment]:
    """Merge node/tool-call segments captured by a tuner-langchain ``TunerAccumulator``
    into the native pipecat transcript, sorted by ``start_ms``.

    Extends the native list with the langchain-sourced segments, then stable-sorts
    by start_ms (ties keep native rows first, since they're appended first).

    Never raises, and degrades per-item, not all-or-nothing. This runs inside
    ``_flush()``'s fire-and-forget task, with no caller in a position to catch
    anything -- a failure here must not cost the caller the entire call's payload
    (transcript, usage, cost) just because part of the LangChain-sourced portion of
    it broke. Only a failure reading ``lg_accumulator.get_invocations()`` itself
    (e.g. an incompatible tuner-langchain shape) falls back to the native-only
    transcript, since there's nothing to partially recover from at that point. A
    failure converting one invocation (``segments_from_invocation``) or one segment
    dict (``_segment_dict_to_transcript_segment``) — a KeyError, a pydantic
    ValidationError, or anything else — logs a warning and skips just that item;
    every other invocation's/segment's data still makes it into the merged
    transcript. Caught broadly (not narrowed to specific exception types)
    deliberately: the whole point is defending against *unknown future* shape
    changes in an external package, not a known, finite set of failure modes.
    """
    if lg_accumulator is None:
        return segments
    if not _LANGCHAIN_AVAILABLE:
        logger.warning(
            "[tuner] lg_accumulator was provided but tuner_langchain is not "
            "installed; skipping LangChain segment merge"
        )
        return segments

    try:
        invocations = lg_accumulator.get_invocations()
    except Exception as exc:
        logger.warning(
            "[tuner] could not read LangChain invocations — tuner-langchain's "
            "accumulator shape may have changed. LangChain-sourced transcript rows "
            "will be unavailable this call: {}",
            exc,
        )
        return segments

    merged = list(segments)
    for invocation in invocations:
        try:
            seg_dicts = segments_from_invocation(invocation, call_start_abs_ns)
        except Exception as exc:
            logger.warning(
                "[tuner] skipping one LangChain invocation — segment extraction "
                "failed: {}",
                exc,
            )
            continue
        for seg_dict in seg_dicts:
            try:
                merged.append(_segment_dict_to_transcript_segment(seg_dict))
            except Exception as exc:
                logger.warning(
                    "[tuner] skipping one malformed LangChain segment: {}", exc
                )
                continue

    merged.sort(key=lambda s: s.start_ms)
    return merged


class LangchainIntegration:
    """Owns all LangChain/LangGraph integration state for one ``Observer``.

    ``_base.py`` holds a single instance of this class and delegates
    ``wrap_graph()``/``wrap_chain()`` to it, so the core observer never imports or
    type-hints against ``tuner_langchain`` (``TunerAccumulator``, ``CaptureConfig``)
    itself, and never branches on LangChain-specific state beyond checking
    ``active``. Constructed unconditionally per-Observer, even when tuner-langchain
    isn't installed -- it only raises once ``wrap_graph()``/``wrap_chain()`` are
    actually called.
    """

    def __init__(self, acc: CallAccumulator) -> None:
        self._acc = acc
        self._lg_accumulator: Any = None  # a TunerAccumulator instance, once created

    @property
    def active(self) -> bool:
        """True once a LangChain/LangGraph runnable has been wrapped for this call.

        ``_base.py`` checks this to skip native ``FunctionCallInProgressFrame``/
        ``FunctionCallResultFrame`` capture once tool calls are being captured via
        tuner-langchain's callback handler instead, so a call is never double-recorded.
        """
        return self._lg_accumulator is not None

    def wrap_graph(self, graph: Any, capture: CaptureConfig | None = None) -> Any:
        """Wrap a LangGraph graph for Tuner observability. See ``Observer.wrap_graph``."""
        return self._wrap(_lg_wrap_graph, graph, capture, "LangGraph")

    def wrap_chain(self, chain: Any, capture: CaptureConfig | None = None) -> Any:
        """Wrap a plain LangChain runnable for Tuner observability. See ``Observer.wrap_chain``."""
        return self._wrap(_lg_wrap_chain, chain, capture, "LangChain")

    def _wrap(
        self, wrap_fn: Any, runnable: Any, capture: CaptureConfig | None, label: str
    ) -> Any:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                f"tuner-langchain is required for {label} support. "
                "Install it with: pip install tuner-pipecat-sdk[langchain]"
            )
        self._ensure_accumulator(capture)
        wrapped = wrap_fn(runnable, accumulator=self._lg_accumulator)
        _AccumulatorUsageSink(self._acc).attach(wrapped)
        return wrapped

    def _ensure_accumulator(self, capture: CaptureConfig | None) -> None:
        """Lazily create the shared LangChain accumulator on first use, and
        register its segment-merging enricher with ``self._acc`` at the same
        time -- exactly once, regardless of how many times wrap_graph()/
        wrap_chain() are called on this Observer. Registering twice would
        merge the same invocations into the transcript twice.
        """
        if self._lg_accumulator is None:
            self._lg_accumulator = TunerAccumulator(capture=capture)
            self._acc.register_segment_enricher(
                partial(
                    merge_langchain_segments,
                    lg_accumulator=self._lg_accumulator,
                    call_start_abs_ns=self._acc.call_start_abs_ns,
                )
            )
        elif capture is not None:
            logger.warning(
                "[tuner] wrap_graph()/wrap_chain() called again with a `capture` "
                "config on an Observer that already has one — ignoring it. All "
                "wrap_graph()/wrap_chain() calls on the same Observer share one "
                "LangChain accumulator, configured by whichever call created it first."
            )

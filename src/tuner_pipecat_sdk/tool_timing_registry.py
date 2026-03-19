"""Tool call timing registry keyed by tool_call_id."""

from dataclasses import dataclass, field


@dataclass
class ToolTimingRegistry:
    """Per-tool-call-id timing store for function invocation and completion."""

    _invocations_ns: dict[str, int] = field(default_factory=dict)
    _completions_ns: dict[str, int] = field(default_factory=dict)

    def record_invocation_ns(self, tool_call_id: str, abs_ns: int) -> None:
        self._invocations_ns[tool_call_id] = abs_ns

    def record_completion_ns(self, tool_call_id: str, abs_ns: int) -> None:
        self._completions_ns[tool_call_id] = abs_ns

    def get_invocation_ns(self, tool_call_id: str) -> int | None:
        return self._invocations_ns.get(tool_call_id)

    def get_completion_ns(self, tool_call_id: str) -> int | None:
        return self._completions_ns.get(tool_call_id)

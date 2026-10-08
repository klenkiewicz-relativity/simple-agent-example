"""Plain request/handler functions: the first wrapper is the outermost."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class Runtime:
    thread_id: str
    context: Any
    config: RunnableConfig


@dataclass(frozen=True)
class ModelRequest:
    model: Any
    messages: list[BaseMessage]
    tools: list[BaseTool]
    system_message: SystemMessage | None
    state: dict[str, Any]
    runtime: Runtime
    model_settings: dict[str, Any] = field(default_factory=dict)

    def override(self, **changes) -> ModelRequest:
        return replace(self, **changes)


@dataclass(frozen=True)
class ToolRequest:
    tool_call: ToolCall
    tool: BaseTool | None
    state: dict[str, Any]
    runtime: Runtime

    def override(self, **changes) -> ToolRequest:
        return replace(self, **changes)


@dataclass(frozen=True)
class ToolResult:
    """A tool reply and JSON state values, committed together in one event."""

    message: ToolMessage
    updates: dict[str, Any] = field(default_factory=dict)


ModelHandler = Callable[[ModelRequest], AIMessage]
ToolHandler = Callable[[ToolRequest], ToolMessage | ToolResult]
ModelWrapper = Callable[[ModelRequest, ModelHandler], AIMessage]
ToolWrapper = Callable[[ToolRequest, ToolHandler], ToolMessage | ToolResult]


def chain(handler, wrappers):
    for wrapper in reversed(wrappers):
        handler = partial(wrapper, handler=handler)
    return handler

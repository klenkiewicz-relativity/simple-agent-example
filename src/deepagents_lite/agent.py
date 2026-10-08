"""Execution is a bounded for loop. Its only durable program counter is the transcript."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool

from .middleware import (
    ModelRequest,
    ModelWrapper,
    Runtime,
    ToolRequest,
    ToolResult,
    ToolWrapper,
    chain,
)
from .store import Event, EventStore, MemoryStore


class StepLimit(RuntimeError):
    """Model-call budget exhausted. invoke(None, same_config) can continue."""


def project(events: list[Event]) -> tuple[dict, bool]:
    state: dict[str, Any] = {"messages": []}
    active = False
    for event in events:
        state.update(event.data.get("updates", {}))
        messages = messages_from_dict(event.data["messages"])
        state["messages"].extend(messages)
        if event.kind == "input":
            active = True
        elif event.kind == "model":
            active = bool(messages[-1].tool_calls)
    return state, active


def pending(messages: list) -> list:
    """Only tool replies following the latest assistant message satisfy its calls."""
    answered = set()
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            answered.add(message.tool_call_id)
        elif isinstance(message, AIMessage):
            return [call for call in message.tool_calls if call["id"] not in answered]
    return []


class Agent:
    def __init__(
        self,
        model,
        tools: Sequence = (),
        *,
        system_prompt: str | SystemMessage | None = None,
        store: EventStore | None = None,
        model_wrappers: Sequence[ModelWrapper] = (),
        tool_wrappers: Sequence[ToolWrapper] = (),
        max_steps: int = 30,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.model, self.max_steps = model, max_steps
        self.tools = [t if isinstance(t, BaseTool) else tool(t) for t in tools]
        self.by_name = {t.name: t for t in self.tools}
        if len(self.by_name) != len(self.tools):
            raise ValueError("Tool names must be unique")
        if any(t.return_direct for t in self.tools):
            raise ValueError("return_direct is unsupported; completion requires a final AIMessage")
        self.system = (
            SystemMessage(content=system_prompt)
            if isinstance(system_prompt, str)
            else system_prompt
        )
        self.store = store if store is not None else MemoryStore()
        self.call_model = chain(self._model, model_wrappers)
        self.call_tool = chain(self._tool, tool_wrappers)

    @staticmethod
    def _thread(config: RunnableConfig | None) -> str:
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("config['configurable']['thread_id'] must be a nonempty string")
        return thread_id

    @staticmethod
    def _model(request: ModelRequest) -> AIMessage:
        model = request.model.bind_tools(request.tools) if request.tools else request.model
        messages = ([request.system_message] if request.system_message else []) + request.messages
        return model.invoke(messages, config=request.runtime.config, **request.model_settings)

    @staticmethod
    def _tool(request: ToolRequest) -> ToolMessage:
        if request.tool is None:
            raise ValueError(f"Unknown tool: {request.tool_call['name']}")
        return request.tool.invoke(request.tool_call, config=request.runtime.config)

    def get_state(self, config: RunnableConfig) -> dict[str, Any]:
        return project(self.store.read(self._thread(config)))[0]

    def invoke(self, input: Mapping | None, config: RunnableConfig, *, context=None) -> dict:
        for _ in self.stream(input, config, context=context):
            pass
        return self.get_state(config)

    def as_runnable(self) -> RunnableLambda:
        """Opt-in LCEL composition; RunnableLambda's streaming is final-state-only."""
        return RunnableLambda(lambda input, config: self.invoke(input, config))

    def stream(
        self, input: Mapping | None, config: RunnableConfig, *, context=None
    ) -> Iterator[Event]:
        """Yield newly committed events, not token chunks. None resumes an unfinished turn."""
        thread_id = self._thread(config)
        runtime = Runtime(thread_id, context, config)
        events = self.store.read(thread_id)
        state, active = project(events)
        version = len(events)

        def commit(kind, messages, updates=None):
            nonlocal version
            updates = updates or {}
            if "messages" in updates or "jump_to" in updates:
                raise ValueError("State updates cannot replace messages or control graph routing")
            data = {"messages": messages_to_dict(messages), "updates": updates}
            event = self.store.append(thread_id, version, kind, data)
            version = event.seq
            # Apply the serialized form: runtime and replay see exactly the same values.
            state.update(deepcopy(event.data["updates"]))
            state["messages"].extend(messages_from_dict(event.data["messages"]))
            return event

        if input is not None:
            if active:
                raise ValueError("Unfinished turn: resume with input=None before adding a new turn")
            messages = convert_to_messages(input.get("messages", []))
            if not messages or any(not isinstance(m, HumanMessage) for m in messages):
                raise ValueError(
                    "Input must contain one or more human messages (new messages only)"
                )
            yield commit("input", messages, {k: v for k, v in input.items() if k != "messages"})
            active = True
        if not active:
            return

        for _ in range(self.max_steps):
            for call in pending(state["messages"]):
                request = ToolRequest(
                    deepcopy(call), self.by_name.get(call["name"]), deepcopy(state), runtime
                )
                result = self.call_tool(request)
                result = result if isinstance(result, ToolResult) else ToolResult(result)
                if not isinstance(result.message, ToolMessage):
                    raise TypeError("Tool wrappers must return ToolMessage or ToolResult")
                if result.message.tool_call_id != call["id"]:
                    raise ValueError("Tool result must retain the original tool_call_id")
                yield commit("tool", [result.message], result.updates)

            response = self.call_model(
                ModelRequest(
                    self.model,
                    deepcopy(state["messages"]),
                    list(self.tools),
                    deepcopy(self.system),
                    deepcopy(state),
                    runtime,
                )
            )
            if not isinstance(response, AIMessage):
                raise TypeError("Model wrappers must return AIMessage")
            ids = [call.get("id") for call in response.tool_calls]
            if (
                response.invalid_tool_calls
                or any(not id for id in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(
                    "Malformed tool calls: IDs must be nonempty and unique per response"
                )
            yield commit("model", [response])
            if not response.tool_calls:
                return
        raise StepLimit(
            f"Reached {self.max_steps} model calls; resume thread {thread_id!r} with None"
        )

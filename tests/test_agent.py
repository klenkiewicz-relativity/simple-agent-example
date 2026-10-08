from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deepagents_lite import Agent, Conflict, MemoryStore, SQLiteStore, StepLimit, ToolResult
from deepagents_lite.workspace import tools, workspace

CONFIG = {"configurable": {"thread_id": "test"}}
INPUT = {"messages": [("user", "Work")]}


class Scripted(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def calls(*names):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": str(i)} for i, name in enumerate(names)],
    )


def test_restart_skips_completed_tools_and_replays_atomic_state(tmp_path):
    path = tmp_path / "events.sqlite"
    executions = []

    @tool
    def first() -> str:
        """First operation."""
        executions.append("first")
        return "one"

    @tool
    def second() -> str:
        """Second operation."""
        executions.append("second")
        return "two"

    def annotate(request, handler):
        return ToolResult(handler(request), {"completed": request.tool_call["name"]})

    model = Scripted(responses=[calls("first", "second")])
    agent = Agent(model, [first, second], store=SQLiteStore(path), tool_wrappers=[annotate])
    stream = agent.stream(INPUT, CONFIG)
    assert [next(stream).kind for _ in range(3)] == ["input", "model", "tool"]
    stream.close()  # Process stops between tool calls.
    recovered = Agent(
        Scripted(responses=[AIMessage(content="done")]),
        [first, second],
        store=SQLiteStore(path),
        tool_wrappers=[annotate],
    )
    with pytest.raises(ValueError, match="Unfinished"):
        recovered.invoke(INPUT, CONFIG)
    state = recovered.invoke(None, CONFIG)
    assert executions == ["first", "second"]
    assert state["completed"] == "second"
    assert [m.type for m in state["messages"]] == ["human", "ai", "tool", "tool", "ai"]
    assert recovered.invoke(None, CONFIG) == state  # Completed turns do no work.
    assert recovered.get_state({"configurable": {"thread_id": "other"}}) == {"messages": []}


def test_tool_failure_remains_pending():
    attempts = []

    @tool
    def flaky() -> str:
        """Fail once."""
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("offline")
        return "ok"

    agent = Agent(Scripted(responses=[calls("flaky"), AIMessage(content="done")]), [flaky])
    with pytest.raises(RuntimeError, match="offline"):
        agent.invoke(INPUT, CONFIG)
    assert [e.kind for e in agent.store.read("test")] == ["input", "model"]
    assert agent.invoke(None, CONFIG)["messages"][-1].content == "done"
    assert len(attempts) == 2


def test_step_budget_stops_before_unbudgeted_tools_and_can_resume():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(
        Scripted(responses=[calls("ping"), AIMessage(content="done")]), [ping], max_steps=1
    )
    with pytest.raises(StepLimit):
        agent.invoke(INPUT, CONFIG)
    assert [e.kind for e in agent.store.read("test")] == ["input", "model"]
    assert agent.invoke(None, CONFIG)["messages"][-1].content == "done"


def test_workspace_and_multiple_turns_with_reused_call_id():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"path": "a.txt", "content": "hello"}, "id": "same"}
        ],
    )
    agent = Agent(
        Scripted(responses=[message, AIMessage(content="done")]), tools, tool_wrappers=[workspace]
    )
    first = agent.invoke(INPUT, CONFIG)
    assert first["files"] == {"a.txt": "hello"}
    second = agent.invoke(INPUT, CONFIG)
    assert sum(isinstance(m, ToolMessage) for m in second["messages"]) == 2


def test_wrapper_order_config_context_and_prompt_changes():
    order = []

    def outer(request, handler):
        order.append("outer before")
        assert request.runtime.context == {"user": "Ada"}
        assert request.runtime.config["tags"] == ["test"]
        # Mutating a request snapshot must not rewrite durable history.
        request.state["messages"].clear()
        result = handler(request.override(model_settings={"temperature": 0}))
        order.append("outer after")
        return result

    def inner(request, handler):
        order.append("inner")
        assert request.model_settings == {"temperature": 0}
        return handler(request)

    agent = Agent(Scripted(responses=[AIMessage(content="done")]), model_wrappers=[outer, inner])
    state = agent.invoke(INPUT, {**CONFIG, "tags": ["test"]}, context={"user": "Ada"})
    assert order == ["outer before", "inner", "outer after"]
    assert len(state["messages"]) == 2


@pytest.mark.parametrize("store_type", [MemoryStore, SQLiteStore])
def test_store_concurrent_compare_and_append_and_json_isolation(store_type, tmp_path):
    store = store_type(tmp_path / "events.sqlite") if store_type is SQLiteStore else store_type()

    def append(_):
        try:
            store.append("t", 0, "input", {"messages": [], "updates": {"x": []}})
            return True
        except Conflict:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(append, range(4))) == 1
    event = store.read("t")[0]
    event.data["updates"]["x"].append(1)
    assert store.read("t")[0].data["updates"]["x"] == []
    with pytest.raises((TypeError, ValueError)):
        store.append("t", 1, "input", {"bad": object()})
    assert len(store.read("t")) == 1


def test_bad_model_response_is_not_committed():
    agent = Agent(
        Scripted(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "x", "args": {}, "id": "dup"},
                        {"name": "x", "args": {}, "id": "dup"},
                    ],
                )
            ]
        )
    )
    with pytest.raises(ValueError, match="Malformed"):
        agent.invoke(INPUT, CONFIG)
    assert len(agent.store.read("test")) == 1


def test_unknown_tool_can_be_converted_to_error_message_by_wrapper():
    def errors(request, handler):
        try:
            return handler(request)
        except ValueError as exc:
            return ToolMessage(
                content=str(exc), tool_call_id=request.tool_call["id"], status="error"
            )

    agent = Agent(
        Scripted(responses=[calls("missing"), AIMessage(content="sorry")]), tool_wrappers=[errors]
    )
    state = agent.invoke(INPUT, CONFIG)
    assert state["messages"][2].status == "error"


def test_model_failure_can_resume_and_runnable_composes():
    def failure(request, handler):
        raise ConnectionError("provider unavailable")

    store = MemoryStore()
    model = Scripted(responses=[AIMessage(content="done")])
    with pytest.raises(ConnectionError):
        Agent(model, store=store, model_wrappers=[failure]).invoke(INPUT, CONFIG)
    agent = Agent(model, store=store)
    pipeline = agent.as_runnable() | (lambda state: state["messages"][-1].content)
    assert pipeline.invoke(None, config=CONFIG) == "done"


def test_wrong_tool_id_does_not_enter_log():
    def wrong(request, handler):
        return ToolMessage(content="oops", tool_call_id="wrong")

    agent = Agent(Scripted(responses=[calls("missing")]), tool_wrappers=[wrong])
    with pytest.raises(ValueError, match="original tool_call_id"):
        agent.invoke(INPUT, CONFIG)
    assert len(agent.store.read("test")) == 2

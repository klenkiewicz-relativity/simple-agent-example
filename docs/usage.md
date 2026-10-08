# Client guide

## API contract

```python
Agent(
    model, tools=(), *, system_prompt=None, store=None,
    model_wrappers=(), tool_wrappers=(), max_steps=30,
)
agent.invoke(input, config, *, context=None) -> dict
agent.stream(input, config, *, context=None) -> Iterator[Event]
agent.get_state(config) -> dict
agent.as_runnable() -> RunnableLambda
```

- `model`: an instantiated LangChain chat model. With tools, it must implement
  `bind_tools`; without tools it only needs `invoke` returning `AIMessage`.
- `tools`: `BaseTool` instances or typed/documented sync callables convertible by
  LangChain's `tool` helper. Names must be unique. Pass unbound models ordinarily.
- `input`: `{"messages": [new human messages], ...JSON state updates}` or `None`.
  Role/content dictionaries, `HumanMessage`, and `("user", text)` tuples work.
  Historical assistant/tool messages are loaded from the log, not imported via input.
- `config`: a `RunnableConfig` containing `configurable.thread_id`, a nonempty
  string. Tags, callbacks, and metadata propagate to model/tool invocations.
- `context`: transient application data available at `request.runtime.context`.
  Pass it again when resuming; it isn't serialized. The dataclass is frozen but
  objects inside context/config are caller-owned and should be treated as read-only.
- `max_steps`: positive model-call budget per invocation; tool calls within one
  response execute sequentially. It doesn't bound wrapper-internal retries, tool
  counts, child-agent calls, wall time, or tokens.
- `Event`: `thread_id`, `seq` (starting at 1), `kind` (`input`, `model`, `tool`), and
  JSON `data` with `messages` and `updates`.

`StepLimit` leaves a recoverable turn. Resume with `agent.invoke(None, config)`.
An unknown tool, invalid arguments, or application error raises by default. Supplying
`return_direct=True` on a tool is rejected because completion requires a final model
message. Only one invoker may execute a given thread at a time.

## Native middleware

Wrappers are ordinary `def wrapper(request, handler)` functions. Name the second
parameter `handler` (or accept `**kwargs`) because composition binds it by keyword.
The first wrapper in each list is outermost. A wrapper may call the handler zero,
one, or several times. Model handlers return `AIMessage`; tool handlers return
`ToolMessage` or `ToolResult`.

### Model routing and prompt customization

```python
from langchain_core.messages import SystemMessage


def route(request, handler):
    context = request.runtime.context
    model = context["large_model"] if context["complex"] else request.model
    return handler(
        request.override(
            model=model,
            system_message=SystemMessage(content=f"Help tenant {context['tenant']}."),
            model_settings={**request.model_settings, "temperature": 0},
        )
    )


agent = Agent(small_model, model_wrappers=[route])
```

`model_settings` is forwarded to model `invoke`, so valid keys depend on the provider.
Tools on a model request can be filtered; requested tool execution still resolves
against the Agent's registered tools. To enforce execution-level policies, use a
tool wrapper too. To introduce new executable tools, register them on the Agent;
changing only a model's advertised schemas doesn't register an implementation.

Request transformations affect the model call but are not logged. Request/state
snapshots are detached from live state. Mutations do not become persistent updates.

### Deliberate retry policy

```python
def retry_connection_errors(request, handler):
    for attempt in range(3):
        try:
            return handler(request)
        except ConnectionError:
            if attempt == 2:
                raise
```

Use the provider's actual exception types and backoff in a real client. Persisted
history contains only the returned response, not wrapper-internal attempts. Tool
retries should account for external effects; see [recovery semantics](architecture.md).

### Expected tool errors as model-visible results

```python
from langchain_core.messages import ToolMessage
from pydantic import ValidationError


def validation_errors(request, handler):
    try:
        return handler(request)
    except ValidationError as exc:
        return ToolMessage(
            content=f"Invalid arguments: {exc}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )
```

An error message satisfies the pending call just like a successful result; the
model can request a corrected call. Unexpected exceptions still stop the invocation.

### Durable custom state

```python
from deepagents_lite import ToolResult


def count_tool_calls(request, handler):
    result = handler(request)
    result = result if isinstance(result, ToolResult) else ToolResult(result)
    return ToolResult(
        result.message,
        {
            **result.updates,
            "tool_count": request.state.get("tool_count", 0) + 1,
        },
    )


agent = Agent(model, tools, tool_wrappers=[count_tool_calls])
state = agent.invoke({"messages": [("user", "Do some work")]}, config)
print(state.get("tool_count", 0))
```

Updates shallowly replace values, including dictionaries and lists. They are
committed in the **same event** as the tool reply, so a restart cannot observe one
without the other. There are no reducers, field annotations, or implicit merging.
Custom state isn't automatically added to prompts; expose it through tools or select
it in a model wrapper. There is no separate `update_state` or message-editing API.

## Workspace and planning

```python
from deepagents_lite import Agent, SQLiteStore
from deepagents_lite.workspace import tools, workspace

agent = Agent(
    model,
    tools,
    system_prompt="Plan your work with write_todos and save documents in the workspace.",
    tool_wrappers=[workspace],
    store=SQLiteStore("work.sqlite"),
)
state = agent.invoke({"messages": [("user", "Write a project outline")]}, config)
print(state.get("files", {}))
print(state.get("todos", []))
```

These are conversation-local **virtual** text files stored in state, not host files.
`read_file`, `write_file`, `list_files`, and `write_todos` are illustrative capability
tools. The matching `workspace` wrapper implements them. Plans are a list of strings,
not a workflow scheduler. Files and plans persist through the event store.

## Child agents

See [`examples/subagent.py`](../examples/subagent.py). A tool wrapper receives the
parent thread and call ID, derives a stable child thread, and invokes a second
Agent. Its result becomes an ordinary `ToolMessage`. Parent and child share an
event store but have isolated transcripts. A repeated call resumes/reuses the
child instead of starting another conversation. Each child has its own budget;
no parent-wide budget, recursive registry, or parallel execution is added implicitly.

## Custom store

Implement the exported `EventStore` protocol:

```python
class MyStore:
    def read(
        self, thread_id: str
    ) -> list[Event]: ...  # Detached events ordered by seq, contiguous starting at 1.

    def append(
        self, thread_id: str, expected: int, kind, data: dict
    ) -> Event: ...  # Atomically verify revision, commit JSON, return the committed Event.
```

Raise `Conflict` if the current revision differs from `expected`. Don't silently
retry a stale append: the caller's tool or model may already have executed. Stores
should serialize/copy payloads so mutation of returned events cannot rewrite history.
`SQLiteStore` creates its database and table; the parent directory must already
exist. Its schema is small enough to inspect directly with SQLite tools.

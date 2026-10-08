# LangChain Deep Agents in ~200 lines

**An agent is a loop. Its memory is an event log. Its capabilities are tools.**

A working, deliberately small Python library sketch: a sequential model/tool
loop, resumable SQLite persistence, composable middleware functions, and real
LangChain messages and tools. The execution module is roughly 200 lines; storage,
adapters, examples, and tests are additional, readable code. This is a design
exploration, not feature parity with Deep Agents.

```python
for _ in range(max_steps):
    for call in pending_tool_calls(history):
        append_event(execute_tool(call))
    response = call_model(history)
    append_event(response)
    if not response.tool_calls:
        return state
```

That's the execution model, including recovery: replay history, finish unanswered
tool calls, and continue. There are no graph nodes, channels, scheduler tasks, or
graph compilation. The sole required dependency is `langchain-core` (which itself
has dependencies); installing the optional middleware adapter brings in LangChain
and its transitive LangGraph dependency, but does not run a graph.

## Try it

Run these commands from this directory:

```bash
uv sync
uv run python examples/offline.py
uv run python examples/subagent.py
uv run pytest -q
```

The offline example streams five committed events, writes a virtual file and
plan, then reconstructs them through a new SQLite connection. No API key needed.

For a real provider:

```bash
export OPENAI_API_KEY=...
uv run --extra openai python examples/openai_agent.py
```

`OPENAI_MODEL` optionally overrides the example model. Provider calls are billable.

## Library usage

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from deepagents_lite import Agent, SQLiteStore


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


agent = Agent(
    ChatOpenAI(model="gpt-4.1-mini"),
    tools=[multiply],
    system_prompt="Use tools for arithmetic.",
    store=SQLiteStore("conversation.sqlite"),
)
config = {"configurable": {"thread_id": "alice-1"}}
state = agent.invoke({"messages": [("user", "What is 12 times 13?")]}, config)
print(state["messages"][-1].content)

# Send only the NEW messages; the stored history is loaded automatically.
state = agent.invoke({"messages": [("user", "And times two?")]}, config)

# After an exception or a stopped stream, continue without duplicating input.
state = agent.invoke(None, config)
```

An omitted store uses `MemoryStore`, retaining threads for that Agent instance.
Always provide a thread ID. `None` on an empty/completed thread returns its current
state without calling the model. An unfinished thread rejects new input until resumed.

### Streaming and inspection

```python
for event in agent.stream({"messages": [("user", "What is 7 times 9?")]}, config):
    print(event.seq, event.kind, event.data)

state = agent.get_state(config)  # Plain dict containing real message objects
events = agent.store.read("alice-1")  # JSON-compatible serialized event payloads
pipeline = agent.as_runnable() | (lambda state: state["messages"][-1].content)
```

`stream()` yields **committed events**, not tokens or LangGraph stream modes. It is
lazy: closing it stops further execution. Resume with `None`. `as_runnable()` is an
explicit LCEL adapter; its stream yields the final state rather than these events.

## Extension points

1. **Model wrappers**: change the prompt, route models, transform context, retry,
   measure usage, or short-circuit with an `AIMessage`.
2. **Tool wrappers**: inject application context, adapt failures, delegate to a child
   agent, or return a `ToolResult(message, updates)` for durable state changes.
3. **EventStore**: implement `read` and atomic compare-and-append for another backend.
4. **Ordinary tools**: use `@tool`, `BaseTool`, or a documented, typed Python function.

```python
from langchain_core.messages import SystemMessage


def personalized_prompt(request, handler):
    name = request.runtime.context["name"]
    return handler(
        request.override(system_message=SystemMessage(content=f"Help {name}. Be concise."))
    )


agent = Agent(model, model_wrappers=[personalized_prompt])  # Your chat model instance
state = agent.invoke({"messages": [("user", "Hello")]}, config, context={"name": "Ada"})
```

Wrappers run first-listed outermost. Their `handler` parameter must accept that
keyword. State/messages are detached snapshots: use returned tool updates to
persist state, rather than mutating a request. Context and model settings are not
persisted. See [the client guide](docs/usage.md) for complete patterns.

## Where to look

| File | Purpose |
| --- | --- |
| [`src/deepagents_lite/agent.py`](src/deepagents_lite/agent.py) | Loop, replay, pending calls, public API |
| [`src/deepagents_lite/store.py`](src/deepagents_lite/store.py) | Memory/SQLite event stores with revision checks |
| [`src/deepagents_lite/middleware.py`](src/deepagents_lite/middleware.py) | Request/result dataclasses and function composition |
| [`src/deepagents_lite/workspace.py`](src/deepagents_lite/workspace.py) | Persistent virtual files and planning as tools |
| [`src/deepagents_lite/compat.py`](src/deepagents_lite/compat.py) | Optional LangChain wrapper adapters |
| [`docs/architecture.md`](docs/architecture.md) | Persistence decision, recovery guarantees, tradeoffs |
| [`docs/compatibility.md`](docs/compatibility.md) | Supported LangChain surface and explicit limits |
| [`docs/usage.md`](docs/usage.md) | Client recipes and API semantics |

## Development

```bash
uv sync --extra compat
uv run --extra compat pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

The adapter test uses actual LangChain decorators; it skips without the `compat`
extra. `uv.lock` captures the tested dependency set. Tests cover recovery between
tools, failed calls, per-call budgets, state replay, writer conflicts, middleware
composition, and invalid responses.

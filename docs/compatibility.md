# LangChain / LangGraph compatibility

The target is **reuse at the model, tool, message, and wrapper boundary**, not a
drop-in `CompiledStateGraph`. The default dependency is `langchain-core`; LangChain
and LangGraph are not imported by the core runtime.

| Surface | Support |
| --- | --- |
| `AIMessage`, `HumanMessage`, `ToolMessage`, `SystemMessage` | Real `langchain-core` objects, including serialized metadata |
| Chat models | `bind_tools` + sync `invoke`; instantiate the provider integration yourself |
| `@tool`, `BaseTool`, plain documented functions | Sync invocation and LangChain argument validation |
| `ToolMessage` artifacts/status | Preserved if JSON-serializable |
| `RunnableConfig` | Forwarded to models/tools; `configurable.thread_id` selects persisted history |
| `invoke({"messages": ...}, config)` | Similar call shape; only **new human input** accepted |
| `get_state` | Returns a dict, not a LangGraph `StateSnapshot` |
| LCEL `|`, `batch`, config helpers | Use `agent.as_runnable()`; don't batch simultaneous calls for one thread |
| Event streaming | Native committed `Event` objects; no LangGraph stream modes or token stream |
| Sync LangChain wrapper methods | Optional, restricted adapters shown below |
| Lifecycle hooks (`before_*`, `after_*`) | No direct adapter; translate to native wrappers |
| `AgentState` reducers / `RemoveMessage` | No; history appends, application state uses shallow replacement |
| `ToolRuntime`, `InjectedState`, `InjectedStore` | No graph injection; use a native tool wrapper and `runtime.context` |
| `Command`, graph jumps, interrupts, checkpointers | No; the event store is a different persistence contract |
| Structured-output strategies / `return_direct` | No automatic handling; model wrappers must ultimately return an `AIMessage` |
| Async-only tools / native async agent methods | No; this sketch is synchronous |
| Deep Agents middleware classes wholesale | No; capabilities depend on graph state/runtime and require explicit ports |

## Optional LangChain wrapper adapters

```bash
uv sync --extra compat
uv run --extra compat pytest tests/test_compat.py -q
```

```python
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import SystemMessage, ToolMessage
from deepagents_lite import Agent
from deepagents_lite.compat import langchain_model_wrapper, langchain_tool_wrapper


@wrap_model_call
def add_prompt(request, handler):
    return handler(request.override(system_message=SystemMessage(content="Be concise.")))


@wrap_tool_call
def handle_missing_file(request, handler):
    try:
        return handler(request)
    except FileNotFoundError:
        return ToolMessage(content="File does not exist", tool_call_id=request.tool_call["id"])


agent = Agent(
    model,
    tools,
    model_wrappers=[langchain_model_wrapper(add_prompt.wrap_model_call)],
    tool_wrappers=[langchain_tool_wrapper(handle_missing_file.wrap_tool_call)],
)
```

Pass a **bound wrapper method**, not an entire middleware instance. The adapters
construct real LangChain `ModelRequest` / `ModelResponse` and LangGraph
`ToolCallRequest` objects. This supports request overrides and ordinary decorators
without implementing graph execution. Runtime is a minimal object exposing
`context`; it isn't a real LangGraph Runtime. Middleware requiring `store`,
`stream_writer`, execution info, or other runtime services will not work.

Model returns must be an `AIMessage` or `ModelResponse` containing exactly one
`AIMessage` and no structured response. `ExtendedModelResponse`, commands,
structured-output strategies, and `tool_choice` on the external request are rejected.
Native wrappers can choose provider-specific model settings explicitly.

Tool returns must be `ToolMessage`; `Command` is rejected. The original call ID
must survive. Argument overrides work; if changing the tool itself, change both
the `tool` field and call name consistently. The adapter can't transport native
`ToolResult` state updates through an external handler; place a state-changing
native wrapper **outside** the adapter, or use native wrappers throughout.

Adapters don't register middleware-contributed tools, process state schemas,
run lifecycle hooks, or persist mutations to external request state. Native
wrappers give the most predictable small-library contract. Full middleware
compatibility would require reproducing substantial graph semantics, contrary
to the aim of this experiment.

`as_runnable()` exposes core Runnable methods. Its inherited async support may
offload synchronous work to threads; this does not make this Agent async-native,
provide token streaming, or make concurrent execution on one thread safe.

## Reference points

The optional integration tests were run against `langchain==1.4.3`,
`langchain-core==1.6.6`, `langgraph==1.2.12`, and `langgraph-prebuilt==1.1.0`;
see `uv.lock`. Constraints allow compatible 1.x releases, but that isn't a claim
that all historical/future versions or middleware packages have been tested.

Upstream source consulted for this sketch:

- [Deep Agents assembly and capability middleware](https://github.com/langchain-ai/deepagents/blob/master/libs/deepagents/deepagents/graph.py)
- [LangChain middleware request, response, and hook interfaces](https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py)

The source shows why a clean boundary matters: middleware interfaces reference
LangGraph runtime, reducers, commands, and tool-node request types. Reusing message
DTOs and tool schemas is straightforward; reproducing every middleware's runtime
expectations is a separate project.

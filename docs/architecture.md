# Design: a loop plus a log

## What is being simplified?

For a sequential agent, a graph's scheduling problem is usually already solved:
ask the model, execute its requested tools, ask again. This project makes that
control flow explicit in one bounded `for` loop. `max_steps` bounds model calls per
invocation. Each iteration first drains pending tools and then calls the model.
The last budgeted response can therefore leave tools pending for the next invocation.
A final `AIMessage` without tool calls completes the turn, even if its text is empty.

Deep Agents also provides useful *capabilities*: planning, files, delegation,
skills, context management. Those don't inherently require a graph. This sketch
implements a few as tools/wrappers and shows how to build the others at that layer.

The comparison is architectural, not a benchmark or a claim that LangGraph's
machinery has no purpose. Scheduling arbitrary graphs, coordinating parallel
branches, durable interrupts, and distributed execution justify more machinery
than this narrow sequential contract.

## Durable state

Each thread has an ordered append-only stream:

```text
1 input  {messages: [HumanMessage], updates: {tenant: "acme"}}
2 model  {messages: [AIMessage(tool_calls=[a, b])], updates: {}}
3 tool   {messages: [ToolMessage(a)], updates: {files: {...}}}
4 tool   {messages: [ToolMessage(b)], updates: {todos: [...]}}
5 model  {messages: [AIMessage("Done")], updates: {}}
```

Messages use `langchain_core.messages.messages_to_dict` / `messages_from_dict`.
Replaying appends messages and shallowly replaces each key in `updates`.
`messages` and `jump_to` are reserved and cannot be replaced by updates. Values
must be JSON-serializable, including message artifacts and provider metadata.
JSON serialization is strict: unsupported values fail before insertion. Runtime
state is reconstructed from the serialized form so replay has the same values.

There is no independently persisted program counter. The latest input starts an
active turn; a model event with no tool calls ends it. Tool replies after the last
assistant message mark that message's calls complete. Tool IDs may recur in later
model responses but must be unique within one response.

This log records accepted inputs, model responses, and completed tool results.
It is **not** a complete execution audit: failed attempts, transformed requests,
middleware internals, timing, configuration, and provider requests aren't events.
Use callbacks/wrappers for that telemetry; model/tool callbacks receive the supplied
`RunnableConfig`. Agent-level graph traces are not synthesized.

## Why event sourcing here?

It matches the conversational data: messages are already facts being appended.
It also gives a small recovery algorithm and makes tool-result/state-update commits
atomic. Purely virtual workspace writes have no external side effect: either the
tool event exists with its update, or it does not.

| Choice | Advantages | Cost / when to choose |
| --- | --- | --- |
| JSON snapshot per step | Simplest persistence contract and fast reads | Rewrites growing history; no natural record of intermediate facts. Good if only latest state matters. |
| Append-only JSONL | Very inspectable | You must handle partial writes, locking, and atomic compare-and-append. Fine for an explicitly single-process demo. |
| SQLite event log (chosen) | Transactional append, easy inspection, recovery, no server | Replay and storage grow; event schema becomes a compatibility contract. |
| Events plus snapshots | Fast startup with retained history | Snapshot versioning/invalidation; add only when measured replay cost matters. |

**Recommendation:** keep the tiny event log if restart-between-tools and an
inspectable history are requirements. If only cross-turn memory matters, replacing
it with a snapshot is a defensible simplification. Avoid generic event buses,
subscriptions, CQRS infrastructure, or per-token persistence here.

Message storage is linear in accumulated message payload. The example workspace
replaces the entire `files` map on each write, so repeated large workspace writes
can have quadratic storage growth. That is a sketch-level tradeoff, not a claim of
efficient file versioning. Use blobs plus references or explicit file-patch events
when actual usage justifies a richer reducer. Replay is linear in total log bytes;
each model request also copies and sends its selected history.

## Crash and failure semantics

| Stop/failure point | Resume behavior |
| --- | --- |
| Before input commit | No turn accepted; submit input again |
| After input, before model response commit | Call model again |
| After model response commit, before tool | Execute pending tool |
| Between tools | Skip recorded replies and run unanswered calls in original order |
| After external side effect, before result commit | May execute the tool again |
| After final response commit | Return state; don't call model again |
| After event commit, before consumer receives it | Event exists in `store.read`; live streaming doesn't redeliver it |

Model/tool exceptions propagate. No failure result is silently inserted and no
automatic retry is hidden inside the loop. A tool wrapper can convert an expected
failure into `ToolMessage(status="error")`; this counts as a completed tool call.
Invalid model output is rejected before it enters the log, leaving the turn resumable.

Execution is effectively **at least once on retry**, not exactly once. An external
effect and a SQLite event are two separate commits. Use the external service's
idempotency support with a key derived from thread, assistant-message position,
and tool-call ID when duplication matters. A provider call can likewise be billed
twice if its response wasn't committed. A retrying wrapper may call its handler
several times: intermediate attempts are not persisted by the outer loop.

## Concurrency contract

The application must allow only one active invoker per thread. Distinct threads
can share a store. SQLite serializes each short append transaction and checks the
expected revision; the memory store does the same under a lock. A stale append
raises `Conflict` rather than overwriting history.

This check protects **log integrity**, not single execution of tools. Two concurrent
resumers can both perform a side effect before one loses the append race. If serving
concurrent requests, serialize per thread outside the agent (or add a renewable
per-thread lease). A lease and distributed scheduling are intentionally separate
requirements, not implied by SQLite's transactions. `invoke` returns the current
projection after draining the stream, under that single-invoker contract.

## Middleware and deterministic replay

Wrapper order is `outer(inner(model_or_tool))`. Native request dataclasses use
`override(...)` to create a changed request. Wrappers can inspect a detached state
snapshot; they do not own durable state. Only accepted inputs and `ToolResult`
updates change persistent application fields.

Replaying the log is deterministic and never reruns a model, tool, or middleware.
**Continuing** a partially finished turn isn't deterministic: it uses today's
model, prompt, tools, wrapper code, and the newly supplied context/configuration.
An input commit does not freeze those dependencies. Pin application configuration
alongside your own thread metadata if historical continuation needs that property.
No event migration framework is provided in this initial schema.

## Extending without rebuilding a graph engine

- **Context management:** transform `request.messages` in a model wrapper; retain
  the full log. Keep each assistant tool-call message together with its tool replies.
  Persistent summaries can be explicit tool updates, then selected in the wrapper.
- **Skills/memory:** load selected instructions into a system-message wrapper and
  expose file-reading tools. No skill discovery framework is included.
- **Delegation:** a tool invokes another `Agent` under a distinct thread ID. The
  included example derives a stable child ID so a parent retry can reuse a completed
  child transcript. This example supports one specialist, not recursive scheduling.
- **Async:** a future async loop would await model and tool calls directly. The
  current runtime is synchronous; wrapping it in a worker thread doesn't provide
  cooperative cancellation or async-only tools.
- **Human pauses:** an exception before execution can leave work pending. That is
  resumability, not a durable approval/interrupt protocol. Add explicit decisions
  as events only if the application needs them.

If future requirements include general routing, parallel branch joins, time travel,
distributed leases, arbitrary reducers, and interrupts, reconsider using LangGraph
rather than reproducing its runtime one feature at a time.

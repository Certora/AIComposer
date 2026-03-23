# IO Layer Design

## Core Problem

A workflow may spawn many graph executions — sequentially, nested,
or in parallel.  Each execution produces a stream of events (state
changes, checkpoints, custom payloads, HITL interrupts).  The IO
layer must:

1. Collect events from all concurrent executions into a single
   ordered stream
2. Preserve the nesting structure so the consumer knows *which*
   execution produced each event
3. Decouple execution speed from rendering speed
4. Allow domain-specific extension without modifying the core

## Conceptual Model

There are three layers, each with a distinct role:

```
┌────────────────────────────────────┐
│         Event Sinks                │  One per run_graph() call.
│  (write-only, synchronous)         │  Wraps events with nesting
│                                    │  info and pushes to the queue.
└──────────────┬─────────────────────┘
               │  push()
               ▼
┌────────────────────────────────────┐
│         EventQueue                 │  One per with_handler() scope.
│  (append-only buffer)              │  All sinks within a scope
│                                    │  converge here.
└──────────────┬─────────────────────┘
               │  stream_events()
               ▼
┌────────────────────────────────────┐
│    _queue_drainer (background)     │  Consumes the queue and
│         ┌─────────┐                │  dispatches to two consumers:
│         │         │                │
│         ▼         ▼                │
│    IOHandler  EventHandler         │
└────────────────────────────────────┘
```

**Many sinks, one queue, one handler.**  Every `run_graph()` call
gets its own event sink.  All sinks within a `with_handler()` scope
feed the same `EventQueue`.  A single background drainer task
consumes the queue and dispatches events.

## Event Sinks and Nesting

An **event sink** is a `Callable[[AllEvents], None]` — fire and
forget.  `run_graph()` creates one per invocation:

- **Top-level:** The sink is `queue.push` directly.
- **Nested:** The sink wraps the *parent's* sink with a `Nested`
  envelope: `lambda e: parent_sink(Nested(e, parent_id=my_tid))`.

This means a deeply nested call produces events like:

```
Nested(Nested(StateUpdate(...), parent_id="mid"), parent_id="outer")
```

The drainer peels off `Nested` layers to reconstruct a **path** —
a list of thread IDs from outermost to innermost — and passes it to
the handler.  The handler uses this path to route the event to the
correct UI target (e.g. a nested panel within a task view).

Nesting is tracked via a `ContextVar[tuple[SinkProtocol, str]]`.
When `run_graph()` starts, it checks whether a parent sink exists;
if so, it wraps.  On exit, it restores the previous value.  This is
transparent to the code inside `run_graph()` — agents and tools
have no knowledge of their nesting depth.

## with_handler: The Scope Boundary

`with_handler(io_handler, event_handler)` defines a **scope** in
which graph executions are connected to a specific pair of handlers:

1. Creates a fresh `EventQueue`
2. Installs the `(queue, io_handler, event_handler)` tuple into a
   `ContextVar`
3. Spawns `_queue_drainer` as a background task
4. Yields — all `run_graph()` calls within the scope use this queue
5. On exit: cancels the drainer, resets the context var

Any `run_graph()` call reads the current tuple from the context var
to obtain the queue (for the sink) and the handler (for HITL).

## Two Consumer Interfaces

The drainer dispatches to two separate interfaces based on event
type:

### IOHandler[H]

A protocol that handles **structural** events — the lifecycle and
state of graph execution.  Methods receive a `path` parameter for
nesting context.

| Method | Called on |
|--------|----------|
| `log_start(path, description)` | `Start` event |
| `log_end(path)` | `End` event |
| `log_state_update(path, state)` | `StateUpdate` event |
| `log_checkpoint_id(path, id)` | `NextCheckpoint` event |
| `human_interaction(ty, ...)` | HITL interrupt (via `run_graph`, not drainer) |

The type parameter `H` lets workflows define their own HITL
without touching the core types.

### EventHandler

An ABC for **domain-specific** custom events emitted by agent tools
via `get_stream_writer()`:

```python
class EventHandler(ABC):
    async def handle_event(
        self, payload: dict, path: list[str], checkpoint_id: str
    ) -> None: ...
```

The drainer calls this for every `CustomUpdate` event.  The handler
casts the untyped `payload` dict to a domain-specific discriminated
union type for type-safe routing.  `NullEventHandler` is the
default no-op.

**Why two interfaces?**  IOHandler is structural — every workflow
needs start/end/state/HITL handling, and the methods are stable.
EventHandler is an extension point — each workflow defines its own
custom event types without touching the core dispatch logic.

## HITL (Human-in-the-Loop)

HITL interrupts do **not** flow through the event queue.  Instead,
`run_graph()` bridges them directly:

1. LangGraph emits an `__interrupt__` in the update stream
2. `graph_runner.run_graph()` extracts the interrupt value
3. Calls `human_handler(value, state)` — a closure over
   `IOHandler.human_interaction()`
4. The handler blocks (e.g. waits on an `asyncio.Queue` fed by a
   TUI input widget)
5. The response is passed back as a `Command(resume=response)`

This is synchronous from the graph's perspective — execution pauses
until the human responds.  Meanwhile the drainer continues
processing other events from other concurrent agents.

## Multi-Agent Scoping

For multi-agent orchestration where N agents run in parallel, each
agent needs its own `with_handler` scope.  The orchestrator creates
a separate `(IOHandler, EventHandler)` pair per task, then wraps
each agent invocation in its own `with_handler`.  This gives each
agent its own `EventQueue` and drainer, so events from different
agents never interleave within a single handler.

The `TaskHandle` dataclass bundles a handler pair with lifecycle
callbacks (`on_start`, `on_done`, `on_error`), and a
`HandlerFactory` is the async callable that creates these bundles
on demand.  The orchestrator owns the concurrency model (semaphore,
task grouping); the factory owns the UI allocation (panels, status
rows).

## IDE Integration

The IO layer supports an optional IDE bridge (WebSocket connection
to a VS Code extension) for richer content viewing.  This is
implemented as a mixin rather than baked into the handler
infrastructure, since it is a UI concern orthogonal to event
dispatch.

### IDEBridge

A WebSocket client that sends JSON-RPC requests to the VS Code
extension.  Capabilities include showing files, opening diffs, and
previewing results with an accept/reject flow.  The bridge is
optional — `IDEBridge.connect()` returns `None` if no extension is
available.

### IDEContentMixin

A mixin for Textual `App` subclasses that provides reusable
infrastructure for content snapshot viewing:

- **Snapshot storage:** `_store_snapshot(label, content, filename)`
  stores content in memory and returns an integer ID.
- **Link rendering:** `_make_content_link_markup(snap_id, text)`
  returns Rich markup with a `@click` action.
  `_make_content_link_widget(snap_id, prefix, text)` returns a
  ready-to-mount `Static` widget.
- **Show action:** `action_show_content(snap_id)` — when the IDE
  bridge is available, sends the content to VS Code.  Otherwise,
  delegates to `_show_content_fallback()`, an override point for
  app-specific no-IDE behavior (e.g. mounting a temporary
  syntax-highlighted pane in a `ContentSwitcher`).

The mixin uses a `TYPE_CHECKING` conditional base class
(`App` for static analysis, `object` at runtime) so it can
reference `App` APIs for type checking without creating a runtime
dependency on the inheritance order.

### Event → Content Link Flow

Custom events carrying content (e.g. an updated spec or stub) are
surfaced as clickable links in the TUI:

1. Agent tool emits a custom event via `get_stream_writer()`
2. `EventHandler.handle_event()` receives the payload
3. Handler calls `app._store_snapshot(...)` to store the content
4. Handler mounts a widget from `app._make_content_link_widget(...)`
5. User clicks → `action_show_content(snap_id)` → IDE or fallback

This pattern decouples event handling (knowing *what* to show) from
content viewing (knowing *how* to show it).

## File Map

| File | Layer | Purpose |
|------|-------|---------|
| `events.py` | Core | Event dataclasses |
| `stream.py` | Core | `EventQueue` |
| `protocol.py` | Core | `IOHandler[H, P]` protocol |
| `event_handler.py` | Core | `EventHandler` ABC |
| `graph_runner.py` | Core | Low-level graph streaming, event emission |
| `context.py` | Core | `with_handler`, `run_graph` wrapper, `_queue_drainer`, nesting |
| `ide_content.py` | UI infra | `IDEContentMixin` — snapshots, clickable links |
| `ide_bridge.py` | UI infra | `IDEBridge` — WebSocket to VS Code extension |
| `message_renderer.py` | UI infra | Renders messages, manages nested widget containers |
| `rich_console.py` | UI | `BaseRichConsoleApp[H, P]` — base TUI for single-graph workflows |
| `codegen_rich.py` | UI | `CodeGenRichApp` — codegen TUI |
| `pipeline_app.py` | UI | `PipelineApp` — multi-agent pipeline TUI |

"""Phase 3 (refactored): per-task ``Task`` IOHandler + thin factory.

Each running task is its own ``IOHandler[None]`` + ``EventHandler``,
installed by ``with_handler`` for the task's scope. The framework
routes every IO event to the right ``Task`` instance by handler scope
— there's no path-based dispatch table.

:class:`AutoProveWebHandler` owns the run, ``make_handler``, and
run-level finalization (``finish`` / ``crashed``). Phase 4 swaps the
fake driver in :mod:`composer.web.mock_pipeline` for the real
``run_autoprove_pipeline`` call; this module doesn't change.

Path semantics inside a Task:

  - empty path → task root (``#log-{task_id}``); reserved for
    handler-internal emissions like pipeline-level errors.
  - ``len(path) == 1`` → top-level execution within the task scope.
    Registered as an alias of root (no ``<details>`` chrome — the
    task panel already serves that role); appends land directly in
    the panel.
  - ``len(path) >= 2`` → real nested workflow. Opens a ``<details>``
    whose parent is the depth-(N-1) selector.

Path entries are langgraph thread IDs; the framework stacks them
outermost-to-innermost. We treat them as opaque keys.
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Never, cast, override

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, ToolCall

from composer.io.event_handler import NullEventHandler
from composer.io.multi_job import TaskHandle, TaskInfo
from composer.spec.source.prover import ProverEvents
from composer.ui.autoprove_app import AutoProvePhase
from composer.ui.tool_display import GroupedTool, ToolDisplayConfig
from composer.web.emit_tree import EmitTree
from composer.web.render import render_fragment, render_markdown
from composer.web.runs import (
    Phase,
    RunState,
    STATUS_ICON,
    Status,
)


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AutoProvePhase ↔ composer.web.runs.Phase
# ---------------------------------------------------------------------------

# We keep ``Phase`` local to ``composer.web`` rather than importing
# :class:`AutoProvePhase` directly: the canonical enum lives under
# ``composer.ui.autoprove_app`` whose module-level ``textual`` imports
# we don't want to drag into the web tree. A 6-line mapping is the
# cost; if AutoProvePhase ever moves to a leaner module, drop the
# mapping and import directly.
_AUTOPROVE_TO_WEB: dict[AutoProvePhase, Phase] = {
    AutoProvePhase.HARNESS:            Phase.HARNESS,
    AutoProvePhase.SUMMARIES:          Phase.SUMMARIES,
    AutoProvePhase.INVARIANTS:         Phase.INVARIANTS,
    AutoProvePhase.COMPONENT_ANALYSIS: Phase.COMPONENT_ANALYSIS,
    AutoProvePhase.BUG_ANALYSIS:       Phase.BUG_ANALYSIS,
    AutoProvePhase.CVL_GEN:            Phase.CVL_GEN,
}


# ---------------------------------------------------------------------------
# Conversation stub
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _conversation_501(_initial: Any) -> AsyncIterator[Never]:
    """v1 stub. The web frontend has no chat UI for HITL refinement
    conversations; surface that loudly rather than silently no-op.
    Phase 4+ will replace this when the chat side ships."""
    raise NotImplementedError(
        "HITL conversation refinement is not implemented in the web "
        "frontend (v1 ships hands-off only)."
    )
    yield  # type: ignore[unreachable]  # satisfies asynccontextmanager protocol


# ---------------------------------------------------------------------------
# Task — IOHandler[None] + EventHandler for one task scope
# ---------------------------------------------------------------------------

class Task(NullEventHandler):
    """One row in the summary aside, one panel on the right, and the
    handler installed by ``with_handler`` for that task's scope.

    Holds its own :class:`EmitTree` rooted at ``#log-{task_id}``. Every
    method that mutates display state pushes wire ops through
    ``self.run.push``; ``RunState`` records them for SSE replay and
    broadcasts to live subscribers.
    """

    def __init__(
        self,
        run: RunState,
        task_id: str,
        label: str,
        phase: Phase,
    ) -> None:
        super().__init__()
        self.run = run
        self.task_id = task_id
        self.label = label
        self.phase = phase
        self.status: Status = Status.PENDING
        self.emit_tree = EmitTree(
            root_selector=f"#log-{task_id}",
            id_prefix=f"task-{task_id}",
        )
        # Reuses the global tool-display registry populated by
        # ``@tool_display`` decorators at module import — same registry
        # the TUI reads. Stateless w.r.t. the run; safe to instantiate
        # per-Task. Lookups go through ``_find_formatter`` which checks
        # local dict → ContextVar scope → ``_ns_global_tools`` →
        # ``_graphcore_global_tools``.
        self.tool_config = ToolDisplayConfig()
        # Grouping state — see ``_emit_grouped_call``. Mirrors the
        # ``_last_tool_*`` fields on the TUI's ``ToolCallRenderer``.
        self._last_group_id: str | None = None
        self._last_group_items: list[str] = []
        self._last_group_row_id: str | None = None
        # Counter for unique grouped-row DOM ids.
        self._group_counter: int = 0

    # ── registration & status -----------------------------------------

    def register(self) -> None:
        """Emit the phase section (if first task in phase), the row in
        the summary aside, and the panel skeleton in the right pane.
        Called once by ``AutoProveWebHandler.make_handler``."""
        if self.phase not in self.run.phase_seen:
            self.run.phase_seen.add(self.phase)
            self.run.push(
                "append", "#summary",
                render_fragment(
                    "fragments/phase_section.j2",
                    phase_id=self.phase.name,
                    phase_label=self.phase.value,
                ),
            )
        self.run.push("append", f"#rows-{self.phase.name}", self._row_html())
        self.run.push(
            "append", "#panels",
            render_fragment(
                "fragments/task_panel.j2",
                task_id=self.task_id,
                label=self.label,
                phase_label=self.phase.value,
            ),
        )

    def _set_status(self, status: Status) -> None:
        self.status = status
        self.run.push("replace", f"#row-{self.task_id}", self._row_html())

    def _row_html(self) -> str:
        return render_fragment(
            "fragments/task_row.j2",
            task_id=self.task_id,
            label=self.label,
            status=self.status.value,
            icon=STATUS_ICON[self.status],
        )

    # ── lifecycle (TaskHandle callbacks) ------------------------------

    def on_start(self) -> None:
        self._set_status(Status.RUNNING)

    def on_done(self) -> None:
        self._set_status(Status.DONE)

    async def on_error(self, exc: Exception, tb: str) -> None:
        # Pipeline-level error: emit at task root (empty path) — the one
        # case we deliberately bypass the EmitTree, since the error
        # isn't bound to any specific nested workflow inside the task.
        self._log_at_root("error", f"{type(exc).__name__}: {exc}")
        for line in tb.rstrip().splitlines():
            if line:
                self._log_at_root("error", line)
        self._set_status(Status.ERROR)

    def _log_at_root(self, kind: str, text: str) -> None:
        self.run.push(
            "append", f"#log-{self.task_id}",
            render_fragment("fragments/log_entry.j2", kind=kind, text=text),
        )

    # ── IOHandler[None] protocol --------------------------------------

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        # Checkpoint noise — the user doesn't need the langgraph
        # internals on the run page.
        return None

    async def log_start(
        self, *, path: list[str], description: str, tool_id: str | None,
    ) -> None:
        if not path:
            return
        # Crossing a nest boundary closes any open tool-call group —
        # the next call inside the new context starts fresh.
        self._reset_group()
        nest = tuple(path)
        if len(nest) == 1:
            # Top-level execution within this task scope: register the
            # path so deeper children can find their parent, but don't
            # emit a <details> wrapper — the task panel already serves
            # that role.
            self.emit_tree.register_alias(nest)
            return
        op = self.emit_tree.enter(nest, description)
        self.run.push(op.op, op.sel, op.html)

    async def log_end(self, path: list[str]) -> None:
        if not path:
            return
        # Same reasoning as log_start — the prior nest's grouping state
        # doesn't carry across to whatever follows in the parent.
        self._reset_group()
        # ``status="done"`` is harmless on aliases — leave() ignores it
        # silently for them — so we don't need depth-aware branching.
        for op in self.emit_tree.leave(tuple(path), status="done"):
            self.run.push(op.op, op.sel, op.html)

    async def log_state_update(self, path: list[str], st: dict) -> None:
        """Walk the state-update message lists and emit one rendered
        item per message, dispatching by type:

          - SystemMessage                 → ``_emit_prompt_block`` (closed
            <details>; long set-once context)
          - HumanMessage                  → ``_emit_prompt_block`` (closed
            <details>; the initial prompt)
          - AIMessage with ``tool_calls`` → one ``_emit_tool_call`` per call
            (group-aware: file/memory tools fold into a replaceable row)
          - ToolMessage                   → ``_emit_tool_result`` (<details>)
          - AIMessage with string content → ``_emit_ai_text``
            (markdown-rendered; resets grouping)
          - AIMessage with list content   → walked block-by-block:
            ``thinking`` → ``_emit_thinking`` (closed-by-default details);
            ``text``     → ``_emit_ai_text``;
            ``tool_use`` → skipped (already covered by ``msg.tool_calls``)
        """
        nest = tuple(path)
        for _node_name, update in st.items():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []):
                if isinstance(msg, ToolMessage):
                    self._emit_tool_result(nest, msg)
                    continue
                # System / initial-human prompts get their own
                # closed-by-default <details> — they're set-once context
                # that's typically long, and rendering them inline as AI
                # messages would dominate the panel.
                if isinstance(msg, SystemMessage):
                    self._emit_prompt_block(nest, "system", "System prompt", msg)
                    continue
                if isinstance(msg, HumanMessage):
                    self._emit_prompt_block(nest, "human", "Initial prompt", msg)
                    continue
                if not isinstance(msg, AIMessage):
                    continue
                # Tool calls live on ``msg.tool_calls`` (the parsed list)
                # rather than embedded in ``content``; emit them here
                # and skip ``tool_use`` blocks during the content walk
                # so we don't double-render.
                tc = msg.tool_calls
                if tc:
                    for call in tc:
                        self._emit_tool_call(nest, call)
                # Walk content. Anthropic extended-thinking emits a
                # list of typed blocks; pre-thinking models emit a
                # plain string. Handle both.
                content = msg.content
                if isinstance(content, str):
                    if content.strip():
                        self._emit_ai_text(nest, content.strip())
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str) and (stripped := block.strip()):
                            self._emit_ai_text(nest, stripped)
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "thinking":
                            txt = block.get("thinking", "")
                            if isinstance(txt, str) and txt.strip():
                                self._emit_thinking(nest, txt.strip())
                        elif btype == "text":
                            txt = block.get("text", "")
                            if isinstance(txt, str) and txt.strip():
                                self._emit_ai_text(nest, txt.strip())
                        # ``tool_use`` blocks: handled via msg.tool_calls
                        # already; skip to avoid double-emission.

    async def human_interaction(
        self, ty: None, debug_thunk: Callable[[], None],
    ) -> str:
        # Autoprove pipeline doesn't normally produce HITL interrupts
        # (``--interactive`` is v1 OOS); if one arrives, surface
        # clearly rather than hang the stream.
        raise RuntimeError(
            "Unexpected HITL interrupt in autoprove web handler"
        )

    # ── EventHandler protocol -----------------------------------------

    @override
    async def handle_event(
        self, payload: dict, path: list[str], checkpoint_id: str,
    ) -> None:
        nest = tuple(path)
        d = cast(ProverEvents, payload)
        match d["type"]:
            case "prover_output" | "cloud_polling":
                # Firehose noise — drop.
                pass
            case "prover_run":
                self._log_event(nest, "tool", "prover started")
            case "prover_result":
                self._log_event(nest, "ai", "prover finished")
            case "rule_analysis":
                self._log_event(nest, "ai", f"rule analysis → {d['rule']}")
            case "cex_analysis":
                self._log_event(nest, "ai", f"cex analysis → {d['rule_name']}")
        return await super().handle_event(payload, path, checkpoint_id)

    def _log_event(self, nest: tuple[str, ...], kind: str, text: str) -> None:
        entry = render_fragment(
            "fragments/log_entry.j2", kind=kind, text=text,
        )
        op = self.emit_tree.append_at(nest, entry)
        self.run.push(op.op, op.sel, op.html)

    # ── tool-call / result / AI-text emission ------------------------

    def _emit_tool_call(self, nest: tuple[str, ...], call: ToolCall) -> None:
        """Dispatch one tool call to either grouped or solo emission."""
        name = call["name"]
        args = call["args"]
        grouped = self.tool_config.get_group(name)
        if grouped is not None:
            self._emit_grouped_call(nest, name, args, grouped)
        else:
            self._emit_solo_call(nest, name, args)

    def _emit_grouped_call(
        self,
        nest: tuple[str, ...],
        name: str,
        args: dict,
        grouped: GroupedTool,
    ) -> None:
        """Fold this call into the current group row, or start a new
        one. Mirrors :class:`ToolCallRenderer`'s grouped-collapsing
        logic — same group_id extends the existing row in-place via
        ``replace``; mismatched group_id resets and appends fresh."""
        raw = grouped.extract_group_items(args)
        new_items = [raw] if isinstance(raw, str) else list(raw)

        if (
            self._last_group_id == grouped.group_id
            and self._last_group_row_id is not None
        ):
            # Extend the existing row.
            self._last_group_items.extend(new_items)
            text = grouped.render_group(self._last_group_items)
            html = render_fragment(
                "fragments/tool_group.j2",
                row_id=self._last_group_row_id, text=text,
            )
            self.run.push(
                "replace", f"#{self._last_group_row_id}", html,
            )
            return

        # Start a fresh group row.
        self._group_counter += 1
        row_id = f"task-{self.task_id}-tg-{self._group_counter}"
        self._last_group_id = grouped.group_id
        self._last_group_items = new_items
        self._last_group_row_id = row_id
        text = grouped.render_group(self._last_group_items)
        html = render_fragment(
            "fragments/tool_group.j2", row_id=row_id, text=text,
        )
        op = self.emit_tree.append_at(nest, html)
        self.run.push(op.op, op.sel, op.html)

    def _emit_solo_call(
        self, nest: tuple[str, ...], name: str, args: dict,
    ) -> None:
        """A non-grouped tool call. Resets group state — any prior
        group is conceptually closed by a new non-grouped call —
        then emits a single tool-call line."""
        self._reset_group()
        label = self.tool_config.format_tool_call(name, args)
        html = render_fragment("fragments/tool_call.j2", text=label)
        op = self.emit_tree.append_at(nest, html)
        self.run.push(op.op, op.sel, op.html)

    def _emit_tool_result(
        self, nest: tuple[str, ...], msg: ToolMessage,
    ) -> None:
        """Render a tool result as a ``<details>`` collapsible. Tool
        results don't reset grouping (they're annotations for already-
        emitted calls, not new turns); :class:`ToolDisplayConfig` may
        suppress the result entirely (returns ``None``) — typical for
        ack-style "Success" responses."""
        name = getattr(msg, "name", None) or "Tool result"
        formatted = self.tool_config.format_result(name, msg)
        if formatted is None:
            return
        label, body = formatted
        html = render_fragment(
            "fragments/tool_result.j2", label=label, body=body,
        )
        op = self.emit_tree.append_at(nest, html)
        self.run.push(op.op, op.sel, op.html)

    def _emit_ai_text(self, nest: tuple[str, ...], text: str) -> None:
        """Render AI / sub-agent text content. Resets group state (a
        textual interjection breaks any in-flight tool-call group).

        Goes through markdown-it for safe HTML rendering — sub-agents
        (CVL research especially) emit prose with code blocks, lists,
        headings; rendering them as plain text loses readability, and
        injecting raw markdown leaves the user staring at literal
        asterisks and backticks."""
        self._reset_group()
        html_body = render_markdown(text)
        entry = render_fragment(
            "fragments/ai_message.j2", html=html_body,
        )
        op = self.emit_tree.append_at(nest, entry)
        self.run.push(op.op, op.sel, op.html)

    def _emit_thinking(self, nest: tuple[str, ...], text: str) -> None:
        """Render an extended-thinking block as a closed-by-default
        ``<details>``. Doesn't reset grouping — thinking is internal
        narration, not a turn boundary; surrounding tool-call groups
        should still fold when the model resumes calling tools."""
        html_body = render_markdown(text)
        entry = render_fragment(
            "fragments/thinking_block.j2", html=html_body,
        )
        op = self.emit_tree.append_at(nest, entry)
        self.run.push(op.op, op.sel, op.html)

    def _emit_prompt_block(
        self,
        nest: tuple[str, ...],
        kind: str,
        label: str,
        msg: object,
    ) -> None:
        """Render a system / initial-human prompt as a closed-by-default
        ``<details>``. Resets grouping (any in-flight tool-call group
        is conceptually closed by a new context block).

        Handles both string content and list-typed content (Anthropic's
        cached system-prompt shape uses a list of typed blocks; we pull
        out the ``text`` blocks and concatenate)."""
        self._reset_group()
        content = getattr(msg, "content", None)
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    t = block.get("text", "")
                    if isinstance(t, str):
                        parts.append(t)
            text = "\n\n".join(parts)
        if not text.strip():
            return
        html_body = render_markdown(text)
        entry = render_fragment(
            "fragments/prompt_block.j2",
            kind=kind, label=label, html=html_body,
        )
        op = self.emit_tree.append_at(nest, entry)
        self.run.push(op.op, op.sel, op.html)

    def _reset_group(self) -> None:
        """Forget the current group so the next call (even if it's the
        same group_id) starts a fresh row."""
        self._last_group_id = None
        self._last_group_items = []
        self._last_group_row_id = None


# ---------------------------------------------------------------------------
# AutoProveWebHandler — factory + run-level finalization
# ---------------------------------------------------------------------------

class AutoProveWebHandler:
    """``HandlerFactory[AutoProvePhase, None]`` for the autoprove
    pipeline, plus run-level wrap-up.

    Stateless beyond ``self.run`` — every per-task concern lives on the
    :class:`Task` instance returned via ``make_handler``. Phase 4
    passes ``handler.make_handler`` as the ``handler_factory`` arg to
    ``run_autoprove_pipeline``; nothing else changes."""

    def __init__(self, run: RunState) -> None:
        self.run = run

    async def make_handler(
        self, info: TaskInfo[AutoProvePhase],
    ) -> TaskHandle[None]:
        task = Task(
            run=self.run,
            task_id=info.task_id,
            label=info.label,
            phase=_AUTOPROVE_TO_WEB[info.phase],
        )
        self.run.tasks[task.task_id] = task
        task.register()
        return TaskHandle(
            handler=task,
            event_handler=task,
            on_start=task.on_start,
            on_done=task.on_done,
            on_error=task.on_error,
            conversation_provider=_conversation_501,
        )

    # ── run-level finalization ----------------------------------------

    def finish(self) -> None:
        """Emit the final run-status pill and outputs section.

        Inspects ``run.tasks`` for any errored tasks to choose the
        run-level status. ``run.output_files`` is populated by the
        caller before invoking ``finish`` (Phase 4 wires the real
        pipeline result; the fake driver writes mock specs)."""
        any_error = any(
            t.status is Status.ERROR for t in self.run.tasks.values()
        )
        if any_error:
            self._finish_with("Completed with failures", "error")
        else:
            self._finish_with("Run complete", "done")

    def crashed(self, exc: Exception) -> None:
        """Mark the run as crashed at the pipeline level (not a single
        task's failure). The full traceback goes to server logs; the
        user sees only the short status text."""
        tb = "".join(traceback.format_exception(exc))
        _logger.error("Run %s crashed:\n%s", self.run.run_id, tb)
        self._finish_with(
            f"Pipeline crashed: {type(exc).__name__}: {exc}",
            "error",
            files=[],
        )

    def _finish_with(
        self, text: str, css_class: str, *, files: list[dict] | None = None,
    ) -> None:
        if files is None:
            files = self.run.output_files
        self.run.final_status_text = text
        self.run.final_status_class = css_class
        self.run.output_files = files
        self.run.done = True
        self.run.push(
            "replace", "#run-status",
            render_fragment(
                "fragments/run_status.j2",
                status_text=text, status_class=css_class,
            ),
        )
        self.run.push(
            "inner", "#outputs",
            render_fragment("fragments/output_files.j2", files=files),
        )

"""
Console-mode handler for the auto-prove source-spec pipeline.

Create one ``AutoProveConsoleHandler``, then pass ``handler.make_handler`` as
the ``handler_factory`` argument to ``run_autoprove_pipeline``.  The same
handler instance is reused across all phases so that path descriptions
accumulate correctly across the whole pipeline run.

Log format:

- Phase boundaries:   ``─────`` header printed by ``on_start``
- Start/end events:   ``[Foo / Bar] start``  /  ``[Foo / Bar] end``
- State updates:      ``[Foo / Bar] at node: <node>``
                      ``[Foo / Bar] at node: <node>; tool calls: [a, b]``

The path label is built lazily from the ``description`` values received in
``log_start`` calls.  Each thread ID maps to its description; the label for a
path is all descriptions joined with `` / ``.
"""

from typing import Callable, override, cast
import sys

from composer.spec.source.prover import ProverEvents
from composer.ui.autoprove_app import AutoProvePhase
from composer.io.event_handler import NullEventHandler
from composer.ui.multi_job_app import TaskHandle, TaskInfo


class AutoProveConsoleHandler(NullEventHandler):
    """``IOHandler[Never]`` + ``HandlerFactory`` for the auto-prove pipeline.

    One instance spans the whole pipeline run.  ``make_handler`` is passed as
    the ``handler_factory`` argument; it returns ``handler=self`` each time so
    path descriptions accumulated by one phase are visible to all later phases.
    """

    def __init__(self) -> None:
        self._descriptions: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, path: list[str]) -> str:
        return " / ".join(self._descriptions.get(tid, tid) for tid in path)

    # ------------------------------------------------------------------
    # IOHandler protocol
    # ------------------------------------------------------------------

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        pass  # checkpoint noise suppressed

    async def log_start(
        self, *, path: list[str], description: str, tool_id: str | None
    ) -> None:
        self._descriptions[path[-1]] = description
        label = self._label(path)
        suffix = f"  (via tool: {tool_id})" if tool_id else ""
        print(f"[{label}] start{suffix}")

    async def log_end(self, path: list[str]) -> None:
        print(f"[{self._label(path)}] end")

    async def log_state_update(self, path: list[str], st: dict) -> None:
        label = self._label(path)
        for node_name, update in st.items():
            if not isinstance(update, dict):
                continue
            tool_names: list[str] = []
            for msg in update.get("messages", []):
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    tool_names.extend(c["name"] for c in tc)
            if tool_names:
                names = ", ".join(tool_names)
                print(f"[{label}] at node: {node_name}; tool calls: [{names}]")
            else:
                print(f"[{label}] at node: {node_name}")

    async def human_interaction(
        self, ty: None, debug_thunk: Callable[[], None]
    ) -> str:
        raise RuntimeError(
            "Unexpected HITL interrupt in auto-prove console handler"
        )
    
    @override
    def handle_event(self, payload: dict, path: list[str], checkpoint_id: str):
        d = cast(ProverEvents, payload)
        match d["type"]:
            case "prover_output":
                pass
            case "cloud_polling":
                pass
            case "prover_run":
                print(f"[{self._label(path)}]: prover start")
            case "prover_result":
                print(f"[{self._label(path)}]; prover complete")
            case "rule_analysis":
                print(f"[{self._label(path)}]: rule analysis complete -> {d['rule']}")
            case "cex_analysis":
                print(f"[{self._label(path)}]: rule analysis start -> {d['rule_name']}")
        return super().handle_event(payload, path, checkpoint_id)

    # ------------------------------------------------------------------
    # HandlerFactory
    # ------------------------------------------------------------------

    async def make_handler(self, info: TaskInfo[AutoProvePhase]) -> TaskHandle[None]:
        """Return a ``TaskHandle`` that routes all events back to *self*.

        Pass this bound method as ``handler_factory`` to
        ``run_autoprove_pipeline``.
        """
        async def _on_error(exc: Exception, tb: str) -> None:
            print(
                f"\n[ERROR] {info.label}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(tb, file=sys.stderr)

        return TaskHandle(
            handler=self,
            event_handler=self,
            on_start=lambda: print(
                f"\n{'─' * 60}\nPhase: {info.label}\n{'─' * 60}"
            ),
            on_done=lambda: print(f"[{info.label}] ✓ done"),
            on_error=_on_error,
        )

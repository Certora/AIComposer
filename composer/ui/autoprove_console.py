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

from typing import override, cast
import sys

from composer.spec.source.prover import ProverEvents
from composer.ui.autoprove_app import AutoProvePhase
from composer.io.event_handler import NullEventHandler
from composer.ui.multi_job_app import TaskHandle, TaskInfo
from composer.ui.simple_console_handler import SimpleConsoleHandler


class AutoProveConsoleHandler(SimpleConsoleHandler, NullEventHandler):
    """``IOHandler[Never]`` + ``HandlerFactory`` for the auto-prove pipeline.

    One instance spans the whole pipeline run.  ``make_handler`` is passed as
    the ``handler_factory`` argument; it returns ``handler=self`` each time so
    path descriptions accumulated by one phase are visible to all later phases.
    """

    def __init__(self) -> None:
        super().__init__()
        self._descriptions: dict[str, str] = {}

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

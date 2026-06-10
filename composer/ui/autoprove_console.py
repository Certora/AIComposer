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

from composer.spec.source.prover import ProverEvents
from composer.spec.source.autosetup import AutoSetupEvents
from composer.ui.autoprove_app import AutoProvePhase
from composer.ui.multi_console_handler import MultiJobConsoleHandler


class AutoProveConsoleHandler(MultiJobConsoleHandler[AutoProvePhase]):
    """``IOHandler[Never]`` + ``HandlerFactory`` for the auto-prove pipeline.

    One instance spans the whole pipeline run.  ``make_handler`` is passed as
    the ``handler_factory`` argument; it returns ``handler=self`` each time so
    path descriptions accumulated by one phase are visible to all later phases.
    """

    @override
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        d = cast(ProverEvents, payload)
        match d["type"]:
            case "prover_output":
                pass
            case "cloud_polling":
                pass
            case "prover_run":
                self._output(f"[{self._label(path)}]: prover start")
            case "prover_link":
                self._output(f"[{self._label(path)}]: prover link -> {d['link']}")
            case "prover_result":
                self._output(f"[{self._label(path)}]; prover complete")
            case "rule_analysis":
                self._output(f"[{self._label(path)}]: rule analysis complete -> {d['rule']}")
            case "cex_analysis":
                self._output(f"[{self._label(path)}]: rule analysis start -> {d['rule_name']}")

    @override
    async def handle_progress_event(self, payload: dict) -> None:
        # AutoSetup is an external subprocess, not a LangGraph agent, so it
        # never trips the graph-level ``[<phase>] start`` log the other phases
        # get. Surface its lifecycle here for parity. Per-line subprocess stdout
        # is suppressed, mirroring how ``prover_output`` is dropped above.
        evt = cast(AutoSetupEvents, payload)
        match evt["type"]:
            case "auto_setup_start":
                self._output("[AutoSetup] start")
            case "auto_setup_output":
                pass
            case "auto_setup_complete":
                self._output(f"[AutoSetup] complete (return code {evt['return_code']})")

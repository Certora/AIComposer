"""Fancy console handler for the codegen workflow.

Drop-in replacement for ``ConsoleHandler`` that styles its log output
in the same shape ``AutoProveConsoleHandler`` uses — accumulated path
labels, suppressed checkpoint noise, structured prover lifecycle
lines. Reuses the existing ``ConsoleHandler``'s human-interaction
prompts and final-output rendering, since those haven't changed.
"""

from typing import override

from composer.diagnostics.stream import ProgressUpdate
from composer.io.protocol import WorkflowPurpose
from composer.ui.console import ConsoleHandler


class CodegenConsoleHandler(ConsoleHandler):
    """``CodeGenIOHandler`` with autoprove-style logging.

    Inherits ``human_interaction``, ``output``, and ``show_error`` from
    the legacy ``ConsoleHandler``. Overrides the noisy logging methods
    (checkpoint id, start, end, state update, progress) to mirror
    ``AutoProveConsoleHandler``'s output shape: per-thread descriptions
    accumulated into a path label, structured prover lifecycle markers
    instead of per-line stdout streaming.
    """

    def __init__(self, capture_prover_output: bool = False) -> None:
        super().__init__(capture_prover_output=capture_prover_output)
        self._descriptions: dict[str, str] = {}

    def _label(self, path: list[str]) -> str:
        return " / ".join(self._descriptions.get(tid, tid) for tid in path)

    @override
    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        return  # suppressed

    @override
    async def log_start(
        self, *, path: list[str], description: str, tool_id: str | None
    ) -> None:
        self._descriptions[path[-1]] = description
        suffix = f"  (via tool: {tool_id})" if tool_id else ""
        print(f"[{self._label(path)}] start{suffix}")

    @override
    async def log_end(self, path: list[str]) -> None:
        print(f"[{self._label(path)}] end")

    @override
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

    @override
    async def log_workflow_thread(self, purpose: WorkflowPurpose, thread_id: str) -> None:
        print(f"[{purpose.value}] thread: {thread_id}")

    @override
    async def progress_update(self, path: list[str], upd: ProgressUpdate) -> None:
        label = self._label(path)
        match upd["type"]:
            case "prover_output" | "cloud_polling":
                return
            case "prover_run":
                print(f"[{label}]: prover start")
            case "prover_result":
                print(f"[{label}]: prover complete")
            case "rule_analysis":
                print(f"[{label}]: rule analysis -> {upd.get('rule', '?')}")
            case "cex_analysis":
                print(f"[{label}]: cex analysis -> {upd.get('rule_name', '?')}")
            case "summarization_notice":
                print(f"[{label}]: summarization")

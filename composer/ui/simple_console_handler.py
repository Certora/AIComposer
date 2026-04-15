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

from typing import Callable

class SimpleConsoleHandler():
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

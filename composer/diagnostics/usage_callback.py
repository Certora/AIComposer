"""LangChain callback that accumulates per-call LLM token usage into the active run.

Attached at model construction (:func:`composer.workflow.services.create_llm_base`)
so it fires for *every* ``invoke`` / ``ainvoke`` through the model — including
``.bind_tools()`` derivatives and out-of-graph side-calls (prover counterexample
analysis, interactive refinement) that never reach the graph's ``StateUpdate`` stream
and so are invisible to the TUI token bar.

``run_inline = True`` keeps dispatch on the event-loop thread (see
``langchain_core.callbacks.manager._ahandle_event_for_handler``): the active-task
context var read by :meth:`RunSummary.record_token_usage` stays visible and the shared
counters are mutated single-threaded (no race). Compatible across the pinned
``langchain_core >=1.2,<1.3.3`` range — uses only the long-stable sync
``BaseCallbackHandler.on_llm_end`` surface.
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from graphcore.utils import get_token_usage
from composer.diagnostics.timing import get_run_summary


class UsageCallback(BaseCallbackHandler):
    """Records each LLM response's token usage into the active ``RunSummary``."""

    # Run on the calling event-loop thread instead of a thread-pool executor.
    # In async runs LangChain otherwise offloads sync handlers to an executor
    # (see manager._ahandle_event_for_handler): inline keeps the active-task
    # context var read by record_token_usage visible, and keeps the shared
    # RunSummary counter mutations single-threaded (no cross-thread race).
    run_inline = True

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except IndexError:
            return
        if not isinstance(generation, ChatGeneration):
            return
        msg = generation.message
        if isinstance(msg, AIMessage):
            # get_run_summary() returns an inert throwaway outside a run, so this is
            # a no-op when no autoprove run is active (e.g. ad-hoc model use).
            get_run_summary().record_token_usage(get_token_usage(msg))

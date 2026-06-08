"""Dump-everything-to-stdout handler factory for foundry-pipeline debugging.

Intentionally ugly. The autoprove console handler is shaped around prover
events / autoprove phases and wouldn't render the foundry workflow's
state transitions usefully. Until there's a real foundry UI, this thing
just prints every callback the graph runner invokes.

DO NOT use this for anything other than spot-debugging — there is no
buffering, no formatting, no anything. It will spam.
"""

import sys
from typing import Any

from rich.console import RenderableType

from composer.io.conversation import (
    ConversationClient, ConversationContextProvider,
)
from composer.io.event_handler import EventHandler
from composer.io.multi_job import HandlerFactory, TaskHandle, TaskInfo
from composer.ui.autoprove_app import AutoProvePhase


def _emit(prefix: str, **fields: Any) -> None:
    pieces = [prefix] + [f"{k}={v!r}" for k, v in fields.items()]
    print(" | ".join(pieces), file=sys.stderr, flush=True)


class _NoisyIO:
    def __init__(self, task_id: str, label: str):
        self.task_id = task_id
        self.label = label

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        # No-op — even for a noisy debug handler this is too much. Every
        # node transition emits one of these.
        pass

    async def log_state_update(self, path: list[str], st: dict) -> None:
        # The full state dict is huge — print keys + a one-line repr per key
        # so the eye can scan, and dump the full thing under a marker.
        _emit(
            "[STATE]",
            task=self.task_id, path="/".join(path),
            keys=sorted(st.keys()),
        )
        for k, v in st.items():
            r = repr(v)
            if len(r) > 300:
                r = r[:297] + "..."
            print(f"    {k} = {r}", file=sys.stderr, flush=True)

    async def log_start(
        self, *, path: list[str], description: str, tool_id: str | None,
    ) -> None:
        _emit(
            "[START]",
            task=self.task_id, path="/".join(path),
            desc=description, tool_id=tool_id,
        )

    async def log_end(self, path: list[str]) -> None:
        _emit("[END]", task=self.task_id, path="/".join(path))

    async def human_interaction(self, ty: Any, debug_thunk: Any) -> str:
        raise RuntimeError(
            f"HITL interrupt fired in foundry debug pipeline "
            f"(task={self.task_id}, ty={type(ty).__name__}); the foundry "
            "workflow shouldn't be invoking human_interaction at all"
        )


class _NoisyEvents(EventHandler):
    def __init__(self, task_id: str):
        self.task_id = task_id

    async def handle_event(
        self, payload: dict, path: list[str], checkpoint_id: str,
    ) -> None:
        _emit(
            "[EVENT]",
            task=self.task_id, path="/".join(path),
            checkpoint=checkpoint_id, payload=payload,
        )

    async def handle_progress_event(self, payload: dict) -> None:
        _emit("[PROGRESS]", task=self.task_id, payload=payload)


class _RaisingConvCM:
    """Async context manager whose ``__aenter__`` raises. Used in place of
    a real conversation provider so the foundry workflow can't quietly
    open a conversation channel without us noticing."""

    def __init__(self, task_id: str):
        self.task_id = task_id

    async def __aenter__(self) -> ConversationClient:
        raise RuntimeError(
            f"ConversationContextProvider opened in foundry debug pipeline "
            f"(task={self.task_id}); the foundry workflow shouldn't be using "
            "the conversation channel"
        )

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _noisy_conv_provider(task_id: str) -> ConversationContextProvider:
    def provider(_initial: RenderableType) -> _RaisingConvCM:
        return _RaisingConvCM(task_id)
    return provider


class NoisyHandlerFactory:
    """A ``HandlerFactory[AutoProvePhase, None]`` that wires every task to
    the noisy stdout handlers above."""

    async def make_handler(
        self, info: TaskInfo[AutoProvePhase],
    ) -> TaskHandle[None]:
        async def on_error(exc: Exception, message: str) -> None:
            _emit(
                "[ERROR]",
                task=info.task_id, phase=info.phase.name, label=info.label,
                exc=f"{type(exc).__name__}: {exc}", msg=message,
            )

        def on_start() -> None:
            _emit(
                "[TASK-START]",
                task=info.task_id, phase=info.phase.name, label=info.label,
            )

        def on_done() -> None:
            _emit(
                "[TASK-DONE]",
                task=info.task_id, phase=info.phase.name, label=info.label,
            )

        return TaskHandle[None](
            handler=_NoisyIO(info.task_id, info.label),
            event_handler=_NoisyEvents(info.task_id),
            conversation_provider=_noisy_conv_provider(info.task_id),
            on_error=on_error,
            on_start=on_start,
            on_done=on_done,
        )


def noisy_handler_factory() -> HandlerFactory[AutoProvePhase, None]:
    """Return a fresh noisy factory. Use this as the second arg to
    ``run_foundry_pipeline``."""
    return NoisyHandlerFactory().make_handler

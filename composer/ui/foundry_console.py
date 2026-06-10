"""Console-mode handler for the foundry test-generation pipeline.

Create one ``FoundryConsoleHandler``, then pass ``handler.make_handler`` as
the ``handler_factory`` argument to ``run_foundry_pipeline``. See
``MultiJobConsoleHandler`` for the shared log format; the only
foundry-specific traffic is the ``forge_test_run`` summary event emitted
by ``ForgeTestTool`` after each ``forge test`` invocation.
"""

from typing import cast, override

from composer.foundry.pipeline import FoundryPhase
from composer.foundry.runner import ForgeTestRunEvent
from composer.ui.multi_console_handler import MultiJobConsoleHandler


class FoundryConsoleHandler(MultiJobConsoleHandler[FoundryPhase]):

    @override
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        evt = cast(ForgeTestRunEvent, payload)
        match evt["type"]:
            case "forge_test_run":
                self._output(f"[{self._label(path)}] forge test run:")
                self._output(evt["summary"])

    @override
    async def handle_progress_event(self, payload: dict) -> None:
        # The foundry pipeline emits no progress-channel events.
        pass

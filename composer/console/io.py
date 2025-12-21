from typing import Callable
from composer.core.io import ComposerIO
from composer.diagnostics.stream import ProgressUpdate
from composer.diagnostics.handlers import summarize_update, print_prover_updates
from composer.human.handlers import handle_human_interrupt
from composer.console.handler import DebugHandler

class ConsoleComposerIO(ComposerIO):
    def __init__(self):
        try:
            self.handler = DebugHandler()
        except ValueError:
            # Not in main thread, signals won't work
            self.handler = None

    def summarize_update(self, state: dict) -> None:
        summarize_update(state)

    def handle_user_update(self, update: ProgressUpdate) -> None:
        print_prover_updates(update)

    def next_checkpoint(self, checkpoint_id: str) -> None:
        print(f"current checkpoint: {checkpoint_id}")

    def log_thread_id(self, thread_id: str) -> None:
        print(f"Selected thread id: {thread_id}")

    def handle_human_interrupt(self, interrupt_data: dict, debug_thunk: Callable[[], None]) -> str:
        return handle_human_interrupt(interrupt_data, debug_thunk)

    def log_info(self, msg: str) -> None:
        print(msg)

    def log_error(self, msg: str) -> None:
        print(f"ERROR: {msg}")

    def check_for_interrupt(self) -> bool:
        return self.handler.requested if self.handler else False

    def reset_interrupt(self) -> None:
        if self.handler:
            self.handler.reset()


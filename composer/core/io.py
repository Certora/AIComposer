from typing import Protocol, Any, Callable
from composer.diagnostics.stream import ProgressUpdate

class ComposerIO(Protocol):
    def summarize_update(self, state: dict) -> None:
        """Summarize current state update (messages, VFS changes, etc.)"""
        ...

    def handle_user_update(self, update: ProgressUpdate) -> None:
        """Handle user-facing updates (prover runs, CEX analysis, etc.)"""
        ...

    def next_checkpoint(self, checkpoint_id: str) -> None:
        """Signal that a new checkpoint has been reached."""
        ...

    def log_thread_id(self, thread_id: str) -> None:
        """Log the thread ID used for the execution."""
        ...

    def handle_human_interrupt(self, interrupt_data: dict, debug_thunk: Callable[[], None]) -> str:
        """Handle human-in-the-loop interrupts and return user response."""
        ...

    def log_info(self, msg: str) -> None:
        """Log general information to the user."""
        ...

    def log_error(self, msg: str) -> None:
        """Log error information to the user."""
        ...

    def check_for_interrupt(self) -> bool:
        """Check if an out-of-band interrupt (e.g. Debug Console) has been requested."""
        ...

    def reset_interrupt(self) -> None:
        """Reset the interrupt request state."""
        ...


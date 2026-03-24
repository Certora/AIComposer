"""
Full-screen log viewer for Textual TUI applications.

Shows all log records from the ``composer`` logger namespace in real-time.
Press ``l`` to push the log screen, ``Escape`` to pop back.

Integration: add ``LogViewerMixin`` to your App's bases and call
``_init_log_viewer()`` from ``__init__``.
"""

import logging
import traceback
from collections import deque

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual import on
from textual.screen import Screen
from textual.widgets import RichLog

from rich.text import Text


# ---------------------------------------------------------------------------
# Textual message for live log delivery
# ---------------------------------------------------------------------------

class NewLogRecord(Message):
    """Posted by the handler when a new log record arrives."""

    def __init__(self, record: logging.LogRecord) -> None:
        super().__init__()
        self.record = record


# ---------------------------------------------------------------------------
# Logging handler
# ---------------------------------------------------------------------------

_LOG_STYLES: dict[int, str] = {
    logging.DEBUG:    "dim",
    logging.INFO:     "",
    logging.WARNING:  "yellow",
    logging.ERROR:    "bold red",
    logging.CRITICAL: "bold red reverse",
}

_MAX_BACKLOG = 5000


class _AppLogHandler(logging.Handler):
    """Captures log records into a bounded deque and posts to the Textual app.

    The deque is the source of truth for backlog.  Messages are posted
    for live updates; if no screen is listening they are silently dropped.
    """

    def __init__(self, app: App) -> None:
        super().__init__()
        self._app = app
        self.records: deque[logging.LogRecord] = deque(maxlen=_MAX_BACKLOG)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        try:
            self._app.post_message(NewLogRecord(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_record(record: logging.LogRecord) -> Text:
    """Format a LogRecord as styled Rich Text."""
    style = _LOG_STYLES.get(record.levelno, "")
    level = record.levelname.ljust(8)
    name = record.name
    msg = record.getMessage()
    text = Text()
    text.append(f"{level} ", style=("bold " + style) if style else "bold")
    text.append(f"{name}: ", style="dim")
    text.append(msg, style=style)
    if record.exc_info and record.exc_info[1] is not None:
        tb = "".join(traceback.format_exception(*record.exc_info))
        text.append(f"\n{tb}", style="red dim")
    return text


# ---------------------------------------------------------------------------
# Log screen
# ---------------------------------------------------------------------------

class LogScreen(Screen):
    """Full-screen scrollable log viewer."""

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=True),
    ]

    CSS = """
    LogScreen { background: $surface; }
    #log-viewer { height: 1fr; }
    """

    def __init__(self, handler: _AppLogHandler) -> None:
        super().__init__()
        self._handler = handler

    def compose(self) -> ComposeResult:
        yield RichLog(id="log-viewer", highlight=True, markup=False)

    def on_mount(self) -> None:
        log = self.query_one("#log-viewer", RichLog)
        for record in self._handler.records:
            log.write(_format_record(record))

    @on(NewLogRecord)
    def _append_log(self, message: NewLogRecord) -> None:
        log = self.query_one("#log-viewer", RichLog)
        log.write(_format_record(message.record))

    def action_dismiss_screen(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class LogViewerMixin(App):
    """Mixin for Textual Apps that adds a log viewer screen.

    Call ``_init_log_viewer()`` from your ``__init__``.
    Contributes ``BINDINGS`` for the ``l`` key via Textual's MRO-based
    binding merging.
    """

    _log_handler: _AppLogHandler

    BINDINGS = [
        Binding("l", "show_logs", "Logs", show=True),
    ]

    def _init_log_viewer(self) -> None:
        self._log_handler = _AppLogHandler(self)
        logger = logging.getLogger("composer")
        logger.addHandler(self._log_handler)
        # Ensure the logger passes all records through to handlers.
        # Individual handlers (e.g., a console handler on root) keep
        # their own level thresholds.
        logger.setLevel(logging.DEBUG)

    def action_show_logs(self) -> None:
        self.push_screen(LogScreen(self._log_handler))

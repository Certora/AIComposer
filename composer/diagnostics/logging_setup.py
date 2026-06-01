"""
Logging setup for the autoprove pipeline.

Configures two rotating log sinks under ``{project_root}/.certora_internal/autoProve/``:

- ``{%Y-%m-%d_%H-%M-%S}.{thread_id}.log`` — human-readable text log from the ``composer`` namespace
- ``{%Y-%m-%d_%H-%M-%S}.{thread_id}.events.jsonl`` — structured event stream (one JSON object per line)

Both files rotate at 1 MiB with up to 5 backups. Third-party loggers
(``langgraph``, ``httpx``, ``anthropic``) are pinned to ``WARNING`` so they
don't drown out the autoprove signal.
"""

import logging
import pathlib
import time
from logging.handlers import RotatingFileHandler


_MAX_BYTES = 1024 * 1024
_BACKUP_COUNT = 5

EVENTS_LOGGER_NAME = "composer.events"

_THIRD_PARTY_QUIET = ("langgraph", "httpx", "anthropic", "urllib3", "openai")


class _RawMessageFormatter(logging.Formatter):
    """Formatter that emits the log message verbatim (already JSON-serialized)."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def setup_autoprove_logging(
    project_root: pathlib.Path | str,
    thread_id: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Configure the ``composer`` logger and its ``composer.events`` child.

    Returns ``(text_log_path, events_log_path)``.

    Files only — no stderr handler — so the TUI display and the console
    handler's own output are not disturbed. Tail the text log file for
    live debugging.

    Callers should arrange to invoke this exactly once per pipeline run.
    """
    log_dir = pathlib.Path(project_root) / ".certora_internal" / "autoProve"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    text_path = log_dir / f"{stamp}.{thread_id}.log"
    events_path = log_dir / f"{stamp}.{thread_id}.events.jsonl"

    text_handler = RotatingFileHandler(
        text_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    text_handler.setLevel(logging.DEBUG)
    text_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    composer = logging.getLogger("composer")
    composer.setLevel(logging.DEBUG)
    composer.addHandler(text_handler)

    events = logging.getLogger(EVENTS_LOGGER_NAME)
    events.setLevel(logging.DEBUG)
    events.propagate = False
    events_handler = RotatingFileHandler(
        events_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    events_handler.setLevel(logging.DEBUG)
    events_handler.setFormatter(_RawMessageFormatter())
    events.addHandler(events_handler)

    for name in _THIRD_PARTY_QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)

    composer.info(
        f"autoprove logging initialized: thread_id={thread_id} text={text_path} events={events_path}"
    )

    return text_path, events_path

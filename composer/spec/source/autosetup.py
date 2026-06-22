"""
AutoSetup integration for spec generation.

Provides compilation analysis and summary generation for verifying specs
against real source code.
"""

import json
import logging
import os
import sys
import tempfile
from pydantic import BaseModel, Field
from pathlib import Path
from contextvars import ContextVar
from typing import Any, TypedDict, Literal, Annotated, Protocol
from pydantic import Discriminator
import asyncio
from composer.prover.core import ProverOptions

from graphcore.utils import TokenUsageDict
from composer.io.context import emit_custom_event

_logger = logging.getLogger(__name__)

class SetupSuccess(BaseModel):
    """Result of running AutoSetup compilation analysis and summary generation."""
    prover_config: dict  # Contents of compilation_config.conf
    summaries_path: str  # Path to summaries-{Contract}.spec, if generated
    user_types: list[dict]

class SetupFailure(BaseModel):
    error: str
    stderr: str | None = Field(default=None)

type SetupResult = SetupSuccess | SetupFailure

class AutoSetupComplete(TypedDict):
    type: Literal["auto_setup_complete"]
    return_code: int

class AutoSetupStart(TypedDict):
    type: Literal["auto_setup_start"]

class AutoSetupStdout(TypedDict):
    type: Literal["auto_setup_output"]
    line: str

type AutoSetupEvents = Annotated[
    AutoSetupComplete | AutoSetupStart | AutoSetupStdout, Discriminator("type")
]

class SetupLifecycleCallbacks(Protocol):
    def log_start(self) -> None:
        ...

    def log_stdout(self, line: str) -> None:
        ...

    def log_complete(self, returncode: int) -> None:
        ...


class SetupImpl(Protocol):
    async def __call__(
        self,
        callbacks: SetupLifecycleCallbacks,
        project_root: Path,
        relative_path: str,
        main_contract: str,
        prover_opts: ProverOptions,
        *extra_files
    ) -> SetupResult:
        ...


_setup_impl : ContextVar[SetupImpl | None] = ContextVar("_setup_impl", default=None)

async def run_autosetup(
    project_root: Path,
    relative_path: str,
    main_contract: str,
    prover_opts: ProverOptions,
    *extra_files: str
) -> SetupResult:
    """
    Run AutoSetup compilation analysis and summary generation.

    Args:
        project_root: Path to the Foundry project root
        relative_path: Relative path to the main contract file
        main_contract: Contract name, e.g. "Token"
        prover_opts: Prover options carrying cloud flag + extra_args forwarded to certoraRun

    Returns:
        SetupResult with compilation config and summaries path
    """

    def emitter(
        s: AutoSetupEvents
    ):
        emit_custom_event(s)

    class CB():
        def log_start(self):
            emitter({
                "type": "auto_setup_start"
            })

        def log_stdout(self, line: str):
            emitter({
                "line": line,
                "type": "auto_setup_output"
            })

        def log_complete(self, returncode: int):
            emitter({
                "return_code": returncode,
                "type": "auto_setup_complete"
            })

    _impl = _setup_impl.get()
    if _impl is None:
        raise RuntimeError("No implementation of autosetup; failing")

    return await _impl(
        CB(),
        project_root,
        relative_path,
        main_contract,
        prover_opts,
        *extra_files
    )


# --- AutoSetup LLM token-usage ingestion -----------------------------------
# AutoSetup runs as a separate process, so its LLM calls never pass through
# composer's model factory and are invisible to the UsageCallback. Its only
# trace is the llm_usage.json it writes to disk. These names are AutoSetup's
# on-disk contract (certora_autosetup constants + the autosetup_result.json it
# drops in .certora_internal); we hard-code them rather than import
# certora_autosetup, which composer must not depend on at import time.
# TODO: improve this once the repos are merged.
_CERTORA_INTERNAL = ".certora_internal"
_AUTOSETUP_RESULT_FILE = "autosetup_result.json"
_REPORTS_DIR = ".CertoraProverLiteReports"
_LLM_USAGE_FILE = "llm_usage.json"


def _resolve_autosetup_usage_file(project_root: Path) -> Path | None:
    """Locate the ``llm_usage.json`` AutoSetup wrote for the most recent run.

    Primary: read ``orchestration_timestamp`` from ``autosetup_result.json`` and
    build ``.CertoraProverLiteReports/<ts>/llm_usage.json`` — the deterministic
    inverse of how AutoSetup names that dir. Fallback (older AutoSetup, or a
    ``--reports-dir`` override that kept the convention): the newest timestamped
    subdir holding an ``llm_usage.json``. Timestamps are ``%Y%m%d_%H%M%S``, so a
    plain lexicographic max picks the most recent. Returns ``None`` if nothing
    is found.
    """
    reports_root = project_root / _REPORTS_DIR

    result_path = project_root / _CERTORA_INTERNAL / _AUTOSETUP_RESULT_FILE
    try:
        timestamp = json.loads(result_path.read_text()).get("orchestration_timestamp")
    except (OSError, ValueError):
        timestamp = None
    if timestamp:
        candidate = reports_root / timestamp / _LLM_USAGE_FILE
        if candidate.exists():
            return candidate

    try:
        subdirs = sorted((d for d in reports_root.iterdir() if d.is_dir()), key=lambda d: d.name)
    except OSError:
        return None
    for d in reversed(subdirs):
        candidate = d / _LLM_USAGE_FILE
        if candidate.exists():
            return candidate
    return None


def _to_token_usage(model: str, bucket: dict) -> TokenUsageDict:
    """Build a graphcore ``TokenUsageDict`` from one AutoSetup rollup bucket,
    keeping only the four token fields composer tracks (AutoSetup's ``calls``
    count has no slot in ``TokenTotals`` and is dropped)."""
    return {
        "model_name": model,
        "input_tokens": int(bucket.get("input_tokens", 0)),
        "output_tokens": int(bucket.get("output_tokens", 0)),
        "cache_read_input_tokens": int(bucket.get("cache_read_input_tokens", 0)),
        "cache_creation_input_tokens": int(bucket.get("cache_creation_input_tokens", 0)),
    }


def read_autosetup_usage(project_root: Path) -> list[TokenUsageDict]:
    """Return AutoSetup's per-model token usage for the most recent run, one
    ``TokenUsageDict`` per model — ready to feed straight into
    ``RunSummary.record_token_usage``.

    Returns ``[]`` on any failure (file absent — autosetup skipped, cache hit,
    crash, replayed snapshot, or an AutoSetup too old to emit it — or malformed
    JSON): missing external usage must never break the phase.
    """
    usage_file = _resolve_autosetup_usage_file(project_root)
    if usage_file is None:
        return []
    try:
        totals = json.loads(usage_file.read_text())["llm_usage_totals"]
        by_model = totals["by_model"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        _logger.debug(f"Could not read AutoSetup llm usage from {usage_file}: {e}")
        return []

    return [_to_token_usage(model, bucket) for model, bucket in by_model.items()]

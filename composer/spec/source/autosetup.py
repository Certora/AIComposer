"""
AutoSetup integration for spec generation.

Provides compilation analysis and summary generation for verifying specs
against real source code.
"""

import json
import os
import sys
import tempfile
from pydantic import BaseModel, Field
from pathlib import Path
from contextvars import ContextVar
from typing import Any, TypedDict, Literal, Annotated, Protocol
from pydantic import Discriminator
import asyncio

from composer.io.context import emit_custom_event

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
        *extra_files
    ) -> SetupResult:
        ...
        

_setup_impl : ContextVar[SetupImpl | None] = ContextVar("_setup_impl", default=None)

async def run_autosetup(
    project_root: Path,
    relative_path: str,
    main_contract: str,
    *extra_files: str
) -> SetupResult:
    """
    Run AutoSetup compilation analysis and summary generation.

    Args:
        project_root: Path to the Foundry project root
        relative_path: Relative path to the main contract file
        main_contract: Contract name, e.g. "Token"

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
        *extra_files
    )

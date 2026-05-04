"""
AutoSetup integration for spec generation.

Provides compilation analysis and summary generation for verifying specs
against real source code.
"""

import json
import os
import sys
import tempfile
from pydantic import BaseModel
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, Literal, Annotated
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

import logging

_logger = logging.getLogger(__name__)

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

    # Path to AutoSetup source directory
    AUTOSETUP_PATH = os.environ.get("AUTOSETUP_PATH")
    if AUTOSETUP_PATH is None:
        raise ValueError("You must set the AUTOSETUP_PATH in your environment")
    AUTOSETUP_SRC_PATH = Path(AUTOSETUP_PATH) / "src"


    contract_name = main_contract
    assert (project_root / relative_path).is_file()

    certora_dir = project_root / "certora"

    # Build environment with AutoSetup on PYTHONPATH
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{AUTOSETUP_SRC_PATH}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(AUTOSETUP_SRC_PATH)

    # Run orchestrator with --setup-only
    with tempfile.NamedTemporaryFile("r") as f:
        main_contract_path = f"{relative_path}:{contract_name}"
        args = [
            sys.executable, "-m", "orchestrator",
            "--composer-setup", f.name,
            "--skip-hashing-bound-detection", "1024",
            "--use-local-runner",
            "--no-strip-contracts",
            "--main-contract",
            main_contract_path,
            main_contract_path,
            *extra_files
        ]
        start_payload : AutoSetupStart = {
            "type": "auto_setup_start"
        }
        emit_custom_event(start_payload)
        proc = await asyncio.subprocess.create_subprocess_exec(
            *args,
            cwd=project_root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode().rstrip()
            if not line:
                continue
            line_payload : AutoSetupStdout = {
                "type": "auto_setup_output",
                "line": line
            }
            emit_custom_event(line_payload)
        returncode = await proc.wait()
        all_output = await proc.stderr.read() # type: ignore
        _logger.error(all_output.decode())
        end_payload : AutoSetupComplete = {
            "type": "auto_setup_complete",
            "return_code": returncode
        }
        emit_custom_event(end_payload)
        if returncode != 0:
            return SetupFailure(
                error="AutoSetup failed",
            )

        data = json.load(f)

    summary_path = Path(data["contract_to_summary"][main_contract])
    if not summary_path.is_relative_to(certora_dir):
        return SetupFailure(error="Summary not in project relative path")

    udts = json.loads((project_root / ".certora_internal" / "all_user_defined_types.json").read_text())

    return SetupSuccess(
        prover_config=json.loads((project_root / data["contract_to_config"][main_contract]).read_text()),
        summaries_path=str(summary_path.relative_to(certora_dir)),
        user_types=udts
    )

"""
PreAudit integration for spec generation.

Provides compilation analysis and summary generation for verifying specs
against real source code.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Path to PreAudit source directory
PREAUDIT_PATH = Path(__file__).parent.parent.parent.parent / "PreAudit"
PREAUDIT_SRC_PATH = PREAUDIT_PATH / "src"


@dataclass
class SetupSuccess:
    """Result of running PreAudit compilation analysis and summary generation."""
    config: dict  # Contents of compilation_config.conf
    summaries_path: Path  # Path to summaries-{Contract}.spec, if generated
    user_types: list[dict]

@dataclass
class SetupFailure:
    error: str

type SetupResult = SetupSuccess | SetupFailure

def run_preaudit_setup(
    project_root: Path,
    relative_path: str,
    main_contract: str,
    *extra_files: str
) -> SetupResult:
    """
    Run PreAudit compilation analysis and summary generation.

    Args:
        project_root: Path to the Foundry project root
        main_contract: Contract handle, e.g. "src/Token.sol:Token" or just "Token"

    Returns:
        SetupResult with compilation config and summaries path
    """
    # Extract contract name for summaries filename
    contract_name = main_contract
    assert (project_root / relative_path).is_file()

    certora_dir = project_root / "certora"

    # Build environment with PreAudit on PYTHONPATH
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{PREAUDIT_SRC_PATH}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(PREAUDIT_SRC_PATH)

    # Run orchestrator with --stop-after-summaries
    with tempfile.NamedTemporaryFile("r") as f:
        main_contract_path = f"{relative_path}:{contract_name}"
        result = subprocess.run(
            [
                sys.executable, "-m", "orchestrator",
                main_contract_path,
                "--setup-only", f.name,
                "--skip-hashing-bound-detection", "1024",
                "--use-local-runner",
                "--additional-contracts", *extra_files,
                "--main-contracts",
                main_contract_path
            ],
            cwd=project_root,
            text=True,
            env=env,
        )
        data = json.load(f)

    if result.returncode != 0:
        return SetupFailure(
            error=f"PreAudit setup failed",
        )

    summary_path = Path(data["contract_to_summary"][main_contract])
    if not summary_path.is_relative_to(certora_dir):
        SetupFailure(error="Summary not in project relative path")

    udts = json.loads((project_root / ".certora_internal" / "all_user_defined_types.json").read_text())

    return SetupSuccess(
        config=json.loads((project_root / data["contract_to_config"][main_contract]).read_text()),
        summaries_path=summary_path.relative_to(certora_dir),
        user_types=udts
    )

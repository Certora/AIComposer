"""
Spec-side prover tool: wraps composer/prover/core.py into a LangGraph tool.

Provides get_prover_tool() which creates a verify_spec tool that:
- Reads curr_spec from injected state
- Writes a temporary .spec file
- Runs the Certora prover via run_prover()
- Streams output/polling events via custom stream writer
"""

import asyncio
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Callable, NotRequired, TypedDict, Iterator

from langchain_core.messages import AnyMessage
from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, create_model

from langgraph.config import get_stream_writer
from langgraph.types import Command
from graphcore.graph import LLM

from composer.prover.core import (
    CloudConfig, ProverOptions, ProverCallbacks, run_prover,
)
from composer.diagnostics.stream import ProverOutputEvent, CloudPollingEvent
from composer.spec.cvl_generation import CVLGenerationState, make_validation_stamper

from graphcore.graph import tool_state_update

class _SpecCallbacks(ProverCallbacks):
    def __init__(self, writer: Callable, tool_call_id: str) -> None:
        self._writer = writer
        self._tool_call_id = tool_call_id

    async def on_stdout_line(self, line: str) -> None:
        evt: ProverOutputEvent = {
            "type": "prover_output",
            "tool_call_id": self._tool_call_id,
            "line": line,
        }
        self._writer(evt)

    async def on_cloud_poll(self, status: str, message: str) -> None:
        evt: CloudPollingEvent = {
            "type": "cloud_polling",
            "tool_call_id": self._tool_call_id,
            "status": status,
            "message": message,
        }
        self._writer(evt)


class VerifySpecSchema(BaseModel):
    """
    Run the Certora prover to verify the current spec against the source code.

    Returns verification results:
    - VERIFIED: Rule holds for all inputs
    - VIOLATED: Counterexample found (with CEX analysis)
    - TIMEOUT: Verification did not complete in time

    Use these results to refine your spec.
    """
    tool_call_id: Annotated[str, InjectedToolCallId]

    rules: list[str] | None = Field(
        default=None,
        description="Specific rules to verify. If None, verifies all rules."
    )
    state: Annotated[CVLGenerationState, InjectedState]


@contextmanager
def tmp_spec(
    *,
    root: str,
    content: str,
    prefix: str = "generated"
) -> Iterator[str]:
    t_name = f"{prefix}_{uuid.uuid4().hex[:16]}.spec"
    rel_path = "certora/" + t_name
    full_path = (Path(root) / "certora" / t_name)
    full_path.write_text(content)
    try:
        yield rel_path
    finally:
        os.unlink(full_path)


def get_prover_tool(
    llm: LLM,
    conf: dict,
    main_contract: str,
    project_root: str,
    cloud: CloudConfig | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> BaseTool:
    sem = semaphore or asyncio.Semaphore(1)
    stamper = make_validation_stamper("prover")

    @tool(args_schema=VerifySpecSchema)
    async def verify_spec(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[CVLGenerationState, InjectedState],
        rules: list[str] | None = None
    ) -> str | Command:
        if state["curr_spec"] is None:
            return "Specification not yet put on VFS"

        with tmp_spec(root=project_root, content=state["curr_spec"]) as generated:
            config = {
                **conf,
                "verify": f"{main_contract}:{generated}",
                "parametric_contracts": main_contract,
                "optimistic_loop": True,
                "rule_sanity": "basic",
            }

            if rules:
                config["rule"] = rules

            certora_dir = Path(project_root) / "certora"
            config_path = certora_dir / "verify.conf"
            config_path.write_text(json.dumps(config, indent=2))

            async with sem:
                result = await run_prover(
                    state,
                    Path(project_root),
                    [str(config_path)],
                    llm,
                    tool_call_id,
                    ProverOptions(cloud=cloud),
                    _SpecCallbacks(get_stream_writer(), tool_call_id),
                )

            if isinstance(result, str):
                return result
            if rules is None and result.all_verified:
                return tool_state_update(tool_call_id=tool_call_id, content=result.report, validations=stamper(state))
            return result.report

    return verify_spec

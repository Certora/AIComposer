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
from typing import Annotated, Callable, NotRequired, Iterator
from typing_extensions import TypedDict

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
from composer.spec.util import temp_certora_file

DELETE_SKIP = "__delete_skip"

VALIDATION_KEY = "prover"

def _merge_rule_skips(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    to_ret = left.copy()
    for (k,v) in right:
        if v == DELETE_SKIP:
            if k in to_ret:
                del to_ret[k]
            continue
        to_ret[k] = v
    return to_ret


class ProverStateExtra(TypedDict):
    rule_skips: Annotated[dict[str, str], _merge_rule_skips]
    config: dict

class StateWithSkips(CVLGenerationState, ProverStateExtra):
    pass

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
    state: Annotated[StateWithSkips, InjectedState]


@contextmanager
def tmp_spec(
    *,
    root: str,
    content: str,
    prefix: str = "generated"
) -> Iterator[str]:
    with temp_certora_file(
        root=root,
        ext="spec",
        content=content,
        prefix=prefix
    ) as tmp:
        yield tmp

def get_prover_tool(
    llm: LLM,
    main_contract: str,
    project_root: str,
    cloud: CloudConfig | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> BaseTool:
    sem = semaphore or asyncio.Semaphore(1)
    stamper = make_validation_stamper(VALIDATION_KEY)

    @tool(args_schema=VerifySpecSchema)
    async def verify_spec(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[StateWithSkips, InjectedState],
        rules: list[str] | None = None
    ) -> str | Command:
        if state["curr_spec"] is None:
            return "Specification not yet put on VFS"
        conf = state["config"]
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
            
            with temp_certora_file(
                root = project_root,
                content=json.dumps(config, indent=2),
                ext="conf",
                prefix="verify"
            ) as config_path:
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
            all_verified = True
            for (r, stat) in result.rule_status.items():
                if r in state["rule_skips"]:
                    continue
                if not stat:
                    all_verified = False
                    break
            if rules is None and all_verified:
                return tool_state_update(tool_call_id=tool_call_id, content=result.report, validations=stamper(state))
            return result.report

    return verify_spec

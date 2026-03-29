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
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Callable, Iterator, override
from typing_extensions import TypedDict

from langchain_core.tools import InjectedToolCallId, tool, BaseTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from langgraph.config import get_stream_writer
from langgraph.types import Command
from composer.prover.ptypes import RuleResult
from graphcore.graph import LLM

from composer.prover.core import (
    CloudConfig, ProverOptions, ProverCallbacks, run_prover, SummarizedReport
)
from composer.diagnostics.stream import (
    ProverOutputEvent, CloudPollingEvent, RuleAnalysisResult,
    CEXAnalysisStart, ProverRun, ProverResult
)
from composer.spec.cvl_generation import CVLGenerationState, make_validation_stamper
from graphcore.graph import tool_state_update
from composer.spec.util import temp_certora_file

DELETE_SKIP = "__delete_skip"

VALIDATION_KEY = "prover"

def _merge_rule_skips(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    to_ret = left.copy()
    for (k,v) in right.items():
        if v == DELETE_SKIP:
            if k in to_ret:
                del to_ret[k]
            continue
        to_ret[k] = v
    return to_ret


class ProverStateExtra(TypedDict):
    rule_skips: Annotated[dict[str, str], _merge_rule_skips]
    config: dict

type ProverEvents = CEXAnalysisStart | CloudPollingEvent | ProverOutputEvent | RuleAnalysisResult | ProverRun | ProverResult

class StateWithSkips(CVLGenerationState, ProverStateExtra):
    pass

class _SpecCallbacks(ProverCallbacks):
    def __init__(self, writer: Callable[[ProverEvents], None], tool_call_id: str) -> None:
        self._writer = writer
        self._tool_call_id = tool_call_id

    @override
    async def on_stdout_line(self, line: str) -> None:
        self._writer({
            "type": "prover_output",
            "tool_call_id": self._tool_call_id,
            "line": line,
        })

    @override
    async def on_cloud_poll(self, status: str, message: str) -> None:
        self._writer({
            "type": "cloud_polling",
            "tool_call_id": self._tool_call_id,
            "status": status,
            "message": message,
        })

    @override
    async def on_analysis_start(self, rule: RuleResult) -> None:
        self._writer({
            "type": "cex_analysis",
            "rule_name": rule.path.pprint(),
            "tool_call_id": self._tool_call_id
        })
    
    @override
    async def on_analysis_complete(self, rule: RuleResult, analysis: str) -> None:
        self._writer({
            "type": "rule_analysis",
            "analysis": analysis,
            "tool_call_id": self._tool_call_id,
            "rule": rule.path.pprint()
        })
    
    @override
    async def on_prover_run(self, args: list[str]) -> None:
        self._writer({
            "type": "prover_run",
            "tool_call_id": self._tool_call_id,
            "args": args
        })

    @override
    async def on_prover_result(self, results: dict[str, RuleResult]) -> None:
        self._writer({
            "type": "prover_result",
            "tool_call_id": self._tool_call_id,
            "status": { k: v.status for (k,v) in results.items() }
        })


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
                "verify": f"{main_contract}:certora/{generated}",
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
                        [f"certora/{config_path}"],
                        llm,
                        tool_call_id,
                        ProverOptions(cloud=cloud),
                        _SpecCallbacks(get_stream_writer(), tool_call_id),
                        summarization_threshold=10
                    )

            if isinstance(result, str):
                return result
            if isinstance(result, SummarizedReport):
                return result.todo_list
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

from typing import Optional, List, Callable
import asyncio
from pathlib import Path

from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime

from langgraph.config import get_store

from composer.diagnostics.stream import ProgressUpdate, AuditUpdate
from composer.prover.ptypes import RuleResult
from composer.prover.core import (
    RawReport, SummarizedReport, ProverOptions as CoreProverOptions,
    ProverCallbacks, AnalysisCache, run_prover,
)
from composer.core.state import AIComposerState
from composer.core.context import AIComposerContext

import sys


class _AuditCallbacks(ProverCallbacks):
    def __init__(self, writer: Callable, tool_call_id: str, capture_output: bool) -> None:
        self._writer = writer
        self._tool_call_id = tool_call_id
        self._capture_output = capture_output

    async def on_stdout_line(self, line: str) -> None:
        if not self._capture_output:
            print(line)

    async def on_prover_run(self, args: list[str]) -> None:
        run_message: ProgressUpdate = {
            "type": "prover_run",
            "args": args,
        }
        self._writer(run_message)

    async def on_prover_result(self, results: dict[str, RuleResult]) -> None:
        result_message = {
            "type": "prover_result",
            "status": {k: v.status for k, v in results.items()},
        }
        self._writer(result_message)

    async def on_analysis_start(self, rule: RuleResult) -> None:
        cex_event: ProgressUpdate = {
            "type": "cex_analysis",
            "rule_name": rule.name,
        }
        self._writer(cex_event)

    async def on_rule_result(self, rule: RuleResult, analysis: str | None) -> None:
        rule_audit_res: AuditUpdate = {
            "analysis": analysis,
            "rule": rule.name,
            "status": rule.status,
            "type": "rule_result",
            "tool_id": self._tool_call_id,
        }
        self._writer(rule_audit_res)


class _StoreCache:
    def __init__(self, store, tool_call_id: str) -> None:
        self._store = store
        self._tool_call_id = tool_call_id

    def get(self, rule: RuleResult) -> str | None:
        d = self._store.get(("cex", self._tool_call_id), rule.path.pprint())
        if d is not None:
            return d.value["analysis"]
        return None

    def put(self, rule: RuleResult, analysis: str) -> None:
        self._store.put(("cex", self._tool_call_id), rule.path.pprint(), {"analysis": analysis})


def certora_prover(
    source_files: List[str],
    target_contract: str,
    compiler_version: str,
    loop_iter: int,
    rule: Optional[str],
    state: AIComposerState,
    tool_call_id: str,
) -> SummarizedReport | RawReport | str:
    runtime = get_runtime(AIComposerContext)
    ctxt = runtime.context
    writer = get_stream_writer()

    with ctxt.vfs_materializer.materialize(state, debug=ctxt.prover_opts.keep_folder) as temp_dir:
        try:
            args = source_files.copy()
            args.extend([
                "--verify",
                f"{target_contract}:./rules.spec",
                "--optimistic_loop",
                "--optimistic_hashing",
                "--loop_iter",
                str(loop_iter),
                "--solc", compiler_version,
                "--solc_via_ir",
                "--strict_solc_optimizer",
                "--prover_args",
                "-timeoutCracker true",
            ])
            if rule is not None:
                args.extend(["--rule", rule])

            store = get_store()
            cache: AnalysisCache | None = _StoreCache(store, tool_call_id) if store is not None else None

            result = asyncio.run(run_prover(
                state,
                Path(temp_dir),
                args,
                ctxt.llm,
                tool_call_id,
                CoreProverOptions(),
                _AuditCallbacks(writer, tool_call_id, ctxt.prover_opts.capture_output),
                analysis_cache=cache,
                summarization_threshold=10,
            ))

            # Preserve the rule-is-None check for all_verified
            if isinstance(result, RawReport) and rule is not None:
                result = RawReport(report=result.report, all_verified=False)

            return result
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

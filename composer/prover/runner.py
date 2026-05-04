import json
from typing import Optional, List, Callable, override
from pathlib import Path

from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime

from composer.diagnostics.stream import (
    ProverRun, ProverResult, RuleAnalysisResult, CEXAnalysisStart,
    ProverOutputEvent, CloudPollingEvent,
)
from composer.prover.ptypes import RuleResult
from composer.prover.core import (
    ProverReport, ProverOptions as CoreProverOptions,
    ProverCallbacks, run_prover,
)
from composer.core.state import AIComposerState
from composer.core.context import AIComposerContext

type _ProverEvents = ProverOutputEvent | CloudPollingEvent | ProverRun | ProverResult | RuleAnalysisResult | CEXAnalysisStart


_COMPOSER_CONF_NAME = "_composer_run.conf"
_WORKING_SPEC_SCRATCH = "_composer_working.spec"


def _build_conf_args(
    *,
    temp_dir: Path,
    overrides: dict,
    source_files: List[str],
    target_contract: str,
    target_spec: str,
    compiler_version: str,
    loop_iter: int,
    rule: Optional[str],
) -> list[str]:
    """Materialize the merged Certora conf to a tempfile and return the CLI args to invoke it.

    Pipeline-authoritative keys (``files``, ``verify``, ``solc``, ``loop_iter``, ``rule``,
    plus the fixed options ``optimistic_loop``, ``optimistic_hashing``, ``solc_via_ir``,
    ``strict_solc_optimizer``, ``prover_args``) are written last so they win over
    whatever the user supplied.
    """
    conf: dict = {**overrides}
    conf["files"] = list(source_files)
    conf["verify"] = f"{target_contract}:./{target_spec}"
    conf["solc"] = compiler_version
    conf["optimistic_loop"] = True
    conf["optimistic_hashing"] = True
    conf["loop_iter"] = str(loop_iter)
    conf["solc_via_ir"] = True
    conf["strict_solc_optimizer"] = True
    conf["prover_args"] = ["-timeoutCracker true"]
    if rule is not None:
        conf["rule"] = [rule]

    (temp_dir / _COMPOSER_CONF_NAME).write_text(json.dumps(conf, indent=2))
    return [_COMPOSER_CONF_NAME]

class _AuditCallbacks(ProverCallbacks):
    def __init__(self, writer: Callable[[_ProverEvents], None], tool_call_id: str) -> None:
        self._writer = writer
        self._tool_call_id = tool_call_id

    @override
    async def on_stdout_line(self, line: str) -> None:
        evt: ProverOutputEvent = {
            "type": "prover_output",
            "tool_call_id": self._tool_call_id,
            "line": line,
        }
        self._writer(evt)

    @override
    async def on_cloud_poll(self, status: str, message: str) -> None:
        evt: CloudPollingEvent = {
            "type": "cloud_polling",
            "tool_call_id": self._tool_call_id,
            "status": status,
            "message": message,
        }
        self._writer(evt)

    @override
    async def on_prover_run(self, args: list[str]) -> None:
        evt: ProverRun = {
            "type": "prover_run",
            "args": args,
            "tool_call_id": self._tool_call_id,
        }
        self._writer(evt)

    @override
    async def on_prover_result(self, results: dict[str, RuleResult]) -> None:
        evt: ProverResult = {
            "type": "prover_result",
            "tool_call_id": self._tool_call_id,
            "status": {k: v.status for k, v in results.items()},
        }
        self._writer(evt)

    @override
    async def on_analysis_complete(self, rule: RuleResult, analysis: str) -> None:
        evt: RuleAnalysisResult = {
            "type": "rule_analysis",
            "tool_call_id": self._tool_call_id,
            "rule": rule.path.pprint(),
            "analysis": analysis,
        }
        self._writer(evt)

    @override
    async def on_analysis_start(self, rule: RuleResult) -> None:
        evt: CEXAnalysisStart = {
            "type": "cex_analysis",
            "tool_call_id": self._tool_call_id,
            "rule_name": rule.name,
        }
        self._writer(evt)

async def certora_prover(
    source_files: List[str],
    target_contract: str,
    compiler_version: str,
    loop_iter: int,
    rule: Optional[str],
    state: AIComposerState,
    tool_call_id: str,
    use_working_spec: bool,
    target_spec: Optional[str],
) -> ProverReport | str:
    if use_working_spec:
        ws = state["working_spec"]
        if not ws:
            return "No working spec written."
        # The working draft has no VFS path; write it to a scratch location
        # inside the materialized tmpdir and verify against that. The caller
        # should not have passed target_spec; if they did, it's ignored.
        effective_spec = _WORKING_SPEC_SCRATCH
    else:
        if target_spec is None:
            return (
                "target_spec is required when use_working_spec is false. "
                "Pass the VFS path of one of the registered spec files."
            )
        effective_spec = target_spec
    runtime = get_runtime(AIComposerContext)
    ctxt = runtime.context
    writer = get_stream_writer()

    with ctxt.vfs_materializer.materialize(state, debug=ctxt.prover_opts.keep_folder) as temp_dir:
        if use_working_spec:
            ws = state["working_spec"]
            assert ws is not None
            (Path(temp_dir) / _WORKING_SPEC_SCRATCH).write_text(ws)
        try:
            if ctxt.prover_conf_overrides is not None:
                args = _build_conf_args(
                    temp_dir=Path(temp_dir),
                    overrides=ctxt.prover_conf_overrides,
                    source_files=source_files,
                    target_contract=target_contract,
                    target_spec=effective_spec,
                    compiler_version=compiler_version,
                    loop_iter=loop_iter,
                    rule=rule,
                )
            else:
                args = source_files.copy()
                args.extend([
                    "--verify",
                    f"{target_contract}:./{effective_spec}",
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

            return await run_prover(
                Path(temp_dir),
                args,
                tool_call_id,
                CoreProverOptions(cloud=ctxt.prover_opts.cloud),
                _AuditCallbacks(writer, tool_call_id),
                ctxt.cex_handler,
            )
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            raise e

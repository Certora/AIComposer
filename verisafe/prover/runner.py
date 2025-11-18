from typing import Optional, List, Tuple, Callable, Awaitable, TypeVar
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import Iterable
import asyncio
from pathlib import Path
import contextlib

import tempfile
from dataclasses import dataclass

from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime
from langchain_core.messages import ToolMessage, HumanMessage

from certoraRun import CertoraRunResult

from graphcore.utils import cached_invoke
from graphcore.graph import BoundLLM

from verisafe.templates.loader import load_jinja_template
from verisafe.diagnostics.stream import ProgressUpdate, AuditUpdate
from verisafe.prover.results import read_and_format_run_result
from verisafe.prover.ptypes import RuleResult
from verisafe.prover.analysis import analyze_cex
from verisafe.core.state import CryptoStateGen
from verisafe.core.context import CryptoContext, ProverOptions


import sys
import subprocess
import pickle

@dataclass
class RawReport:
    report: str
    all_verified: bool

@dataclass
class SummarizedReport:
    report: str
    todo_list: str

@dataclass
class SandboxedRunResult:
    """
    Represents the result of a certoraRunWrapper execution
    """
    exit_code: int
    stdout: str
    stderr: str
    run_result: Optional[CertoraRunResult]

class CertoraRunFailure(RuntimeError):
    def __init__(self, return_code: int, stdout: str, stderr: str):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr

class CertoraRunException(RuntimeError):
    def __init__(self, exc: Exception, stdout: str, stderr: str):
        self.wrapped = exc
        self.stdout = stdout
        self.stderr = stderr

def sandboxed_certora_run(
    args: List[str],
    prover_opts: ProverOptions
) -> SandboxedRunResult:
    wrapper_script = Path(__file__).parent / "certoraRunWrapper.py"
    with tempfile.NamedTemporaryFile("rb") as dump:
        sub_args = [sys.executable, str(wrapper_script)]
        sub_args.append(dump.name)
        sub_args.extend(args)
        r = subprocess.run(sub_args, encoding="utf-8", capture_output=prover_opts.capture_output)
        if r.returncode != 0:
            raise CertoraRunFailure(
                return_code=r.returncode,
                stderr=r.stderr,
                stdout=r.stdout
            )
        o = pickle.load(dump)
        if isinstance(o, Exception):
            raise CertoraRunException(exc=o, stderr=r.stderr, stdout=r.stdout)
        return SandboxedRunResult(
            exit_code=r.returncode,
            stderr=r.stderr,
            stdout=r.stdout,
            run_result=o
        )

async def _analyze(
    llm: BoundLLM, state: CryptoStateGen, res: RuleResult, tool_call_id: str
) -> tuple[RuleResult, str | None]:
    cex_analysis = None
    if res.status == "VIOLATED":
        cex_analysis = await analyze_cex(llm, state, res, tool_call_id=tool_call_id)
    return (res, cex_analysis)

T = TypeVar('T')
R = TypeVar('R')

def apply_async_parallel(
    func: Callable[[T], Awaitable[R]], 
    items: Iterable[T]
) -> list[R]:
    """
    Apply an async function to items in parallel and return results.
    
    Works whether or not there's an active event loop.
    """
    async def _gather_results():
        tasks = [func(item) for item in items]
        return await asyncio.gather(*tasks)
    
    in_loop = False
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        pass
    if not in_loop:
        return asyncio.run(_gather_results())
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, _gather_results())
            return future.result()

def certora_prover(
    source_files: List[str],
    # spec_file: str,
    target_contract: str,
    compiler_version: str,
    loop_iter: int,
    rule: Optional[str],
    state: CryptoStateGen,
    tool_call_id: str
) -> SummarizedReport | RawReport | str:
    runtime = get_runtime(CryptoContext)
    ctxt = runtime.context
    writer = get_stream_writer()
    with ctxt.vfs_materializer.materialize(state, debug=ctxt.prover_opts.keep_folder) as temp_dir:
        with contextlib.chdir(temp_dir):
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
                    "-timeoutCracker true"
                ])
                if rule is not None:
                    args.extend([
                        "--rule", rule
                    ])
                run_message: ProgressUpdate = {
                    "type": "prover_run",
                    "args": args
                }
                writer(run_message)

                try:
                    res = sandboxed_certora_run(
                        args, runtime.context.prover_opts
                    )
                except CertoraRunFailure as e:
                    return f"Certora Prover run exited with non-zero returncode {e.return_code}.\nStdout:\n{e.stdout}.\nStderr: {e.stderr}"
                except CertoraRunException as e:
                    return f"Certora Prover run failed exceptionally with {str(e.wrapped)}.\nStdout:\n{e.stdout}\nStderr: {e.stderr}"
                if res is None or res.run_result is None:
                    return "Certora prover didn't actually run, this is likely a bug you should consult the user about"
                run_result = res.run_result
                assert run_result.is_local_link and run_result.link is not None

                formatted_run_result = read_and_format_run_result(Path(run_result.link))
                if isinstance(formatted_run_result, str):
                    return formatted_run_result
                run_message = {
                    "type": "prover_result",
                    "status": {k: v.status for (k, v) in formatted_run_result.items()}
                }
                writer(run_message)

                runtime = get_runtime(CryptoContext)
                failed_count = 0
                results_param = apply_async_parallel(
                    lambda d: _analyze(runtime.context.llm, state, d, tool_call_id=tool_call_id),
                    [ stat for (_, stat) in formatted_run_result.items() ]
                )
                for (stat, analysis) in results_param:
                    if stat.status == "VIOLATED":
                        failed_count += 1
                    rule_audit_res: AuditUpdate = {
                        "analysis": analysis,
                        "rule": stat.name,
                        "status": stat.status,
                        "type": "rule_result",
                        "tool_id": tool_call_id
                    }
                    writer(rule_audit_res)
                rule_report = load_jinja_template("rule_feedback.j2", results=results_param)
                if failed_count > 10:
                    todo_list = report_to_todo_list(state, rule_report, tool_call_id)
                    return SummarizedReport(
                        report=rule_report,
                        todo_list=todo_list
                    )
                return RawReport(rule_report, all_verified=(failed_count == 0 and rule is None))
            except Exception as e:
                print(str(e))
                import traceback
                traceback.print_exc()
                sys.exit(1)

def report_to_todo_list(state: CryptoStateGen, report: str, tool_call_id: str) -> str:
    runtime = get_runtime(CryptoContext)
    ctxt = runtime.context
    llm = ctxt.llm
    messages = state["messages"].copy()
    messages.append(ToolMessage(
        content=report,
        tool_call_id=tool_call_id
    ))
    messages.append(HumanMessage(
        content="""
Analyze the counterexamples returned by the prover.
Investigate the rule-by-rule feedback and create a TODO list of the code changes you
need to perform to address the identified issues.
In particular, find common issues exhibited between the different counterexamples and identify
the common root causes and what changes are necessary.
"""
    ))
    res = cached_invoke(llm, messages)
    return res.text()
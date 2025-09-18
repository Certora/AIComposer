from typing import Optional, List, Protocol, ContextManager, Iterator, Tuple
from verisafe.core.state import CryptoStateGen
from verisafe.core.context import CryptoContext, ProverOptions
from pathlib import Path
import contextlib
from verisafe.templates.loader import load_jinja_template
import tempfile
from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime
from verisafe.diagnostics.stream import ProgressUpdate, RuleAuditResult, AuditUpdate
from certoraRun import CertoraRunResult
from dataclasses import dataclass
from verisafe.prover.results import read_and_format_run_result
from verisafe.prover.ptypes import RuleResult
from verisafe.prover.analysis import analyze_cex


import sys
import subprocess
import pickle

@dataclass
class SandboxedRunResult:
    """
    Represents the result of a certoraRunWrapper execution
    """
    exit_code: int
    stdout: str
    stderr: str
    run_result: Optional[CertoraRunResult]

class TempDirectoryProvider(Protocol):
    def __call__(self) -> ContextManager[str]:
        ...

# have to do this because `delete` as an argument to `TemporaryDirectory` is in python3.12 ...
@contextlib.contextmanager
def debugging_tmp_directory() -> Iterator[str]:
    temp_dir = tempfile.mkdtemp()
    print(f"DEBUG: Working directory: {temp_dir}")
    yield temp_dir

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
            raise Exception(r.stderr)
        o = pickle.load(dump)
        if isinstance(o, Exception):
            raise o
        return SandboxedRunResult(
            exit_code=r.returncode,
            stderr=r.stderr,
            stdout=r.stdout,
            run_result=o
        )

def certora_prover(
    source_files: List[str],
    # spec_file: str,
    target_contract: str,
    compiler_version: str,
    loop_iter: int,
    rule: Optional[str],
    state: CryptoStateGen,
    tool_call_id: str
) -> str:
    # Create temporary directory and keep it for debugging
    runtime = get_runtime(CryptoContext)
    provider: TempDirectoryProvider = tempfile.TemporaryDirectory
    if runtime.context.prover_opts.keep_folder:
        provider = debugging_tmp_directory
    writer = get_stream_writer()
    with provider() as temp_dir:
        with contextlib.chdir(temp_dir):
            try:
                t = Path(temp_dir)

                # Step 2: For each relative path in source_files, write the solidity source
                for (relative_path, solidity_content) in state["virtual_fs"].items():
                    file_path = t / relative_path
                    # Create parent directories if they don't exist
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(solidity_content)

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
                    "--strict_solc_optimizer"
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
                except Exception as e:
                    return f"Certora Prover run failed exceptionally with {str(e)}"
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
                results_param: List[Tuple[RuleResult, Optional[str]]] = []
                for (_, stat) in formatted_run_result.items():
                    cex_analysis = None
                    if stat.status == "VIOLATED":
                        cex_analysis = analyze_cex(runtime.context.llm, state, stat, tool_call_id=tool_call_id)
                    rule_audit_res: AuditUpdate = {
                        "analysis": cex_analysis,
                        "rule": stat.name,
                        "status": stat.status,
                        "type": "rule_result",
                        "tool_id": tool_call_id
                    }
                    writer(rule_audit_res)
                    results_param.append((stat, cex_analysis))
                rule_report = load_jinja_template("rule_feedback.j2", results=results_param)
                return rule_report
            except Exception as e:
                print(str(e))
                sys.exit(1)

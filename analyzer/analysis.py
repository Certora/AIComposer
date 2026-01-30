from typing import Any, NotRequired, TypedDict, Iterator
from dataclasses import dataclass
import pathlib
import os
import tempfile
import tarfile
import urllib.request
import urllib.parse
from contextlib import contextmanager

import uuid

from langgraph.graph import MessagesState

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.config import RunnableConfig

from composer.rag.db import PostgreSQLRAGDatabase, DEFAULT_CONNECTION
from composer.rag.models import get_model
import composer.prover.results as R
from composer.templates.loader import load_jinja_template

from composer.workflow.factories import get_checkpointer

from graphcore.tools.vfs import VFSState, VFSToolConfig, vfs_tools
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.results import result_tool_generator

from pydantic import BaseModel
from analyzer.types import AnalysisArgs, Ecosystem, DefaultAnalysisResult, DEFAULT_RESULT_TOOL_CONFIG

class EcosystemConfig(TypedDict):
    spec_name: str
    spec_name_full: str
    spec_coda: NotRequired[str]
    token_example: str
    ecosystem_name: str
    spec_description: str
    language_name: str


def find_tree_view_node(stat: R.TreeViewStatus, context: pathlib.Path, target: R.RulePath) -> R.RuleResult | None:
    for r in stat.rules:
        if r.name != target.rule:
            continue
        for d in R.flatten_tree_view(context=context, path=R.RulePath(rule=r.name), r=r):
            if d.path == target:
                return d
    return None

@dataclass
class ExplainerContext:
    rag_db: PostgreSQLRAGDatabase

class SimpleState(VFSState, MessagesState):
    result: NotRequired[Any]


def main() -> int:
    """CLI entry point for the analyzer."""
    import argparse
    import sys
    from typing import cast

    parser = argparse.ArgumentParser(
        description='Analyze Certora Prover counterexamples and generate natural language explanations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cex-analyzer /path/to/report myRule
  cex-analyzer /path/to/report myRule --method myMethod
  cex-analyzer /path/to/report myRule --method MyContract.myMethod
"""
    )

    parser.add_argument(
        'folder',
        type=str,
        help='Path to the Certora report directory containing the counterexample data'
    )

    parser.add_argument(
        'rule',
        type=str,
        help='Name of the rule to analyze'
    )

    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help='Optional method identifier. Can be either "method" or "contract.method" format'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress intermediate output during analysis (only show final result)'
    )

    parser.add_argument(
        "--ecosystem",
        type=str,
        default="evm"
    )

    parser.add_argument(
        '--recursion-limit',
        type=int,
        default=30,
        help="The recursion limit to use for the cex analysis"
    )

    parser.add_argument(
        "--thread-id",
        type=str,
        help="The thread id (for resuming halted/crashed runs)"
    )

    parser.add_argument(
        "--checkpoint-id",
        type=str,
        help="The checkpoint id (for resuming halted/crashed runs)"
    )

    parser.add_argument(
        "--thinking-tokens",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--rag-db",
        type=str,
        default=DEFAULT_CONNECTION,
        help="Database connection string for CVL manual search"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path for the analysis result (if not specified, prints to stdout)"
    )

    args = parser.parse_args()
    args.result_tool_config = DEFAULT_RESULT_TOOL_CONFIG
    return analyze(cast(AnalysisArgs, args))

ecosystem_params: dict[Ecosystem, EcosystemConfig] = {
    "evm": {
        "spec_coda": "cvl_description.j2",
        "spec_name": "CVL",
        "spec_name_full": "Certora Verification Language",
        "spec_description": "a DSL for writing specifications of smart contracts",
        "ecosystem_name": "Solidity",
        "token_example": "ERC20 token",
        "language_name": "Solidity"
    },
    "soroban": {
        "spec_name": "CVLR",
        "ecosystem_name": "Soroban",
        "spec_description": "a DSL embedded into Rust for writing specifications of smart contracts",
        "spec_name_full": "Certora Verification Language for Rust",
        "token_example": "token",
        "language_name": "Rust"
    },
    "move": {
        "spec_name": "CVLM",
        "ecosystem_name": "Move",
        "spec_description": "a DSL embedded into the Move language for writing specifications of smart contracts",
        "spec_name_full": "Certora Verification Language for Move",
        "token_example": "token",
        "language_name": "Move"
    },
    "solana": {
        "spec_name": "CVLR",
        "spec_description": "a DSL embedded into Rust for writing specifications of smart contracts",
        "spec_name_full": "Certora Verification Language for Rust",
        "ecosystem_name": "Solana",
        "language_name": "Rust",
        "token_example": "SPL token"
    }
}

def _looks_like_url(path: str) -> bool:
    """Check if a path looks like a URL using built-in Python heuristics."""
    parsed = urllib.parse.urlparse(path)
    return bool(parsed.scheme and parsed.netloc)

@contextmanager
def _download_and_extract_report(url: str) -> Iterator[pathlib.Path]:
    """Download tar.gz from Certora URL and extract to temporary directory."""
    zip_url = url.replace('/output/', '/zipOutput/')
    
    certora_key = os.environ.get("CERTORAKEY")
    if not certora_key:
        raise ValueError("CERTORAKEY environment variable not set")
    
    with tempfile.TemporaryDirectory(prefix="certora_report_") as temp_dir:
        request = urllib.request.Request(zip_url)
        request.add_header('Cookie', f'certoraKey={certora_key}')
        
        with urllib.request.urlopen(request) as response:
            tar_path = os.path.join(temp_dir, "report.tar.gz")
            with open(tar_path, 'wb') as f:
                f.write(response.read())
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
        
        os.remove(tar_path)
        
        yield pathlib.Path(temp_dir, "TarName")

def _analyze_core(
    input_messages: list[str],
    initial_prompt: str,
    report_dir: pathlib.Path,
    args: AnalysisArgs
) -> int:
    """Run the analysis workflow with custom calltraces and prompt.

    This is the lowest-level function that executes the analysis workflow.
    It sets up tools, LLM, and runs the workflow with the provided calltraces and prompt.

    Args:
        input_messages: List of input strings including context messages and XML calltraces
        initial_prompt: Custom initial prompt for the workflow
        report_dir: Path to the report directory (must be a local path)
        args: Configuration parameters

    Returns:
        Exit code (0 for success)
    """
    (v_tools, _) = vfs_tools(
        ty=SimpleState,
        conf=VFSToolConfig(
            immutable=True,
            forbidden_read=r"^\..*$",
            fs_layer=str(report_dir / "inputs" / ".certora_sources")
        )
    )

    analysis_output_tool = result_tool_generator(
        "result",
        args.result_tool_config.schema,
        args.result_tool_config.doc,
    )

    tools = [analysis_output_tool, *v_tools]
    if args.ecosystem == "evm":
        #import here to lazily load sentencetransformers
        from composer.tools.search import cvl_manual_search
        tools.append(cvl_manual_search)

    llm = ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens_to_sample=args.tokens,
        temperature=1,
        timeout=None,
        max_retries=2,
        stop=None,
        thinking={"type": "enabled", "budget_tokens": args.thinking_tokens},
        betas=["interleaved-thinking-2025-05-14", "context-management-2025-06-27"],
    )

    system_prompt = load_jinja_template("analyzer_system_prompt.j2")

    graph = build_workflow(
        input_type=FlowInput,
        context_schema=ExplainerContext,
        output_key="result",
        tools_list=tools,
        unbound_llm=llm,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt,
        state_class=SimpleState
    )[0].compile(checkpointer=get_checkpointer())

    conf : RunnableConfig = {"configurable": {}}
    tid : str
    if args.thread_id is not None:
        tid = args.thread_id
    else:
        tid = f"cex-analysis-{uuid.uuid1().hex}"
        if not args.quiet:
            print(f"Chose thread id: {tid}")

    conf["configurable"]["thread_id"] = tid
    if args.checkpoint_id is not None:
        conf["configurable"]["checkpoint_id"] = args.checkpoint_id

    conf["recursion_limit"] = args.recursion_limit

    for (ty, d) in graph.stream(input=FlowInput(input=input_messages), context=ExplainerContext(
        rag_db=PostgreSQLRAGDatabase(conn_string=args.rag_db, model=get_model(), skip_test=True)
    ), config=conf, stream_mode=["checkpoints", "updates"]):
        if ty == "checkpoints":
            assert isinstance(d, dict)
            if not args.quiet:
                print("current checkpoint: " + d["config"]["configurable"]["checkpoint_id"])
        else:
            if not args.quiet:
                print(d)

    raw_result = graph.get_state({"configurable": {"thread_id": tid}}).values["result"]

    if args.result_tool_config.schema is DefaultAnalysisResult:
        output_text = raw_result.result
    elif isinstance(raw_result, BaseModel):
        output_text = raw_result.model_dump_json(indent=2)
    else:
        output_text = str(raw_result)

    if args.output is not None:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Analysis written to {args.output}")
    else:
        print(output_text)

    return 0

def _analyze_from_report(
    report_dir: pathlib.Path,
    args: AnalysisArgs
) -> int:
    try:
        (stat, treeView) = R.get_final_treeview(report_dir)
    except (R.MalformedTreeVew, R.NoTreeViewResultError):
        print(f"Couldn't parse tree view from {report_dir}")
        return 1
    
    rule_target = args.rule

    contract: str | None = None
    method: str | None = None
    if args.method is not None:
        parametric_name = args.method
        components = parametric_name.split(".")

        if len(components) == 1:
            method = components[0]
        else:
            assert len(components) == 2
            contract = components[0]
            method = parametric_name

    target_path = R.RulePath(rule=rule_target, contract=contract, method=method)

    m = find_tree_view_node(stat, treeView, target_path)

    if m is None:
        print(f"Couldn't find {target_path.pprint()}")
        return 1

    if m.status != "VIOLATED":
        print("Rule wasn't violated?")
        return 1

    calltrace_xml = m.cex_dump

    assert calltrace_xml is not None

    # Build the initial prompt from ecosystem template
    process = ecosystem_params[args.ecosystem]
    initial_prompt = load_jinja_template("analyzer_tool_prompt.j2", **process)

    # Prepare the input messages with rule context and XML calltrace
    input_messages = [
        f"The individual rule that was checked by the prover was {args.rule}",
        calltrace_xml
    ]

    return _analyze_core(input_messages, initial_prompt, report_dir, args)

def analyze(
    args: AnalysisArgs
) -> int:
    """Analyze counterexamples, handling both local folders and URLs."""
    if _looks_like_url(args.folder):
        with _download_and_extract_report(args.folder) as report_dir:
            return _analyze_from_report(report_dir, args)
    else:
        report_dir = pathlib.Path(args.folder)
        return _analyze_from_report(report_dir, args)

def analyze_with_calltraces(
    input_messages: list[str],
    initial_prompt: str,
    args: AnalysisArgs
) -> int:
    """Run analysis workflow with custom calltraces and prompt.

    Entry point for external callers who want to run the analyzer
    with custom counterexample XMLs and prompts, bypassing report parsing.

    This is useful for:
    - Analyzing multiple violations from the same rule
    - Providing custom analysis prompts
    - Integrating the analyzer into other workflows

    Args:
        input_messages: List of input strings including contextual messages and XML calltraces.
            For example: ["Rule: myRule", "Context info...", "<calltrace>...</calltrace>"]
        initial_prompt: Custom initial prompt for the workflow. This sets up
            the context for the analysis (e.g., ecosystem-specific instructions).
        args: Configuration parameters. Note that args.folder must be a local
            folder path (not a URL) pointing to the Certora report directory.

    Returns:
        Exit code (0 for success, non-zero for errors)

    Raises:
        AssertionError: If args.folder appears to be a URL

    Example:
        >>> from analyzer import analyze_with_calltraces
        >>> args = MyArgs(
        ...     folder="/path/to/report",
        ...     ecosystem="evm",
        ...     rag_db="postgresql://...",
        ...     tokens=4096,
        ...     thinking_tokens=2048,
        ...     recursion_limit=30,
        ...     thread_id=None,
        ...     checkpoint_id=None,
        ...     quiet=False
        ... )
        >>> messages = ["Rule: myRule", "<calltrace>...</calltrace>"]
        >>> prompt = "Analyze this counterexample..."
        >>> result = analyze_with_calltraces(messages, prompt, args)
    """
    assert not _looks_like_url(args.folder), \
        "args.folder must be a local folder path, not a URL. Use analyze() to handle URLs."

    report_dir = pathlib.Path(args.folder)
    return _analyze_core(input_messages, initial_prompt, report_dir, args)

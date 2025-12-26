from typing import NotRequired, TypedDict, Iterator
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

from analyzer.types import AnalysisArgs, Ecosystem

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
    result: NotRequired[str]

analysis_output_tool = result_tool_generator(
    "result", 
    (str, "The textual analysis explaining the counterexample. You MAY use markdown in your output."),
    "Tool to communicate the result of your analysis.")


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

    args = parser.parse_args()
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

    (v_tools, _) = vfs_tools(
        ty=SimpleState,
        conf=VFSToolConfig(
            immutable=True,
            forbidden_read=r"^\..*$",
            fs_layer=str(report_dir / "inputs" / ".certora_sources")
        )
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

    process = ecosystem_params[args.ecosystem]

    initial_prompt = load_jinja_template("analyzer_tool_prompt.j2", **process)

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
        print(f"Chose thread id: {tid}")
    
    conf["configurable"]["thread_id"] = tid
    if args.checkpoint_id is not None:
        conf["configurable"]["checkpoint_id"] = args.checkpoint_id
    
    conf["recursion_limit"] = args.recursion_limit

    for (ty, d) in graph.stream(input=FlowInput(input=[
        f"The individual rule that was checked by the prover was {args.rule}",
        calltrace_xml
    ]), context=ExplainerContext(
        rag_db=PostgreSQLRAGDatabase(conn_string=args.rag_db, model=get_model(), skip_test=True)
    ), config=conf, stream_mode=["checkpoints", "updates"]):
        if ty == "checkpoints":
            assert isinstance(d, dict)
            print("current checkpoint: " + d["config"]["configurable"]["checkpoint_id"])
        else:
            if not args.quiet:
                print(d)

    print(graph.get_state({"configurable": {"thread_id": tid}}).values["result"])
    return 0

def analyze(
    args: AnalysisArgs
) -> int:
    """Analyze counterexamples, handling both local folders and URLs."""
    if _looks_like_url(args.folder):
        with _download_and_extract_report(args.folder) as report_dir:
            return _analyze_core(report_dir, args)
    else:
        report_dir = pathlib.Path(args.folder)
        return _analyze_core(report_dir, args)

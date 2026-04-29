from typing import NotRequired, TypedDict, Iterator, Iterable, TypeVar, overload, Any, Generic
import asyncio
import pathlib
import os
import tempfile
import tarfile
import urllib.request
import urllib.parse
from contextlib import contextmanager

import uuid

from langgraph.graph import MessagesState

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from composer.workflow.services import create_llm
import composer.prover.results as R
from composer.templates.loader import load_jinja_template

from composer.input.types import ModelOptions, LanggraphOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args

from graphcore.tools.vfs import fs_tools
from graphcore.graph import FlowInput, build_async_workflow
from graphcore.tools.results import result_tool_generator
from graphcore.utils import get_token_usage

from analyzer.types import Ecosystem, AnalysisArgs
from pydantic import BaseModel

class EcosystemConfig(TypedDict):
    spec_name: str
    spec_name_full: str
    spec_coda: NotRequired[str]
    token_example: str
    ecosystem_name: str
    spec_description: str
    language_name: str

T = TypeVar("T")


class SimpleState(MessagesState, Generic[T]):
    result: NotRequired[T]

def find_tree_view_node(stat: R.TreeViewStatus, context: pathlib.Path, target: R.RulePath) -> R.RuleResult | None:
    for r in stat.rules:
        if r.name != target.rule:
            continue
        for d in R.flatten_tree_view(context=context, path=R.RulePath(rule=r.name), r=r):
            if d.path == target:
                return d
    return None

_analysis_doc = """REQUIRED: You MUST call this tool to submit your final analysis.
Do NOT write your answer as plain text - the workflow cannot complete until you call this tool.
When you have reached a conclusion about the counterexample, call this tool immediately."""

_default_text = "The textual analysis explaining the counterexample. You MAY use markdown in your output."

_default_format = (str, _default_text)


def _accumulate_token_usage(update: dict, usage_dict: dict[str, int] | None) -> None:
    """Extract token usage from stream update messages and accumulate into usage_dict."""
    if usage_dict is None:
        return
    for node_output in update.values():
        if not isinstance(node_output, dict):
            continue
        for msg in node_output.get("messages", []):
            if isinstance(msg, AIMessage):
                usage = get_token_usage(msg)
                for key in usage:
                    usage_dict[key] = usage_dict.get(key, 0) + usage[key]  # type: ignore[operator]


def main() -> int:
    """CLI entry point for the analyzer."""
    import argparse
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

    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, LanggraphOptions)
    add_protocol_args(parser, RAGDBOptions)

    args = parser.parse_args()
    return asyncio.run(analyze(cast(AnalysisArgs, args)))

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

    res = urllib.parse.urlparse(
        url
    )

    path_component = pathlib.PurePosixPath(res.path).parts
    if not (len(path_component) == 4 and path_component[0] == "/" and path_component[1] == "output"):
        raise ValueError(f"{url} does not appear to be an `/output` url")

    job_id = path_component[3]

    qs = urllib.parse.parse_qs(res.query)
    if "anonymousKey" not in qs:
        raise ValueError(f"No anonymous key found in url {url}")
    
    anon_key = qs["anonymousKey"][0]
    
    zip_url = f"{res.scheme}://{res.netloc}/v1/domain/jobs/{job_id}/f/outputs?anonymousKey={anon_key}"
    
    with tempfile.TemporaryDirectory(prefix="certora_report_") as temp_dir:
        request = urllib.request.Request(zip_url, headers={"User-Agent": "curl/8.0"})
        
        with urllib.request.urlopen(request) as response:
            tar_path = os.path.join(temp_dir, "report.tar.gz")
            with open(tar_path, 'wb') as f:
                f.write(response.read())
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
        
        os.remove(tar_path)
        
        yield pathlib.Path(temp_dir, "TarName")

async def _analyze_core(
    input_messages: Iterable[str],
    initial_prompt: str,
    report_dir: pathlib.Path,
    args: AnalysisArgs,
    out_type: tuple[type[T], str] | type[T],
    token_usage: dict[str, int] | None = None
) -> T:
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

    d = out_type
    result_tool : BaseTool
    if isinstance(d, tuple):
        result_tool = result_tool_generator(
            "result",
            d,
            _analysis_doc
        )
    elif issubclass(d, BaseModel):
        result_tool = result_tool_generator(
            "result",
            d,
            _analysis_doc
        )
    else:
        raise ValueError(f"Invalid type parameter: {d}")
    
    v_tools = fs_tools(
        fs_layer=str(report_dir / "inputs" / ".certora_sources"),
        forbidden_read=r"^\..*$"
    )

    tools = [result_tool, *v_tools]

    if args.ecosystem == "evm":
        #import here to lazily load sentencetransformers
        from composer.tools.search import cvl_manual_search
        from composer.rag.models import get_model
        from composer.rag.db import get_rag_db
        tools.append(cvl_manual_search(await get_rag_db(args.rag_db, get_model())))

    llm = create_llm(args)

    system_prompt = load_jinja_template("analyzer_system_prompt.j2")

    # Wrap initial_prompt with cache_control for prompt caching
    initial_prompt_with_cache = {
        "type": "text",
        "text": initial_prompt,
        "cache_control": {"type": "ephemeral"}
    }

    graph = build_async_workflow(
        input_type=FlowInput,
        output_key="result",
        tools_list=tools,
        unbound_llm=llm,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt_with_cache,
        state_class=SimpleState
    )[0].compile(checkpointer=InMemorySaver())

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
    async for (ty, d) in graph.astream(input=FlowInput(input=list(input_messages)), config=conf, stream_mode=["checkpoints", "updates"]):
        if ty == "checkpoints":
            assert isinstance(d, dict)
            if not args.quiet:
                print("current checkpoint: " + d["config"]["configurable"]["checkpoint_id"])
        else:
            if isinstance(d, dict):
                _accumulate_token_usage(d, token_usage)
            if not args.quiet:
                print(d)

    return (await graph.aget_state({"configurable": {"thread_id": tid}})).values["result"]

async def _analyze_from_report(
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

    msg = await _analyze_core(input_messages, initial_prompt, report_dir, args, _default_format)
    print(msg)
    return 0

async def analyze(
    args: AnalysisArgs
) -> int:
    """Analyze counterexamples, handling both local folders and URLs."""
    if _looks_like_url(args.folder):
        with _download_and_extract_report(args.folder) as report_dir:
            return await _analyze_from_report(report_dir, args)
    else:
        report_dir = pathlib.Path(args.folder)
        return await _analyze_from_report(report_dir, args)

B = TypeVar("B", bound=BaseModel)

@overload
async def analyze_with_calltraces(
    input_messages: list[str],
    initial_prompt: str,
    args: AnalysisArgs,
    output: None = None,
    token_usage: dict[str, int] | None = None
) -> str:
    ...

@overload
async def analyze_with_calltraces(
    input_messages: list[str],
    initial_prompt: str,
    args: AnalysisArgs,
    output: tuple[type[T], str],
    token_usage: dict[str, int] | None = None
) -> T:
    ...

@overload
async def analyze_with_calltraces(
    input_messages: list[str],
    initial_prompt: str,
    args: AnalysisArgs,
    output: type[B],
    token_usage: dict[str, int] | None = None
) -> B:
    ...

async def analyze_with_calltraces(
    input_messages: list[str],
    initial_prompt: str,
    args: AnalysisArgs,
    output: tuple[type, str] | type | None = None,
    token_usage: dict[str, int] | None = None
) -> Any:
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
    return await _analyze_core(input_messages, initial_prompt, report_dir, args, output if output else _default_format, token_usage)

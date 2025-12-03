from typing import NotRequired
from dataclasses import dataclass
import pathlib

from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from langchain_anthropic import ChatAnthropic

from composer.rag.db import PostgreSQLRAGDatabase, DEFAULT_CONNECTION
from composer.rag.models import get_model
import composer.certora as _
from composer.tools.search import cvl_manual_search
import composer.prover.results as R
from composer.templates.loader import load_jinja_template

from graphcore.tools.vfs import VFSState, VFSToolConfig, vfs_tools
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.results import result_tool_generator

from analyzer.types import AnalysisArgs


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

    args = parser.parse_args()
    return analyze(cast(AnalysisArgs, args))


def analyze(
    args: AnalysisArgs
) -> int:
    report_dir = pathlib.Path(args.folder)
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

    tools = [cvl_manual_search, analysis_output_tool, *v_tools]

    llm = ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens_to_sample=4096,
        temperature=1,
        timeout=None,
        max_retries=2,
        stop=None,
        thinking={"type": "enabled", "budget_tokens": 2048},
        betas=["interleaved-thinking-2025-05-14", "context-management-2025-06-27"],
    )

    system_prompt = load_jinja_template("analyzer_system_prompt.j2")

    initial_prompt = load_jinja_template("analyzer_tool_prompt.j2")
    in_memory = InMemorySaver()
    graph = build_workflow(
        input_type=FlowInput,
        context_schema=ExplainerContext,
        output_key="result",
        tools_list=tools,
        unbound_llm=llm,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt,
        state_class=SimpleState
    )[0].compile(checkpointer=in_memory)

    id = "cex-analysis"

    for d in graph.stream(input=FlowInput(input=[
        f"The individual rule that was checked by the prover was {args.rule}",
        calltrace_xml
    ]), context=ExplainerContext(
        rag_db=PostgreSQLRAGDatabase(conn_string=DEFAULT_CONNECTION, model=get_model(), skip_test=True)
    ), config={"configurable": {"thread_id": id}}):
        if not args.quiet:
            print(d)

    print(graph.get_state({"configurable": {"thread_id": id}}).values["result"])
    return 0

from typing import NotRequired
from dataclasses import dataclass
import pathlib
import uuid
import json

from langgraph.graph import MessagesState
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.config import RunnableConfig

from composer.rag.db import PostgreSQLRAGDatabase, VACUITY_DEFAULT_CONNECTION
from composer.rag.models import get_model
from composer.tools.search import cvl_manual_search
from composer.templates.loader import load_jinja_template
from composer.workflow.factories import get_checkpointer

from graphcore.tools.vfs import VFSState, VFSToolConfig, vfs_tools
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.results import result_tool_generator

from vacuity_analyzer.types import VacuityAnalysisArgs


@dataclass
class VacuityExplainerContext:
    rag_db: PostgreSQLRAGDatabase


class VacuityState(VFSState, MessagesState):
    result: NotRequired[str]


vacuity_analysis_output_tool = result_tool_generator(
    "result",
    (str, """The textual analysis explaining the vacuity issue in the following structured format:

## Root Cause: [Brief title describing the main issue]
[2-3 sentence summary of what's causing the unsatisfiability]

## Detailed Analysis
### 1. **The Problematic Constraint Sequence**
[Trace through the [in UC] commands explaining which constraints conflict and why]

### 2. **Why This Creates Unsatisfiability** 
[Explain the logical contradiction and why these constraints cannot be satisfied together]

## Solution: [Title of the recommended fix]
[Detailed explanation of how to fix the issue, including:]
- Configuration changes (flags, settings)
- CVL rule modifications if needed
- Implementation changes if needed
- Flag explanations with rationale for values

## Summary
[Brief paragraph covering:]
- Issue Type: [Category like "Prover Configuration", "Specification Issue", etc.]
- Root Cause: [One line summary]
- Impact: [What this means for verification]
- Fix: [Concrete action items]
- Expected Result: [What should happen after the fix]

Use markdown formatting for readability."""),
    "Tool to communicate the result of your vacuity analysis in a structured format.")


def main() -> int:
    """CLI entry point for the vacuity analyzer."""
    import argparse
    import sys
    from typing import cast

    parser = argparse.ArgumentParser(
        description='Analyze Certora Prover unsat cores and identify vacuity issues.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vacuity-analyzer /path/to/report/Reports/unsat_core.txt
  vacuity-analyzer /path/to/report/Reports/unsat_core.txt --rule myRule
  vacuity-analyzer /path/to/report/Reports/unsat_core.txt --rule myRule --method myMethod
"""
    )

    parser.add_argument(
        'vacuity_txt_path',
        type=str,
        help='Path to the unsat core txt file'
    )

    parser.add_argument(
        '--rule',
        type=str,
        default=None,
        help='Name of the rule being analyzed (extracted from filename if not provided)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help='Optional method identifier. Can be either "method" or "contract.method" format (extracted from filename if not provided)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress intermediate output during analysis (only show final result)'
    )

    parser.add_argument(
        '--recursion-limit',
        type=int,
        default=30,
        help="The recursion limit to use for the vacuity analysis"
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
        default=VACUITY_DEFAULT_CONNECTION,
        help="RAG database connection string (defaults to extended_rag_db with prover documentation)"
    )

    args = parser.parse_args()
    return analyze(cast(VacuityAnalysisArgs, args))


def parse_vacuity_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Parse unsat core filename to extract rule and method.

    Expected format: UnsatCoreTAC-{rule}-{method}-{description}-{counter}.txt
    Or without method: UnsatCoreTAC-{rule}-{description}-{counter}.txt

    Returns: (rule, method) tuple where method may be None
    """
    # Remove .txt extension
    name = filename.replace('.txt', '')

    # Check if it starts with UnsatCoreTAC-
    if not name.startswith('UnsatCoreTAC-'):
        return None, None

    # Remove prefix and split by '-'
    parts = name[len('UnsatCoreTAC-'):].split('-')

    if len(parts) < 2:
        return None, None

    rule = parts[0]

    # Check if second part looks like a method name or description
    # Description parts typically start with keywords like "Satisfy", "Reaching", etc.
    # or contain "LP...RP" wrapper patterns
    if len(parts) >= 2:
        potential_method = parts[1]
        # If it starts with common description keywords, it's not a method
        description_keywords = ['Satisfy', 'Reaching', 'Unsat', 'Vacuous']
        if any(potential_method.startswith(kw) for kw in description_keywords):
            return rule, None
        # Otherwise, treat it as a method
        return rule, potential_method

    return rule, None


def analyze(args: VacuityAnalysisArgs) -> int:
    vacuity_txt_path = pathlib.Path(args.vacuity_txt_path).resolve()

    # Extract report directory - check if txt file is in Reports subdirectory
    if vacuity_txt_path.parent.name == "Reports":
        report_dir = vacuity_txt_path.parent.parent
    else:
        print(f"Error: Expected txt file to be in 'Reports' directory, but found: {vacuity_txt_path.parent}")
        return 1

    # Verify paths exist
    if not report_dir.exists():
        print(f"Report directory not found: {report_dir}")
        return 1

    if not vacuity_txt_path.exists():
        print(f"Unsat txt file not found: {vacuity_txt_path}")
        return 1

    # Extract rule and method from arguments or filename
    rule = args.rule
    method = args.method

    if rule is None or method is None:
        parsed_rule, parsed_method = parse_vacuity_filename(vacuity_txt_path.name)
        if rule is None:
            rule = parsed_rule
        if method is None:
            method = parsed_method

    if rule is None:
        print(f"Error: Could not determine rule name from arguments or filename: {vacuity_txt_path.name}")
        return 1

    print(f"Analyzing vacuity issue: {vacuity_txt_path.name}")
    print(f"Rule: {rule}")
    if method:
        print(f"Method: {method}")

    # Load unsat core data from txt file
    try:
        with open(vacuity_txt_path, 'r') as f:
            vacuity_txt_content = f.read()
    except FileNotFoundError as e:
        print(f"Error reading unsat core file: {e}")
        return 1
    
    # Set up VFS tools for accessing source files
    (v_tools, _) = vfs_tools(
        ty=VacuityState,
        conf=VFSToolConfig(
            immutable=True,
            forbidden_read=r"^\..*$",
            fs_layer=str(report_dir / "inputs" / ".certora_sources")
        )
    )

    tools = [cvl_manual_search, vacuity_analysis_output_tool, *v_tools]

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

    # Load custom prompts for vacuity analysis
    system_prompt = load_jinja_template("vacuity_system_prompt.j2")
    initial_prompt = load_jinja_template("vacuity_tool_prompt.j2")

    graph = build_workflow(
        input_type=FlowInput,
        context_schema=VacuityExplainerContext,
        output_key="result",
        tools_list=tools,
        unbound_llm=llm,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt,
        state_class=VacuityState
    )[0].compile(checkpointer=get_checkpointer())

    conf: RunnableConfig = {"configurable": {}}
    tid: str
    if args.thread_id is not None:
        tid = args.thread_id
    else:
        tid = f"vacuity-analysis-{uuid.uuid1().hex}"
        print(f"Chose thread id: {tid}")
    
    conf["configurable"]["thread_id"] = tid
    if args.checkpoint_id is not None:
        conf["configurable"]["checkpoint_id"] = args.checkpoint_id
    
    conf["recursion_limit"] = args.recursion_limit

    for (ty, d) in graph.stream(input=FlowInput(input=[
        f"The rule being analyzed is: {rule}",
        f"Method context: {method if method else 'N/A'}",
        f"Unsat core data:\n{vacuity_txt_content}"
    ]), context=VacuityExplainerContext(
        rag_db=PostgreSQLRAGDatabase(conn_string=args.rag_db, model=get_model(), skip_test=True)
    ), config=conf, stream_mode=["checkpoints", "updates"]):
        if ty == "checkpoints":
            assert isinstance(d, dict)
            print("current checkpoint: " + d["config"]["configurable"]["checkpoint_id"])
        else:
            if not args.quiet:
                print(d)

    final_result = graph.get_state({"configurable": {"thread_id": tid}}).values["result"]
    print(final_result)
    return 0
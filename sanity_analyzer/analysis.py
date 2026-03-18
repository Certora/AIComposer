from typing import NotRequired
import pathlib
import uuid
import json

from langgraph.graph import MessagesState
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from composer.input.parsing import add_protocol_args
from composer.input.types import ModelOptions, WorkflowOptions
from composer.rag.db import PostgreSQLRAGDatabase, SANITY_DEFAULT_CONNECTION
from composer.rag.models import get_model
from composer.tools.search import cvl_manual_search
from composer.templates.loader import load_jinja_template
from composer.workflow.factories import get_checkpointer
from composer.workflow.services import create_llm, get_memory

from composer.tools.thinking import get_rough_draft_tools, RoughDraftState

from graphcore.tools.memory import memory_tool
from graphcore.tools.vfs import fs_tools
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.results import result_tool_generator

from sanity_analyzer.types import SanityAnalysisArgs


class SanityState(MessagesState, RoughDraftState):
    result: NotRequired[str]

class SanityAnalysisMitigation(BaseModel):
    config_changes: list[tuple[str,str]] = Field(description="If there is a fix via changing the configuration, list the necessary config values as tuples of key and value.")
class SanityAnalysisResult(BaseModel):
    issue_type: str = Field(description='Category of the cause of the sanity issue like "Prover Configuration", "Specification Issue", etc.')
    mitigation_options: list[SanityAnalysisMitigation] = Field(description="A list of possible fixes that can be expressed as a SanityAnalysisMitigation. Can be empty if the fix is of a more complicated form.")
    short_summary: str = Field(description="A short summary of the issue and possible fixes.")
    root_cause: str = Field(description="""Explanation of the root cause of the issue of the form:
[Brief title describing the main issue]
[2-3 sentence summary of what's causing the unsatisfiability]""")
    detailed_analysis: str = Field(description="""Textual analysis giving more details about the issue in the form:
### 1. **The Problematic Constraint Sequence**
[Trace through the [in UC] commands explaining which constraints conflict and why]

### 2. **Why This Creates Unsatisfiability**
[Explain the logical contradiction and why these constraints cannot be satisfied together]""")
    solution: str = Field(description="""Explanation of how to solve the issue of the form:
## Solution: [Title of the recommended fix]
[Detailed explanation of how to fix the issue, including:]
- Configuration changes (flags, settings)
- CVL rule modifications if needed
- Implementation changes if needed
- Flag explanations with rationale for values""")

    def format(self) -> str:
        lines = [
            f"# Sanity Analysis Result",
            f"",
            f"**Issue Type:** {self.issue_type}",
            f"",
            f"## Summary",
            f"",
            self.short_summary,
            f"",
            f"## Root Cause",
            f"",
            self.root_cause,
            f"",
            f"## Detailed Analysis",
            f"",
            self.detailed_analysis,
            f"",
            self.solution,
        ]
        if self.mitigation_options:
            lines += [f"", f"## Mitigation Options", f""]
            for i, opt in enumerate(self.mitigation_options, 1):
                if opt.config_changes:
                    lines.append(f"**Option {i}:**")
                    lines.append("```")
                    for (key, value) in opt.config_changes:
                        lines.append(f"{key} = {value}")
                    lines.append("```")
        return "\n".join(lines)

sanity_analysis_output_tool = result_tool_generator(
    "result",
    SanityAnalysisResult,
    "Tool to communicate the result of your sanity analysis in a structured format.",
    validator=(
        SanityState,
        lambda state, _result, _tool_call_id: (
            "You must call read_rough_draft before submitting your final result."
            if state.get("memory") and not state["did_read"]
            else None
        ),
    ),
)


def main() -> int:
    """CLI entry point for the sanity analyzer."""
    import argparse
    import sys
    from typing import cast

    parser = argparse.ArgumentParser(
        description='Analyze Certora Prover unsat cores and identify sanity issues.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sanity-analyzer /path/to/report/Reports/UnsatCoreTAC-myRule-....txt
  sanity-analyzer /path/to/report/Reports/UnsatCoreTAC-myRule-....txt --rule myRule
  sanity-analyzer /path/to/report/Reports/UnsatCoreTAC-myRule-....txt --rule myRule --method myMethod
"""
    )

    parser.add_argument(
        'unsat_core_txt_path',
        type=str,
        help='Path to the UnsatCoreTAC txt file'
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

    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, WorkflowOptions)
    parser.set_defaults(rag_db=SANITY_DEFAULT_CONNECTION, model="claude-sonnet-4-5-20250929", interleaved_thinking=True)

    args = parser.parse_args()
    details = analyze(cast(SanityAnalysisArgs, args))
    if details:
        print(details.format())
        return 0
    return 1


def parse_unsat_core_filename(filename: str) -> tuple[str | None, str | None]:
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


def analyze(args: SanityAnalysisArgs) -> SanityAnalysisResult | None:
    unsat_core_txt_path = pathlib.Path(args.unsat_core_txt_path).resolve()

    # Extract report directory - check if txt file is in Reports subdirectory
    if unsat_core_txt_path.parent.name == "Reports":
        report_dir = unsat_core_txt_path.parent.parent
    else:
        print(f"Error: Expected txt file to be in 'Reports' directory, but found: {unsat_core_txt_path.parent}")
        return None

    # Verify paths exist
    if not report_dir.exists():
        print(f"Report directory not found: {report_dir}")
        return None

    if not unsat_core_txt_path.exists():
        print(f"Unsat txt file not found: {unsat_core_txt_path}")
        return None

    # Extract rule and method from arguments or filename
    rule = args.rule
    method = args.method

    if rule is None or method is None:
        parsed_rule, parsed_method = parse_unsat_core_filename(unsat_core_txt_path.name)
        if rule is None:
            rule = parsed_rule
        if method is None:
            method = parsed_method

    if rule is None:
        print(f"Error: Could not determine rule name from arguments or filename: {unsat_core_txt_path.name}")
        return None

    print(f"Analyzing sanity issue: {unsat_core_txt_path.name}")
    print(f"Rule: {rule}")
    if method:
        print(f"Method: {method}")

    # Load unsat core data from txt file
    try:
        with open(unsat_core_txt_path, 'r') as f:
            unsat_core_txt_content = f.read()
    except FileNotFoundError as e:
        print(f"Error reading unsat core file: {e}")
        return None

    v_tools = fs_tools(
        fs_layer=str(report_dir / "inputs" / ".certora_sources"),
        forbidden_read=r"^\..*$"
    )

    tid = args.thread_id if args.thread_id is not None else f"sanity-analysis-{uuid.uuid1().hex}"
    if args.thread_id is None:
        print(f"Chose thread id: {tid}")

    rag_db = PostgreSQLRAGDatabase(conn_string=args.rag_db, model=get_model(), skip_test=True)
    tools = [cvl_manual_search(rag_db), sanity_analysis_output_tool, *get_rough_draft_tools(SanityState), *v_tools]
    if args.memory_tool:
        tools.append(memory_tool(get_memory(tid, "sanity")))

    llm = create_llm(args)

    # Load custom prompts for sanity analysis
    system_prompt = load_jinja_template("sanity_system_prompt.j2")
    initial_prompt = load_jinja_template("sanity_tool_prompt.j2")

    graph = build_workflow(
        input_type=FlowInput,
        output_key="result",
        tools_list=tools,
        unbound_llm=llm,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt,
        state_class=SanityState
    )[0].compile(checkpointer=get_checkpointer())

    conf: RunnableConfig = {"configurable": {}}

    conf["configurable"]["thread_id"] = tid
    if args.checkpoint_id is not None:
        conf["configurable"]["checkpoint_id"] = args.checkpoint_id

    conf["recursion_limit"] = args.recursion_limit

    for (ty, d) in graph.stream(input=FlowInput(input=[
        f"The rule being analyzed is: {rule}",
        f"Method context: {method if method else 'N/A'}",
        f"Unsat core data:\n{unsat_core_txt_content}"
    ]), config=conf, stream_mode=["checkpoints", "updates"]):
        if ty == "checkpoints":
            assert isinstance(d, dict)
            print("current checkpoint: " + d["config"]["configurable"]["checkpoint_id"])
        else:
            if not args.quiet:
                print(d)

    final_result = graph.get_state({"configurable": {"thread_id": tid}}).values["result"]
    return final_result

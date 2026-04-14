"""
Analysis stage — agentic CVL rule extraction per source tree.

For each cloud run source tree, an agent browses the downloaded sources
using fs_tools to locate CVL rules, extract their source code, and
document their verification mechanism.
"""

from typing import NotRequired
from pathlib import Path
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from graphcore.graph import Builder, FlowInput, MessagesState
from graphcore.tools.vfs import fs_tools

from composer.corpus.models import (
    RuleRef, CorpusEntry, UnmatchedRule,
)
from composer.spec.graph_builder import bind_standard
from composer.spec.code_explorer import code_explorer_tool
from composer.templates.loader import load_jinja_template


# ---------------------------------------------------------------------------
# LLM output model — only new data the agent produces
# ---------------------------------------------------------------------------

class AnalyzedRule(BaseModel):
    """A single rule that was successfully located and analyzed.

    Contains only data the agent discovers by reading code — contextual
    data (property group info, status, etc.) is stitched in by the caller.
    """
    rule_name: str = Field(description="The CVL rule/invariant name (must match a name from the input)")
    cvl_code: str = Field(description="The full CVL source code of the rule")
    spec_file: str = Field(description="Path to the .spec file containing this rule (relative to repo root)")
    mechanism: str = Field(
        description=(
            "How the rule achieves its verification goal — ghost variables, hooks, "
            "invariant preservation strategy, etc."
        )
    )
    implementation_notes: str = Field(
        description="Notable implementation details: helper functions, dispatchers, summarizations, etc."
    )
    commentary: str = Field(
        description="Why this matters to the protocol."
    )
    property_description: str = Field(description="A brief description of the property being checked by the rule/invariant.")


class UnmatchedRuleResult(BaseModel):
    """A rule that could not be located in the repository."""
    rule_name: str = Field(description="The rule name from the report")
    reason: str = Field(description="Why the rule could not be found")


class SourceTreeAnalysisResult(BaseModel):
    """LLM agent output for a single source tree.

    Contains only what the agent discovers — the caller assembles
    full CorpusEntry objects by combining this with existing context.
    """
    analyzed: list[AnalyzedRule] = Field(description="Rules that were found and analyzed")
    unmatched: list[UnmatchedRuleResult] = Field(description="Rules that could not be located")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _AnalysisST(MessagesState):
    result: NotRequired[SourceTreeAnalysisResult]

_LIBRARY_INDICATORS = frozenset({
    "openzeppelin-contracts", "openzeppelin-contracts-upgradeable", "@openzeppelin",
    "solady", "solmate", "forge-std", "ds-test",
    "prb-math", "prb-test", "prb-proxy", "erc4626-tests",
})

def lib_dir_is_packages(path: str) -> bool:
    if (lib_dir := Path(path) / "lib").exists():
        for li in _LIBRARY_INDICATORS:
            if (lib_dir / li).exists():
                return True
    return False

def _forbid_read_re(
    path: str
) -> str:
    forbid_node_modules = "^node_modules/.*$"
    if not lib_dir_is_packages(path):
        return forbid_node_modules
    return f"({forbid_node_modules})|(^lib/.*$)"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@dataclass
class Env:
    base_source_tools: tuple[BaseTool, ...]
    builder: Builder[None, None, None]

    @property
    def has_source(self) -> bool:
        return True
    

async def analyze_source_tree(
    rules: list[RuleRef],
    source_path: str,
    protocol_description: str,
    llm: BaseChatModel,
    extra_tools: list[BaseTool] | None = None,
) -> tuple[list[CorpusEntry], list[UnmatchedRule]]:
    """Run an agent to analyze rules against a single cloud run source tree.

    The agent uses fs_tools to grep for rule names, read spec files, and
    produce structured analysis of each rule. It also has access to a code
    explorer sub-agent and any additional tools (CVL manual, CVL research)
    passed via ``extra_tools``.

    Rules may come from different property groups but share the same
    source tree (cloud run). The raw agent output is combined with
    each rule's property group context to produce CorpusEntry objects.

    Args:
        rules: Rule references pointing into their parent PropertyGroups.
        source_path: Path to the downloaded source directory.
        protocol_description: Brief description of the protocol.
        llm: LLM instance for the agent.
        extra_tools: Additional tools (CVL manual search, CVL research, etc.).

    Returns:
        Tuple of (corpus entries, unmatched rules).
    """
    repo_fs = fs_tools(source_path, forbidden_read=_forbid_read_re(source_path))

    build = Builder().with_llm(llm).with_loader(load_jinja_template)

    explorer = code_explorer_tool(Env(tuple(repo_fs), build))

    all_tools: list[BaseTool] = [*repo_fs, explorer]
    if extra_tools:
        all_tools.extend(extra_tools)

    graph = bind_standard(
        build, _AnalysisST,
    ).with_tools(
        all_tools
    ).with_input(
        FlowInput,
    ).with_sys_prompt_template(
        "corpus_analysis_system.j2",
    ).with_initial_prompt_template(
        "corpus_analysis_prompt.j2",
        protocol_description=protocol_description,
        rules=rules,
    ).compile_async(checkpointer=InMemorySaver())

    st = await graph.ainvoke(
        FlowInput(input=[]),
        config={
            "configurable": {"thread_id": "analysis"},
            "recursion_limit": 100,
        },
    )

    assert "result" in st, "Analysis agent did not produce a result"
    raw_result = st["result"]
    assert isinstance(raw_result, SourceTreeAnalysisResult)

    # Build lookup for O(1) access by rule name
    refs_by_name = {ref.rule.name: ref for ref in rules}

    # Assemble CorpusEntry objects by combining agent output with existing context
    entries: list[CorpusEntry] = []
    for ar in raw_result.analyzed:
        ref = refs_by_name.get(ar.rule_name)
        if ref is not None:
            entries.append(CorpusEntry(
                rule_name=ar.rule_name,
                property_id=ref.group.id,
                property_title=ref.group.title,
                property_description=ref.group.description,
                rule_description=ref.rule.description,
                status=ref.rule.status,
                assumptions=ref.group.assumptions,
                cvl_code=ar.cvl_code,
                spec_file=ar.spec_file,
                mechanism=ar.mechanism,
                implementation_notes=ar.implementation_notes,
                commentary=ar.commentary
            ))

    unmatched: list[UnmatchedRule] = []
    for ur in raw_result.unmatched:
        ref = refs_by_name.get(ur.rule_name)
        if ref is not None:
            unmatched.append(UnmatchedRule(
                rule=ref.rule,
                property_id=ref.group.id,
                reason=ur.reason,
            ))

    return entries, unmatched

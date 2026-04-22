"""
Property generation agent: extracts security properties from application components.

Parameterized by source availability via AnalysisInput tuple.
"""

from typing import NotRequired, Protocol, override, Literal
import re
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langgraph.types import interrupt, Command

from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage

from graphcore.graph import MessagesState, FlowInput
from graphcore.tools.schemas import WithImplementation

from composer.spec.context import WorkflowContext, CacheKey, ComponentGroup
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.system_model import ContractComponentInstance
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.tool_env import BasicAgentTools
from composer.io.conversation import ConversationContextProvider
from composer.spec.refinement import refinement_loop, EndConversation, SyncStateUpdateTool
from composer.templates.loader import load_jinja_template
from composer.ui.tool_display import tool_display
from composer.spec.util import string_hash

from rich.markdown import Markdown
from rich.console import Group
from rich.text import Text

class _BugAnalysisCache(BaseModel):
    items: list[PropertyFormulation]

class _AgentHistory(_BugAnalysisCache):
    agent_conversation: list[AnyMessage]

def bug_analysis_key(
    threat_model: dict | str | None
) -> CacheKey[ComponentGroup, _BugAnalysisCache]:
    if threat_model is None:
        return CacheKey[ComponentGroup, _BugAnalysisCache]("bug_analysis")
    return CacheKey[ComponentGroup, _BugAnalysisCache]("bug_analysis-tm-" + string_hash(str(threat_model)))

AGENT_RESULT_KEY = CacheKey[_BugAnalysisCache, _AgentHistory]("agent_bug_analysis")

DESCRIPTION = "Property extraction"

class BugEnvironment(BasicAgentTools, Protocol):
    @property
    def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def has_source(self) -> bool:
        ...

class RefinementState(MessagesState):
    properties: list[PropertyFormulation]

def _get_initial_prompt(
    context: ContractComponentInstance,
    has_source: bool
) -> str:
    return load_jinja_template(
        "property_analysis_prompt.j2",
        context=context,
        has_source=has_source
    )

@tool_display("Ending conversation...", None)
class Exit(WithImplementation[str]):
    """
    Call this when the user has indicated they are happy with the properties you have generated
    """
    @override
    def run(self) -> str:
        return interrupt(EndConversation())

@tool_display("Updating requirements", None)
class SetRequirements(SyncStateUpdateTool[list[PropertyFormulation]]):
    """
    Called with the new properties as requested by the user
    """

    new_requirements: list[PropertyFormulation] = Field(description="The new requirements after taking into account user feedback.")

    @override
    def run(self) -> Command:
        return self._update(self.new_requirements)


# GitHub-ish dark theme
LINE_DEL = "red on #3a1d1d"
LINE_ADD = "green on #1d3a1d"
WORD_DEL = "bold white on #802020"
WORD_ADD = "bold white on #206020"
DIM      = "grey50"


def _word_diff(a: str, b: str) -> tuple[Text, Text]:
    """Return (minus, plus) Text with word-level highlights."""
    a_toks = re.findall(r"\S+|\s+", a)
    b_toks = re.findall(r"\S+|\s+", b)
    sm = SequenceMatcher(None, a_toks, b_toks)

    minus, plus = Text(), Text()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_chunk = "".join(a_toks[i1:i2])
        b_chunk = "".join(b_toks[j1:j2])
        if tag == "equal":
            minus.append(a_chunk, style=LINE_DEL)
            plus.append(b_chunk, style=LINE_ADD)
        elif tag == "delete":
            minus.append(a_chunk, style=WORD_DEL)
        elif tag == "insert":
            plus.append(b_chunk, style=WORD_ADD)
        elif tag == "replace":
            minus.append(a_chunk, style=WORD_DEL)
            plus.append(b_chunk, style=WORD_ADD)
    return minus, plus


def _diff_replace_block(a_block: list[str], b_block: list[str]) -> list[Text]:
    """Nested line-level diff inside an outer 'replace' block.

    Only inner 'replace' pairs get word-diffed; inner insert/delete become
    whole-line adds/removes. This avoids noisy word-diffs when lines shift.
    """
    out: list[Text] = []
    sm = SequenceMatcher(None, a_block, b_block)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in a_block[i1:i2]:
                out.append(Text(f"  {line}", style=DIM))
        elif tag == "delete":
            for line in a_block[i1:i2]:
                out.append(Text(f"- {line}", style=LINE_DEL))
        elif tag == "insert":
            for line in b_block[j1:j2]:
                out.append(Text(f"+ {line}", style=LINE_ADD))
        elif tag == "replace":
            a_lines, b_lines = a_block[i1:i2], b_block[j1:j2]
            n = min(len(a_lines), len(b_lines))
            for k in range(n):
                m, p = _word_diff(a_lines[k], b_lines[k])
                out.append(Text("- ", style=LINE_DEL) + m)
                out.append(Text("+ ", style=LINE_ADD) + p)
            for line in a_lines[n:]:
                out.append(Text(f"- {line}", style=LINE_DEL))
            for line in b_lines[n:]:
                out.append(Text(f"+ {line}", style=LINE_ADD))
    return out


def diff_states(state_a: list[str], state_b: list[str]) -> Group:
    sm = SequenceMatcher(None, state_a, state_b)
    out: list[Text] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in state_a[i1:i2]:
                out.append(Text(f"  {line}", style=DIM))
        elif tag == "delete":
            for line in state_a[i1:i2]:
                out.append(Text(f"- {line}", style=LINE_DEL))
        elif tag == "insert":
            for line in state_b[j1:j2]:
                out.append(Text(f"+ {line}", style=LINE_ADD))
        elif tag == "replace":
            out.extend(_diff_replace_block(state_a[i1:i2], state_b[j1:j2]))

    return Group(*out)


async def _run_bug_analysis_inner(
    agent_component_analysis: WorkflowContext[_AgentHistory],
    env: BugEnvironment,
    component: ContractComponentInstance,
    threat_model: str | dict | None
) -> _AgentHistory:
    if (cached := await agent_component_analysis.cache_get(_AgentHistory)) is not None:
        return cached
    
    builder = env.builder

    class BugAnalysisInput(FlowInput, RoughDraftState):
        pass

    class ST(MessagesState, RoughDraftState):
        result: NotRequired[list[PropertyFormulation]]

    d = bind_standard(
        builder, ST, "The security properties you have extracted about the component"
    ).with_input(
        BugAnalysisInput
    ).with_initial_prompt(
        _get_initial_prompt(component, env.has_source)
    ).with_tools(
        get_rough_draft_tools(ST)
    ).with_tools(
        env.bug_analysis_tools
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols"
    ).compile_async()

    extra_input = []

    if threat_model is not None:
        extra_input = [
            "In addition, a coworker has already written a 'threat model' for this application, which may include vulnerabilities/issues that"
            "are common in this type of application. This threat model is written for the entire application (not just the component you are analyzing) "
            "so some of the issues/vulnerabilities/attacks may not be relevant to your analysis. Do *NOT* overfit to this threat model; carefully "
            "analyze what content of the provided threat model is worth considering vs out of scope. Further, this threat model is just a starting point, "
            "you should ALSO look for threats *not* mentioned in this document.",
            threat_model
        ]

    r = await run_to_completion(
        d,
        BugAnalysisInput(input=extra_input, memory=None, did_read=False),
        thread_id=agent_component_analysis.thread_id,
        description=DESCRIPTION,
    )
    assert "result" in r

    result: list[PropertyFormulation] = r["result"]

    to_ret = _AgentHistory(items=result, agent_conversation=r["messages"])

    await agent_component_analysis.cache_put(to_ret)
    return to_ret

async def run_bug_analysis(
    ctx: WorkflowContext[ComponentGroup],
    env: BugEnvironment,
    component: ContractComponentInstance,
    threat_model: str | dict | None = None,
    refinement: ConversationContextProvider | None = None
) -> list[PropertyFormulation]:
    """
    Extract security properties for a component.
    """

    component_analysis = ctx.child(bug_analysis_key(threat_model))
    if (cached := await component_analysis.cache_get(_BugAnalysisCache)) is not None:
        return cached.items
    
    agent_attempt = await _run_bug_analysis_inner(
        component_analysis.child(AGENT_RESULT_KEY),
        env,
        component,
        threat_model
    )
    if refinement is None:
        to_ret = agent_attempt.items
        await component_analysis.cache_put(_BugAnalysisCache(items=to_ret))
        return to_ret

    msg_history = agent_attempt.agent_conversation
    assert isinstance(msg_history[0], SystemMessage) and isinstance(msg_history[-1], ToolMessage)
    import uuid
    edited_history = [
        SystemMessage(load_jinja_template("bug_refinement_chat_system_prompt.j2")),
        *msg_history[1:],
        AIMessage("<task-complete>", id=uuid.uuid4().hex)
    ]

    def sort_to_string(
        s: Literal["attack_vector", "invariant", "safety_property"]
    ) -> str:
        match s:
            case "attack_vector":
                return "Attack Vector"
            case "invariant":
                return "Invariant"
            case "safety_property":
                return "Safety Property"

    def property_as_text(
        prop: PropertyFormulation
    ) -> str:
        return f"* [{sort_to_string(prop.sort)}] {prop.description}"

    def property_as_md(
        prop: PropertyFormulation
    ) -> str:
        sort_str = sort_to_string(prop.sort)
        return f"* \\[{sort_str}\\] {prop.description}"
    
    def properties_as_text(
        l: list[PropertyFormulation]
    ) -> list[str]:
        return [ property_as_text(p) for p in l ]

    def properties_as_md(
        l: list[PropertyFormulation]
    ) -> list[str]:
        return [ property_as_md(p) for p in l ]
    
    def render_properties_as_md(
        l: list[PropertyFormulation]
    ) -> Markdown:
        md = "## Current Properties\n"
        return Markdown(md + "\n".join(properties_as_md(l)))
    
    async with refinement(render_properties_as_md(agent_attempt.items)) as client:
        res = await refinement_loop(
            llm=env.llm,
            client=client,
            init_messages=edited_history,
            init_data=agent_attempt.items,
            tools=[*env.bug_analysis_tools, Exit.as_tool("finalize_properties"), SetRequirements.as_tool("update_requirements")],
            state_renderer=render_properties_as_md,
            diff_renderer=lambda a, b: \
                Group(
                    Text("Properties changed"),
                    diff_states(properties_as_text(a), properties_as_text(b))
                )
        )
    to_ret = res["extra_data"]
    await component_analysis.cache_put(_BugAnalysisCache(items = to_ret))
    return to_ret

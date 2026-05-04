"""
Property generation agent: extracts security properties from application components.

Parameterized by source availability via AnalysisInput tuple.
"""

from typing import NotRequired, Protocol, Callable, AsyncContextManager

from pydantic import BaseModel

from langchain_core.tools import BaseTool

from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage

from graphcore.graph import MessagesState, FlowInput

from composer.spec.context import WorkflowContext, CacheKey, ComponentGroup
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.prop import PropertyFormulation
from composer.spec.system_model import ContractComponentInstance
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.tool_env import BasicAgentTools
from composer.io.conversation import ConversationClient
from composer.spec.refinement import refinement_loop
from composer.templates.loader import load_jinja_template


class _BugAnalysisCache(BaseModel):
    items: list[PropertyFormulation]

class _AgentHistory(_BugAnalysisCache):
    agent_conversation: list[AnyMessage]


BUG_ANALYSIS_KEY = CacheKey[ComponentGroup, _BugAnalysisCache]("bug_analysis")

AGENT_RESULT_KEY = CacheKey[_BugAnalysisCache, _AgentHistory]("agent_bug_analysis")

DESCRIPTION = "Property extraction"

class BugEnvironment(BasicAgentTools, Protocol):
    @property
    def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def has_source(self) -> bool:
        ...

type ConversationContextProvider = Callable[[str], AsyncContextManager[ConversationClient]]

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

async def _run_bug_analysis_inner(
    agent_component_analysis: WorkflowContext[_AgentHistory],
    env: BugEnvironment,
    component: ContractComponentInstance,
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

    r = await run_to_completion(
        d,
        BugAnalysisInput(input=[], memory=None, did_read=False),
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
    refinement: ConversationContextProvider | None = None
) -> list[PropertyFormulation]:
    """
    Extract security properties for a component.
    """

    component_analysis = ctx.child(BUG_ANALYSIS_KEY)
    if (cached := await component_analysis.cache_get(_BugAnalysisCache)) is not None:
        return cached.items
    
    agent_attempt = await _run_bug_analysis_inner(
        component_analysis.child(AGENT_RESULT_KEY),
        env,
        component
    )
    if refinement is None:
        to_ret = agent_attempt.items
        await component_analysis.cache_put(_BugAnalysisCache(items=to_ret))
        return to_ret

    msg_history = agent_attempt.agent_conversation
    assert isinstance(msg_history[0], SystemMessage) and isinstance(msg_history[-1], ToolMessage)
    edited_history = [
        SystemMessage(load_jinja_template("bug_refinement_chat_system_prompt.j2")),
        *msg_history[1:],
        AIMessage("<task-complete>")
    ]

    async with refinement("Entering refinement loop") as client:
        res = await refinement_loop(
            llm=env.llm,
            client=client,
            init_messages=edited_history,
            init_data=agent_attempt.items,
            tool_display=None,
            tools=[]
        )
    to_ret = res["extra_data"]
    await component_analysis.cache_put(_BugAnalysisCache(items = to_ret))
    return to_ret

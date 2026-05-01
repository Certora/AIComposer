from typing_extensions import NotRequired, Annotated, TypedDict
from typing import Literal, Any, cast
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field

from graphcore.tools.memory import AsyncMemoryBackend, async_memory_tool
from graphcore.tools.schemas import WithAsyncDependencies, WithInjectedId, WithInjectedState
from graphcore.graph import FlowInput, build_async_workflow, WithToolCallId, tool_state_update, Builder
from graphcore.tools.results import result_tool_generator
from graphcore.tools.vfs import VFSState

from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

from composer.templates.loader import load_jinja_template
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.core.state import AIComposerState
from composer.core.validation import REQS_VALIDATION as req_key
from composer.core.context import AIComposerContext, compute_state_digest
from composer.io.context import run_graph
from composer.ui.tool_display import tool_display
from composer.spec.util import uniq_thread_id

class JudgeExtra(RoughDraftState):
    vfs: dict[str, str]
    orig_reqs: list[str]

class JudgeInput(FlowInput, JudgeExtra):
    pass

ClassificationType = Literal["SATISFIED", "LIKELY", "PARTIAL", "VIOLATED"]

class RequirementAnalysis(BaseModel):
    """
    The analysis result for one of the requirements.
    """
    classification: ClassificationType = Field(description="The final classification on whether " \
    "the implementation satisfies the requirement.")

    requirement: str = Field(description="The original requirement text against which the implementation was judged. Do NOT include the numeric prefix (e.g., \"1. \")")

    requirement_number : int = Field(description="The requirement number.")

    commentary: str | None = Field(description="Any commentary or explanation for the classification of this requirement. In the case of" \
    "PARTIAL or VIOLATED classifications, do NOT suggest code changes: simply explain the deficiencies in the implementation.")

class JudgeResult(BaseModel):
    judgement_result: list[RequirementAnalysis] = Field(description="A list of analysis results, with one element per original requirement.")

class JudgeState(MessagesState, JudgeExtra):
    result: NotRequired[JudgeResult]

def _gen_workflow(
    builder: Builder[None, None, None],
    mem: AsyncMemoryBackend
) -> CompiledStateGraph[JudgeState, None, JudgeInput, Any]:
    res = result_tool_generator(
        "result",
        result_schema=JudgeResult,
        doc="""
Called with the result of your analysis. Use this to signal that your analysis is complete
and communicate it back to the user.

*IMPORTANT*: Once you call this tool, this workflow will end. You MUST perform any memory operations
BEFORE calling this tool.
""",
        validator=(JudgeState, judge_res_checker)
    )
    return (
        builder
        .with_output_key("result")
        .with_initial_prompt_template("req_judge_prompt.j2")
        .with_sys_prompt_template("req_role_prompt.j2")
        .with_tools([res, *get_rough_draft_tools(JudgeState), async_memory_tool(mem)])
        .with_state(JudgeState)
        .with_input(JudgeInput)
    ).compile_async()

def _format_result(
    r: JudgeResult,
    skipped: set[int]
) -> str:
    res_list = []
    for req_res in r.judgement_result:
        buff = "<result>"
        buff += f"<requirement>{req_res.requirement}</requirement>\n"
        if req_res.requirement_number in skipped:
            buff += "<classification>IGNORED</classification></result>"
            continue
        buff += f"<classification>{req_res.classification}</classification>\n"
        if req_res.commentary:
            buff += f"<comments>{req_res.commentary}</comments>\n"
        buff += "</result>"
        res_list.append(buff)
    return "\n".join(res_list)

def judge_res_checker(
    st: JudgeState,
    r: JudgeResult,
    _: str
) -> str | None:
    if not st["did_read"]:
        return "Completion REJECTED: You must read your rough draft before submitting. Call read_rough_draft first."
    reqs = st["orig_reqs"]
    if len(reqs) != len(r.judgement_result):
        return f"Completion REJECTED: Incorrect number of requirement results: expected {len(reqs)} received {len(r.judgement_result)}"
    seen_nums = set()
    for j in r.judgement_result:
        if j.requirement_number in seen_nums:
            return f"Completion REJECTED: Already seen judgment for {j.requirement_number}"
        seen_nums.add(j.requirement_number)
        if j.requirement_number not in range(1, len(reqs) + 1):
            return f"Completion REJECTED: Requirement number {j.requirement_number} is not valid"
        if j.requirement != reqs[j.requirement_number - 1]:
            return f"Completion REJECTED: Requirement text `{j.requirement}` does not match the original text: `{reqs[j.requirement_number - 1]}`"
    return None

@dataclass
class _JudgeToolDeps:
    implementation_graph: CompiledStateGraph[JudgeState, None, JudgeInput, Any]
    orig_reqs: list[str]

@tool_display("Evaluating requirements", "Requirements evaluation")
class RequirementsEvalTool(WithAsyncDependencies[str | Command, _JudgeToolDeps], WithInjectedState[AIComposerState], WithInjectedId):
    __doc__ = f"""
Query an oracle to determine if the generated implementation meets the requirements list
provided.

Each requirement is evaluated against the current implementation and assigned a classification:
{load_jinja_template("req_classifications.j2")}

If any requirements are classified as PARTIAL or VIOLATED, you must address this feedback.
    """

    async def run(self) -> Command | str:
        with self.tool_deps() as deps:
            skipped = self.state["skipped_reqs"]
            reqs = deps.orig_reqs
            state = self.state
            req_list = "\n".join([f"{i}. {r}" for (i, r) in enumerate(deps.orig_reqs, start = 1)])
            compiled_graph = deps.implementation_graph

            judge_config: RunnableConfig = {"configurable": {"thread_id": uniq_thread_id("requirements-judge")}}
            judge_state = await run_graph(
                compiled_graph,
                None,
                JudgeInput(input=[req_list], vfs=state["vfs"], orig_reqs=reqs, memory=None, did_read=False),
                judge_config,
                description="Requirements evaluation",
                within_tool=self.tool_call_id
            )

            assert "result" in judge_state
            res = judge_state["result"]

            all_satisfied = True
            for j in res.judgement_result:
                if j.classification != "LIKELY" and j.classification != "SATISFIED":
                    if j.requirement_number not in skipped:
                        all_satisfied = False
                        break

            formatted_res = _format_result(judge_state["result"], skipped)
            if not all_satisfied:
                return formatted_res

            digest = compute_state_digest(
                state=state
            )
            return tool_state_update(
                self.tool_call_id, formatted_res, validation={
                    str(req_key): digest
                }
            )

def get_judge_tool(
    builder_with_tools: Builder[None, None, None],
    reqs: list[str],
    mem: AsyncMemoryBackend
) -> BaseTool:
    
    workflow = _gen_workflow(builder_with_tools, mem)
    return RequirementsEvalTool.bind(_JudgeToolDeps(
        implementation_graph=workflow,
        orig_reqs=reqs
    )).as_tool("requirements_evaluation")

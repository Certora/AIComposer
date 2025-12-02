from typing_extensions import NotRequired, Annotated
from typing import Literal, Any, cast, Callable
from functools import partial

from pydantic import BaseModel, Field

from graphcore.tools.memory import MemoryBackend, memory_tool
from graphcore.graph import FlowInput, build_workflow, WithToolCallId, tool_output
from graphcore.tools.results import result_tool_generator
from graphcore.tools.vfs import VFSState

from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.runtime import get_runtime

from composer.templates.loader import load_jinja_template
from composer.core.state import AIComposerState
from composer.core.validation import reqs as req_key
from composer.core.context import AIComposerContext, compute_state_digest

class JudgeInput(FlowInput):
    vfs: dict[str, str]
    orig_reqs: list[str]

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

class JudgeState(MessagesState, VFSState):
    result: NotRequired[JudgeResult]
    orig_reqs: list[str]

def _gen_workflow(
    vfs_tools: list[BaseTool],
    mem: MemoryBackend,
    llm: BaseChatModel,
) -> StateGraph[JudgeState, None, JudgeInput, Any]:
    mem_tool = memory_tool(mem)
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
    return build_workflow(
        input_type=JudgeInput,
        output_key="result",
        context_schema=None,
        state_class=JudgeState,
        unbound_llm=llm,
        tools_list=[mem_tool, *vfs_tools, res],
        sys_prompt=load_jinja_template("req_role_prompt.j2"),
        initial_prompt=load_jinja_template("req_judge_prompt.j2")
    )[0]

classification_explanation = load_jinja_template("req_classifications.j2")

class RequirementEvaluationSchema(WithToolCallId):
    state: Annotated[AIComposerState, InjectedState]

RequirementEvaluationSchema.__doc__ = f"""
Query an oracle to determine if the generated implementation meets the requirements list
provided.

Each requirement is evaluated against the current implementation and assigned a classification:
{classification_explanation} 

If any requirements are classified as PARTIAL or VIOLATED, you must address this feedback.
    """

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

def get_judge_tool(
    reqs: list[str],
    mem: MemoryBackend,
    vfs_tools: list[BaseTool],
    unbound: BaseChatModel
) -> BaseTool:
    workflow = _gen_workflow(vfs_tools, mem, unbound)
    compiled_graph = workflow.compile()
    req_list = "\n".join([f"{i}. {r}" for (i, r) in enumerate(reqs, start = 1)])
    @tool(args_schema=RequirementEvaluationSchema)
    def requirements_evaluation(
        state: AIComposerState,
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command | str:
        r = compiled_graph.invoke(JudgeInput(
            input=[req_list],
            vfs=state["vfs"],
            orig_reqs=reqs
        ))
        skipped = state.get("skipped_reqs", set())
        res = cast(JudgeResult, r["result"])
        all_satisfied = True
        for j in res.judgement_result:
            if j.classification != "LIKELY" and j.classification != "SATISFIED":
                if j.requirement_number not in skipped:
                    all_satisfied = False
                    break
        formatted_res = _format_result(r["result"], skipped)
        if not all_satisfied:
            return formatted_res
        digest = compute_state_digest(
            c=get_runtime(AIComposerContext).context,
            state=state
        )
        return Command(update={
            "messages": [
                ToolMessage(content=formatted_res, tool_call_id=tool_call_id)
            ],
            "validation": {
                req_key: digest
            }
        })
    return requirements_evaluation


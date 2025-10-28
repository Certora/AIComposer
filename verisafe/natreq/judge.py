from typing_extensions import NotRequired, Annotated
from typing import Literal, Any

from pydantic import BaseModel, Field

from graphcore.tools.memory import MemoryBackend, memory_tool
from graphcore.graph import FlowInput, build_workflow
from graphcore.tools.results import result_tool_generator
from graphcore.tools.vfs import VFSState

from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models.chat_models import BaseChatModel

from verisafe.templates.loader import load_jinja_template
from verisafe.core.state import CryptoStateGen

class JudgeInput(FlowInput):
    vfs: dict[str, str]

class RequirementAnalysis(BaseModel):
    """
    The analysis result for one of the requirements.
    """
    classification: Literal["SATISFIED", "LIKELY", "PARTIAL", "VIOLATED"] = Field(description="The final classification on whether " \
    "the implementation satisfies the requirement.")

    requirement: str = Field(description="The original requirement text against which the implementation was judged.")

    commentary: str | None = Field(description="Any commentary or explanation for the classification of this requirement. In the case of" \
    "PARTIAL or VIOLATED classifications, do NOT suggest code changes: simply explain the deficiencies in the implementation.")

class JudgeResult(BaseModel):
    judgement_result: list[RequirementAnalysis] = Field(description="A list of analysis results, with one element per original requirement.")

class JudgeState(MessagesState, VFSState):
    result: NotRequired[JudgeResult]

def _gen_workflow(
    vfs_tools: list[BaseTool],
    mem: MemoryBackend,
    llm: BaseChatModel
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
"""
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

class RequirementEvaluationSchema(BaseModel):
    state: Annotated[CryptoStateGen, InjectedState]

RequirementEvaluationSchema.__doc__ = f"""
Query an oracle to determine if the generated implementation meets the requirements list
provided.

Each requirement is evaluated against the current implementation and assigned a classification:
{classification_explanation} 

If any requirements are classified as PARTIAL or VIOLATED, you must address this feedback.
    """


def _format_result(
    r: JudgeResult
) -> str:
    res_list = []
    for req_res in r.judgement_result:
        buff = "<result>"
        buff += f"<requirement>{req_res.requirement}</requirement>\n"
        buff += f"<classification>{req_res.classification}</classification>\n"
        if req_res.commentary:
            buff += f"<comments>{req_res.commentary}</comments>\n"
        buff += "</result>"
        res_list.append(buff)
    return "\n".join(res_list)

def get_judge_tool(
    reqs: list[str],
    mem: MemoryBackend,
    vfs_tools: list[BaseTool],
    unbound: BaseChatModel
) -> BaseTool:
    workflow = _gen_workflow(vfs_tools, mem, unbound)
    compiled_graph = workflow.compile()
    @tool(args_schema=RequirementEvaluationSchema)
    def requirements_evaluation(
        state: CryptoStateGen
    ) -> str:
        req_list = "\n".join([f"{i} {r}" for (i, r) in enumerate(reqs)])
        r = compiled_graph.invoke(JudgeInput(
            input=[req_list],
            vfs=state["vfs"]
        ))
        return _format_result(r["result"])
    return requirements_evaluation


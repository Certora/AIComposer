#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional, List, TypedDict, Annotated, Literal, TypeVar, Type, Tuple, NotRequired
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import InjectedToolCallId, BaseTool
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, MessagesState
from langgraph._internal._typing import StateLike
from langgraph.types import Command
from pydantic import BaseModel
from graphcore.summary import SummaryConfig

# TypeVars for generic typing
InputState = TypeVar('InputState', bound='FlowInput')
StateT = TypeVar('StateT', bound=MessagesState)
OutputT = TypeVar('OutputT', bound=StateLike)
ContextT = TypeVar("ContextT", bound=StateLike)

# Constants
INITIAL_NODE: Literal["initial"]
TOOLS_NODE: Literal["tools"]
TOOL_RESULT_NODE: Literal["tool_result"]

# Type aliases
BoundLLM = Runnable[LanguageModelInput, BaseMessage]

class WithToolCallId(BaseModel):
    """
    A common schema used for tools which need explicit access to the tool_id (usually
    for calling tool_output
    """
    tool_call_id: Annotated[str, InjectedToolCallId]

class FlowInput(TypedDict):
    """
    Upper bound on any type used as an input to a workflow.
    """
    front_matter: NotRequired[list[HumanMessage]]
    """
    Any contents to be placed *before* the initial prompt but
    *after* the system prompt.
    """
    
    input: list[str | dict]
    """
    Any workflow specific data to add *after* the initial prompt.
    """

def tool_output(tool_call_id: str, res: dict) -> Command:
    """
    Create a LangGraph Command for final tool outputs that update workflow state.

    Args:
        tool_call_id: The ID of the tool call being responded to
        res: Dictionary containing the final workflow results to merge into state

    Returns:
        Command that updates state with final results and a success message
    """
    ...

def tool_return(
    tool_call_id: str,
    content: str
) -> Command:
    """
    Create a LangGraph Command for tool responses that need to continue processing.

    Args:
        tool_call_id: The ID of the tool call being responded to
        content: The response content from the tool execution

    Returns:
        Command that updates messages and continues workflow
    """
    ...

def build_workflow(
    state_class: Type[StateT],
    input_type: Type[InputState],
    tools_list: List[BaseTool],
    sys_prompt: str,
    initial_prompt: str,
    output_key: str,
    unbound_llm: BaseChatModel,
    output_schema: Optional[Type[OutputT]] = None,
    context_schema: Optional[Type[ContextT]] = None,
    summary_config: SummaryConfig[StateT] | None = None
) -> Tuple[StateGraph[StateT, ContextT, InputState, OutputT], BoundLLM]:
    """
    Build a standard workflow with initial node -> tools -> tool_result pattern.

    Args:
        state_class: The type of the "main" state, bounded by `MessagesState`
        input_type: The type of the "input" state, bounded by `FlowInput`
        tools_list: A list of tools that the LLM can call during iteration
        sys_prompt: The system prompt sent to start the conversation
        initial_prompt: The static prompt sent describing the task and how to use the tools
        output_key: The designated "output" tool should set this key to be non-None in the current state
        unbound_llm: The llm to use for the computation and looping
        output_schema: (Optional) describes the output format of the computation
        context_schema: (Optional) the type of contexts passed through the computation

    Returns:
        The state graph compiled to execute the workflow above, and the llm with the tools bound
    """
    ...
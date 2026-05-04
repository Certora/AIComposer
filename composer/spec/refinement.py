from dataclasses import dataclass
import enum

from typing import Callable, Literal, Never, cast

from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.types import Command

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver


from graphcore.utils import acached_invoke

from langchain_core.messages import AnyMessage, BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from composer.io.conversation import ConversationClient, AIYapping, ToolComplete, ToolStart, ThinkingStart
from composer.io.protocol import IOHandler
from composer.io.event_handler import NullEventHandler
from composer.io.context import with_handler
from composer.ui.tool_display import ToolDisplayConfig

from composer.spec.util import uniq_thread_id

@dataclass
class EndConversation:
    pass

@dataclass
class HumanPrompt:
    ai_message: str | None

class ConversationStateEnum(enum.Enum):
    CHAT = 1
    INIT = 2


class ConversationState[T](MessagesState):
    state: ConversationStateEnum
    extra_data: T

async def refinement_loop[T](
    llm: BaseChatModel,
    client: ConversationClient,
    init_data: T,
    init_messages: list[AnyMessage],
    tools: list[BaseTool],
    tool_display: ToolDisplayConfig | None = None
) -> ConversationState[T]:
    graph = StateGraph(
        state_schema=ConversationState[T],
        context_schema=None,
        input_schema=None,
        output_schema=None
    )
    bound_llm = llm.bind_tools(tools)
    init_state : ConversationState[T] = {
        "messages": init_messages,
        "extra_data": init_data,
        "state": ConversationStateEnum.INIT
    }

    async def llm_echo(state: ConversationState[T]) -> dict[str, list[BaseMessage]]:
        client.progress_update(ThinkingStart())
        res = await acached_invoke(bound_llm, state["messages"])
        assert isinstance(res, AIMessage)
        if len(res.tool_calls):
            if len(res.text) > 0:
                client.progress_update(AIYapping(res.text))
            for t in res.tool_calls:
                tid = t["id"]
                assert tid is not None
                client.progress_update(ToolStart(
                    tid=tid,
                    tool_name=t["name"] # todo: pretty print
                ))
        return {
            "messages": [res]
        }

    tool_node = ToolNode(tools, handle_tool_errors=(ToolInvocationError,))

    def chat_node(
        state: ConversationState[T]
    ) -> dict[str, list[BaseMessage] | ConversationStateEnum]:
        payload_text = None
        if state["state"] != ConversationStateEnum.INIT:
            msg = state["messages"][-1]
            if isinstance(msg, AIMessage):
                payload_text = msg.text

        res = interrupt(HumanPrompt(
            payload_text
        ))
        assert isinstance(res, str)
        return {
            "messages": [HumanMessage(res)],
            "state": ConversationStateEnum.CHAT
        }
    
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    
    graph.add_node("llm_echo", llm_echo)
    graph.add_edge("chat_node", "llm_echo")

    graph.add_node("tools", tool_node)

    def conditional_decider(
        state: ConversationState[T]
    ) -> Literal["tools", "chat_node"]:
        last_msg = state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        if len(last_msg.tool_calls) == 0:
            return "chat_node"
        else:
            return "tools"

    graph.add_conditional_edges("llm_echo", conditional_decider)

    runner = graph.compile(
        checkpointer=InMemorySaver()
    )

    tid = uniq_thread_id("refinement_conversation")

    class NullHandler(IOHandler[Never]):
        async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str):
            pass

        async def log_end(self, path: list[str]):
            pass

        async def log_state_update(self, path: list[str], st: dict):
            pass

        async def human_interaction(self, ty: Never, debug_thunk: Callable[[], None]) -> str:
            raise RuntimeError("This should never be called")
        
        async def log_start(self, *, path: list[str], description: str, tool_id: str | None):
            pass

    graph_input : ConversationState[T] | Command | None = init_state
    async with with_handler(
        NullHandler(), NullEventHandler()
    ):
        while graph_input:
            human_question : str | None | Literal[False] = False
            to_run = graph_input
            graph_input = None
            async for (ev, payload) in runner.astream(
                to_run, config = {
                    "configurable": {
                        "thread_id": tid
                    }
                },
                stream_mode=["updates"]
            ):
                assert ev == "updates" and isinstance(payload, dict)
                if "__interrupt__" in payload:
                    interrupt_data = payload["__interrupt__"][0].value
                    assert isinstance(interrupt_data, EndConversation) or isinstance(interrupt_data, HumanPrompt)
                    if not isinstance(interrupt_data, EndConversation):
                        human_question = interrupt_data.ai_message
                    break
                if "tools" in payload and "messages" in payload["tools"]:
                    for m in payload["tools"]["messages"]:
                        if isinstance(m, ToolMessage):
                            client.progress_update(
                                ToolComplete(
                                    tid=m.tool_call_id
                                )
                            )
            if human_question is not False:
                res = await client.human_turn(ai_response=human_question)
                graph_input = Command(resume=res)

    to_res = await runner.aget_state({
        "configurable": {
            "thread_id": tid
        }
    })
    return cast(ConversationState[T], to_res.values)
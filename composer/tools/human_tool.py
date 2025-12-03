from typing import TypeVar, Callable, Literal, Annotated, get_args, get_origin, cast, Any

from pydantic import create_model, Field

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from composer.core.state import AIComposerState
from composer.human.handlers import HumanInteractionType

T = TypeVar("T", bound=HumanInteractionType)

_injected_state_name = "composer_injected_state"

def human_interaction_tool(
    t: type[T],
    name: str,
    state_updater: Callable[[AIComposerState, T, str], dict] = lambda x, y, z: {}
) -> BaseTool:
    fields = {}
    disc : str | None = None
    for (k, v) in t.__annotations__.items():
        if get_origin(v) is Literal:
            if k != "type":
                raise RuntimeError(f"Illegal type annotation: {v} on {k}")
            disc = get_args(v)[0]
            continue
        elif get_origin(v) is Annotated:
            a = get_args(v)
            if len(a) != 2 or not isinstance(a[1], str):
                raise RuntimeError(f"Illegal type annotation: {v} for {k}")
            fields[k] = (a[0], Field(description=a[1]))
        else:
            raise RuntimeError(f"Illegal type annotation: {v} for {k}")
    fields[_injected_state_name] = (Annotated[AIComposerState, InjectedState], Field())
    fields["tool_call_id"] = (Annotated[str, InjectedToolCallId], Field())

    model = create_model(
        t.__name__,
        __doc__ = t.__doc__,
        **cast(dict[str, Any], fields)
    )
    @tool(args_schema=model)
    def interaction_tool(
        **kwargs
    ) -> Command:
        dict_args = {
            k: v for (k, v) in kwargs.items() if k != "tool_call_id" and k != _injected_state_name
        }
        dict_args["type"] = disc
        payload : T = t(**dict_args) #type: ignore
        response = interrupt(payload)
        state_update = state_updater(kwargs[_injected_state_name], payload, response)
        response_update = {
            "messages": [
                ToolMessage(content=response, tool_call_id=kwargs["tool_call_id"])
            ]
        }
        response_update.update(state_update)
        return Command(update=response_update)
    interaction_tool.name = name
    return interaction_tool

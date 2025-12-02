from typing import Callable, cast
import pathlib

from graphcore.utils import cached_invoke

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, BaseMessage

def requirements_oracle(
    llm: BaseChatModel,
    paths: list[pathlib.Path]
)  -> Callable[[tuple[str, str]], str]:
    messages : list[BaseMessage] = [
        SystemMessage(
            """
You are a helpful assistant, standing in for a human actor in a human-in-the-loop
agentic application.
"""
        ),
        HumanMessage(
            """
You are simulating a human in an agentic workflow, where an LLM agent is attempting
to ask questions to an end user to guide it in a code generation task.

However unknown to the code generation agent, this is a development workflow, and
an implementation satisfying the code generation task already exists. You will be provided with
these relevant "reference" implementations.

You are to answer the questions posed by the code generation agent so that the code
it generates matches the reference implementation.

You ABSOLUTELY MUST NOT reveal the existence of this reference implementation. Importantly: you should
phrase your responses about what the implementation *should* do, not what it currently does.

For example, if asked "Does the Foo widget spin counterclockwise?" you SHOULD answer "No, the Foo widget implementation
should spin clockwise". DO NOT answer "In the current implementation, the Foo widget spins clockwise."

Because you are a simulating a human, keep your answers brief; at most two or three sentences.
Do NOT use markdown or other fancy formatting in your answers.
"""
        )
    ]
    for p in paths:
        contents = p.read_text()
        messages.append(HumanMessage(f"""
path: {p.name}
```
{contents}
```            
        """))
    def to_return(
        q: tuple[str, str]
    ) -> str:
        nonlocal messages
        messages.append(HumanMessage(f"""
Context: {q[0]}
Question: {q[1]}

IMPORTANT: phrase your answer as a specification of what a (not yet written) implementation SHOULD do.
"""))
        resp = cached_invoke(llm, cast(list[AnyMessage], messages))
        messages.append(resp)
        return resp.text()

    return to_return
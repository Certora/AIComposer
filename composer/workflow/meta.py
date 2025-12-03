# for meta iteration

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage, HumanMessage

from pydantic import BaseModel, Field

from composer.core.state import AIComposerState
from composer.templates.loader import load_jinja_template

class ResumeCommentary(BaseModel):
    """
    The structured output to use for generating the resume commentary.
    """

    commentary: str = Field(description="Your commentary describing your work, what you did, and " \
    "what should be kept in mind if this work needs to be resume.")

    interface_path: str = Field(description="The path of the interface file on the VFS")

def create_resume_commentary(state: AIComposerState, llm: BaseChatModel) -> ResumeCommentary:
    bound = llm.with_structured_output(ResumeCommentary)
    messages = state["messages"].copy()

    last = messages[-1]
    assert isinstance(last, ToolMessage)

    messages.append(HumanMessage(load_jinja_template("final_commentary_prompt.j2")))

    res = bound.invoke(messages)
    assert isinstance(res, ResumeCommentary)
    return res

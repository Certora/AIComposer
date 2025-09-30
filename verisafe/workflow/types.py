from typing import TypedDict

from graphcore.graph import FlowInput

class PromptParams(TypedDict):
    is_resume: bool

class Input(FlowInput):
    """
    Input state, with initial virtual fs definitions
    """
    vfs: dict[str, str]

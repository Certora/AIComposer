from typing import TypedDict

from graphcore.graph import FlowInput

class PromptParams(TypedDict):
    is_resume: bool
    has_project_root: bool
    has_foundry_tests: bool

class Input(FlowInput):
    """
    Input state, with initial virtual fs definitions
    """
    vfs: dict[str, str]

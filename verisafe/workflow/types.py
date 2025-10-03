from graphcore.graph import FlowInput
from typing import Dict

class Input(FlowInput):
    """
    Input state, with initial virtual fs definitions
    """
    vfs: dict[str, str]

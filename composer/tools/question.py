from composer.human.types import QuestionType
from graphcore.tools.human import human_interaction_tool
from composer.core.state import AIComposerState
from composer.ui.tool_display import tool_display

human_in_the_loop = tool_display(
    lambda p: (
        f"Asking for input: {p['question']}"
        if p.get("question") else "Asking for input"
    ),
    None,
)(human_interaction_tool(QuestionType, AIComposerState, "human_in_the_loop"))

from composer.human.types import QuestionType
from graphcore.tools.human import human_interaction_tool
from composer.core.state import AIComposerState
from composer.ui.tool_display import tool_display

human_in_the_loop = (human_interaction_tool(QuestionType, AIComposerState, "human_in_the_loop"))

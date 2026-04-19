from composer.core.state import AIComposerState
from composer.human.types import RequirementRelaxationType
from graphcore.tools.human import human_interaction_tool
from composer.ui.tool_display import tool_display

def _maybe_relax(s: AIComposerState, q: RequirementRelaxationType, resp: str) -> dict:
    if resp.startswith("ACCEPTED"):
        return {
            "skipped_reqs": {q["req_number"]}
        }
    else:
        return {}

requirements_relaxation = tool_display(
    lambda p: (
        f"Requesting requirement relaxation #{p.get('req_number', '?')}: {p.get('req_text', '')}"
        if p.get("req_text")
        else "Requesting requirement relaxation"
    ),
    None,
)(human_interaction_tool(
    RequirementRelaxationType,
    AIComposerState,
    "requirement_relaxation_request",
    _maybe_relax
))

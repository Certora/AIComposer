from composer.core.state import CryptoStateGen
from composer.human.types import RequirementRelaxationType
from composer.tools.human_tool import human_interaction_tool

def _maybe_relax(s: CryptoStateGen, q: RequirementRelaxationType, resp: str) -> dict:
    if resp.startswith("ACCEPTED"):
        return {
            "skipped_reqs": {q["req_number"]}
        }
    else:
        return {}

requirements_relaxation = human_interaction_tool(
    RequirementRelaxationType,
    "requirement_relaxation_request",
    _maybe_relax
)

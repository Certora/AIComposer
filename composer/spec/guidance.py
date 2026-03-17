from typing import override
from graphcore.tools.schemas import WithImplementation

from composer.templates.loader import load_jinja_template

class ResolutionGuidance(WithImplementation[str]):
    """
    Retrieve guidance on resolution.
    """

    @override
    def run(self) -> str:
        return load_jinja_template("resolution_guidance.j2")

class ERC20TokenGuidance(WithImplementation[str]):
    """
    Invoke this tool to receive guidance on how ERC20 is usually modelled using the prover.
    """
    @override
    def run(self) -> str:
        return load_jinja_template("erc20_advice.j2")


class UnresolvedCallGuidance(WithImplementation[str]):
    """
Invoke this tool to receive guidance on how to deal with verification failures due to havocs caused by
unresolved calls.
    """
    @override
    def run(self) -> str:
        return load_jinja_template("unresolved_call_guidance.j2")

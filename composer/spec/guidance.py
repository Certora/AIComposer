from typing import override
from graphcore.tools.schemas import WithImplementation

from composer.templates.loader import load_jinja_template
from composer.ui.tool_display import tool_display

@tool_display("Getting ERC20 guidance", None)
class ERC20TokenGuidance(WithImplementation[str]):
    """
    Invoke this tool to receive guidance on how ERC20 is usually modelled using the prover.
    """
    @override
    def run(self) -> str:
        return load_jinja_template("erc20_advice.j2")

@tool_display("Getting unresolved call guidance", None)
class UnresolvedCallGuidance(WithImplementation[str]):
    """
Invoke this tool to receive guidance on how to deal with verification failures due to havocs caused by
unresolved calls.
    """
    @override
    def run(self) -> str:
        return load_jinja_template("unresolved_call_guidance.j2")

@tool_display("Getting call resolution guidance", None)
class ResolutionGuidance(WithImplementation[str]):
    """
Invoke this tool to receive guidance on how the Certora Prover resolves ambiguous/polymorphic calls (e.g.,
calls through an interface type), and what it takes to *soundly* conclude that a callee is definitely one of
the contracts in the prover's inputs. Consult this before relying on a DISPATCHER summary, which unsoundly
assumes the callee is always one of the known contracts.
    """
    @override
    def run(self) -> str:
        return load_jinja_template("resolution_guidance.j2")

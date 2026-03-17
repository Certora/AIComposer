
from dataclasses import dataclass, field
from typing import Callable, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool



from composer.spec.context import (
    SourceBuilder, CVLBuilder, CVLOnlyBuilder,
    SourceCode, SystemDoc,
)

# ---------------------------------------------------------------------------
# GenerationEnv — unified configuration for CVL generation
# ---------------------------------------------------------------------------

class CVLResource(BaseModel):
    import_path: str = Field(description="the path to the resource (relative to `certora/`)")
    required: bool = Field(description="whether this resource *must* be used in the verification process")
    description: str = Field(description="A description of this resource")
    sort: Literal["import"]

class Capabilities(TypedDict):
    has_source: bool
    has_stub_tools: bool
    has_prover: bool

class ContractContext(TypedDict):
    contract_name: str | None
    relative_path: str | None

class GenerationTemplateParams(ContractContext, Capabilities):
    pass

@dataclass
class SourceInput(SourceCode):
    prover_tool: BaseTool
    source_tools: SourceBuilder

    def params(self) -> GenerationTemplateParams:
        return {
            "has_prover": True,
            "has_source": True,
            "contract_name": self.contract_name,
            "relative_path": self.relative_path,
            "has_stub_tools": False
        }

@dataclass
class NatspecInput(SystemDoc):
    stub_provider: Callable[[], str]

    def params(self) -> GenerationTemplateParams:
        return {
            "has_prover": False,
            "contract_name": None,
            "relative_path": None,
            "has_source": False,
            "has_stub_tools": True
        }

type GenerationInput = NatspecInput | SourceInput

@dataclass
class StandardResult:
    result_template: str = "deliver_spec_fragment.j2"

@dataclass
class CustomOutput:
    publish_tools: list[BaseTool]
    result_template: str

@dataclass
class GenerationEnv:
    """Environment configuration for CVL generation.

    Bundles input, role-based builders, and optional capabilities.
    Each capability adds tools and template conditionals.

    Builder roles:
    - cvl_authorship: main CVL generation agent and feedback judge
    - cvl_research: CVL research sub-agents (manual/KB search only)
    - source_tools: code exploration sub-agent (None if no source code)
    """
    input: GenerationInput
    cvl_authorship: CVLBuilder | CVLOnlyBuilder
    cvl_research: CVLOnlyBuilder
    output: CustomOutput | StandardResult

    resources: list[CVLResource] = field(default_factory=list)
    extra_tools: list[BaseTool] = field(default_factory=list)

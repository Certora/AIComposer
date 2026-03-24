
from dataclasses import dataclass, field
from typing import Callable, Literal, TYPE_CHECKING
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

if TYPE_CHECKING:
    class TypedTemplate[T]:
        def __init__(self, name: str):
            self._wrapped = name

        def __str__(self) -> str:
            return self._wrapped
else:
    class TypedTemplate(str):
        def __new__(cls, s: str) -> str:
            return s
        
        def __class_getitem__(cls, _):
            return cls

@dataclass
class TemplateInstantiation:
    template: TypedTemplate
    args: dict

    @staticmethod
    def create[T](
        templ: TypedTemplate[T],
        args: T
    ) -> "TemplateInstantiation":
        assert isinstance(args, dict)
        return TemplateInstantiation(templ, args)

@dataclass
class GenerationPrompt:
    cvl_prompt: TemplateInstantiation
    feedback_prompt: TemplateInstantiation

    cvl_prompt_extras: list[str | dict] = field(default_factory=list)
    feedback_prompt_extras: Callable[[], list[str | dict]] = field(default=lambda: [])

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
    prompt: GenerationPrompt
    cvl_authorship: CVLBuilder | CVLOnlyBuilder
    cvl_research: CVLOnlyBuilder
    validation_tools: list[tuple[str, BaseTool]] = field(default_factory=list)
    output_tools: list[BaseTool] | None = None

    resources: list[CVLResource] = field(default_factory=list)
    extra_tools: list[BaseTool] = field(default_factory=list)

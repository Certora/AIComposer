
from dataclasses import dataclass
from typing import Literal, Mapping, Any, Protocol

from pydantic import BaseModel, Field





# ---------------------------------------------------------------------------
# GenerationEnv — unified configuration for CVL generation
# ---------------------------------------------------------------------------

class CVLResource(BaseModel):
    import_path: str = Field(description="the path to the resource (relative to `certora/`)")
    required: bool = Field(description="whether this resource *must* be used in the verification process")
    description: str = Field(description="A description of this resource")
    sort: Literal["import"]

class TypedTemplate[T: Mapping[str, Any]]:
    def __init__(self, name: str):
        self._wrapped = name

    def __str__(self) -> str:
        return self._wrapped

    def bind(self, params: T) -> "TemplateInstantiation":
        return TemplateInstantiation.create(self, params)

class TemplateRenderer[T](Protocol):
    def __call__(self, template: str, **kwargs) -> T:
        ...

@dataclass
class TemplateInstantiation:
    template: TypedTemplate
    args: dict

    @staticmethod
    def create[T: Mapping[str, Any]](
        templ: TypedTemplate[T],
        args: T
    ) -> "TemplateInstantiation":
        assert isinstance(args, dict)
        return TemplateInstantiation(templ, args)

    def render_to[T](
        self,
        cb: TemplateRenderer[T]
    ) -> T:
        return cb(
            str(self.template),
            **self.args
        )

    def depends[X: Mapping[str, Any]](self, other: type[X]) -> "InjectedTemplate[X]":
        return InjectedTemplate(self)

@dataclass
class InjectedTemplate[X: Mapping[str, Any]]:
    wrapped: TemplateInstantiation

    def inject(self, injected: X) -> TemplateInstantiation:
        return TemplateInstantiation(
            TypedTemplate(str(self.wrapped.template)),
            {
                **self.wrapped.args,
                **injected
            }
        )

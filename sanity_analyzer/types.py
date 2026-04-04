from typing import Protocol

from composer.input.types import ModelOptions, WorkflowOptions


class SanityAnalysisArgs(ModelOptions, WorkflowOptions, Protocol):
    @property
    def unsat_core_txt_path(self) -> str:
        ...

    @property
    def rule(self) -> str | None:
        ...

    @property
    def method(self) -> str | None:
        ...

    @property
    def quiet(self) -> bool:
        ...

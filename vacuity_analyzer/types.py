from typing import Protocol

from composer.input.types import ModelOptions, WorkflowOptions


class VacuityAnalysisArgs(ModelOptions, WorkflowOptions, Protocol):
    @property
    def vacuity_txt_path(self) -> str:
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

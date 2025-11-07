from typing import Callable, TypeVar, overload
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langgraph.types import Command


ST = TypeVar("ST")
R = TypeVar("R")
M = TypeVar("M", bound=BaseModel)

ValidationResult = Command | str | None
@overload
def result_tool_generator(
    outkey: str,
    result_schema: type[M],
    doc: str,
    validator: tuple[type[ST], Callable[[ST, M, str], ValidationResult]]
) -> BaseTool:
    ...

@overload
def result_tool_generator(
    outkey: str,
    result_schema: type[M],
    doc: str,
    validator: Callable[[M, str], ValidationResult]
) -> BaseTool:
    ...


@overload
def result_tool_generator(
    outkey: str,
    result_schema: tuple[type[R], str],
    doc: str,
    validator: Callable[[R, str], ValidationResult]
) -> BaseTool:
    ...

@overload
def result_tool_generator(
    outkey: str,
    result_schema: tuple[type[R], str],
    doc: str,
    validator: tuple[type[ST], Callable[[ST, R, str], ValidationResult]]
) -> BaseTool:
    ...

@overload
def result_tool_generator(
    outkey: str,
    result_schema: type[BaseModel] | tuple[type, str],
    doc: str,
) -> BaseTool:
    ...


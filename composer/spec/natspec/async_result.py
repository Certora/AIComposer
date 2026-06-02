from pydantic import Field
from abc import ABC, abstractmethod
from typing import ClassVar

from langgraph.types import Command

from graphcore.graph import tool_state_update
from graphcore.tools.schemas import WithAsyncImplementation, WithInjectedId

class AsyncResultTool[T](WithAsyncImplementation[str | Command], WithInjectedId, ABC):
    value: T = Field(description="The result of your task")

    RESULT_KEY: ClassVar[str] = "result"

    @abstractmethod
    async def validate(self, res: T) -> str | None:
        ...

    async def run(self) -> str | Command:
        if (err_msg := await self.validate(self.value)) is not None:
            return err_msg
        result_key = type(self).RESULT_KEY
        upd = {
            result_key: self.value
        }
        return tool_state_update(self.tool_call_id, "Accepted", **upd)

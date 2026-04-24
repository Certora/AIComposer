from typing import Literal, Sequence
from dataclasses import dataclass

from graphcore.graph import Builder

from langchain_core.language_models.chat_models import BaseChatModel as LLM
from langchain_core.tools import BaseTool


@dataclass
class PureServiceHost:
    llm: LLM
    builder: Builder[None, None, None]
    cvl_tools: tuple[BaseTool, ...]
    sort: Literal["greenfield", "existing", "update"]

    def bind_source_tools(self, tools: Sequence[BaseTool]) -> "ServiceHost":
        return ServiceHost(
            llm=self.llm, builder=self.builder, cvl_tools=self.cvl_tools,
            source_tools=tuple(tools), sort=self.sort
        )

@dataclass
class ServiceHost(PureServiceHost):
    source_tools: tuple[BaseTool, ...]

    @property
    def all_tools(self) -> tuple[BaseTool, ...]:
        return self.source_tools + self.cvl_tools


from typing import Literal, Sequence, Protocol, Callable
from dataclasses import dataclass

from graphcore.graph import Builder

from langchain_core.language_models.chat_models import BaseChatModel as LLM
from langchain_core.tools import BaseTool

class ServiceHostProtocol(Protocol):
    @property
    def llm(self) -> LLM:
        ...

    @property
    def builder(self) -> Builder[None, None, None]:
        ...

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

    @classmethod
    def from_protocol[T: ServiceHostProtocol](
        cls, p: T, src: Callable[[T], tuple[BaseTool, ...]], cvl: Callable[[T], tuple[BaseTool, ...]], sort: Literal["greenfield", "existing", "update"] = "update"
    ) -> "ServiceHost":
        return ServiceHost(
            llm=p.llm,
            builder=p.builder,
            sort=sort,
            cvl_tools=cvl(p),
            source_tools=src(p)
        )

    @property
    def all_tools(self) -> tuple[BaseTool, ...]:
        return self.source_tools + self.cvl_tools


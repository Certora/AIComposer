from typing import Protocol
from langchain_core.tools import BaseTool
from graphcore.graph import Builder

class BasicAgentTools(Protocol):
    @property
    def builder(self) -> Builder[None, None, None]:
        ...

    @property
    def has_source(self) -> bool:
        ...

class BaseRAGTools(Protocol):
    @property
    def base_rag_tools(self) -> tuple[BaseTool, ...]:
        ...

class BaseSourceTools(Protocol):
    @property
    def base_source_tools(self) -> tuple[BaseTool, ...]:
        ...

class RAGTools(Protocol):
    @property
    def rag_tools(self) -> tuple[BaseTool, ...]:
        ...

class SourceTools(Protocol):
    @property
    def source_tools(self) -> tuple[BaseTool, ...]:
        ...

class ToolEnvironment(BasicAgentTools, RAGTools, Protocol):
    @property
    def cvl_authorship_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def feedback_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def system_analysis_tools(self) -> tuple[BaseTool, ...]:
        ...

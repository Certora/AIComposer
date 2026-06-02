from typing import Protocol
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from graphcore.graph import Builder
from composer.spec.service_host import Sort

class BasicAgentTools(Protocol):
    @property
    def llm(self) -> BaseChatModel:
        ...

    @property
    def builder(self) -> Builder[None, None, None]:
        ...

    @property
    def sort(self) -> Sort:
        """The workflow's relationship with the underlying source tree:
        ``greenfield`` (no source yet), ``update`` (extending an existing
        codebase), or ``existing`` (verifying as-is). Replaces the old
        ``has_source: bool``: ``has_source`` == ``sort != "greenfield"``.
        """
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

"""Structural protocols for sub-agent tool implementations.

``BasicAgentTools`` is the universal minimum every agent runner needs:
``llm`` + ``builder``. It's intentionally narrower than ``ServiceHost``
(which adds ``sort`` and full tool tuples) so the internal researcher /
explorer envs (``CVLResearchEnv``, ``CodeExplorerEnv``) can satisfy it
without having to fabricate a workflow ``sort`` they don't actually use.

The per-agent envs that used to live here (``ToolEnvironment``,
``RAGTools``, ``SourceTools``, plus their per-role children) are gone —
agents now take a concrete ``ServiceHost`` directly.
"""

from typing import Protocol
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from graphcore.graph import Builder


class BasicAgentTools(Protocol):
    @property
    def llm(self) -> BaseChatModel:
        ...

    @property
    def builder(self) -> Builder[None, None, None]:
        ...


class BaseRAGTools(Protocol):
    """The unwrapped RAG-tool surface — what ``cvl_researcher``'s own
    implementation depends on (so it doesn't recursively include itself)."""

    @property
    def base_rag_tools(self) -> tuple[BaseTool, ...]:
        ...


class BaseSourceTools(Protocol):
    """The unwrapped source-tool surface — what ``code_explorer``'s own
    implementation depends on (so it doesn't recursively include itself)."""

    @property
    def base_source_tools(self) -> tuple[BaseTool, ...]:
        ...

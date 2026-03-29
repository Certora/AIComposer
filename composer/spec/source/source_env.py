from typing import Protocol, Unpack
from dataclasses import dataclass
from functools import cached_property

from langchain_core.tools import BaseTool

from composer.spec.tool_env import ToolEnvironment, SourceTools, BaseSourceTools, BasicAgentTools
from composer.spec.services import build_rag_tool_env, _BaseTools, RAGInputs
from graphcore.tools.vfs import fs_tools
from composer.spec.code_explorer import code_explorer_tool


@dataclass(frozen=True)
class _BaseSourceTools():
    base_source_tools: tuple[BaseTool, ...]

@dataclass(frozen=True)
class _SourceTools:
    source_tools: tuple[BaseTool,...]

def build_basic_source_tools(
    root: str,
    forbidden_read: str
) -> BaseSourceTools:
    return _BaseSourceTools(
        tuple(fs_tools(fs_layer=root, forbidden_read=forbidden_read, cache_listing=False))
    )


def build_source_tools(
    s: BaseSourceTools,
    llm: BasicAgentTools
) -> SourceTools:
    @dataclass(frozen=True)
    class _ExplorerEnv(_BaseTools, _BaseSourceTools):
        pass

    explorer_tool = code_explorer_tool(_ExplorerEnv(
        builder=llm.builder,
        has_source=llm.has_source,
        base_source_tools=s.base_source_tools
    ))

    return _SourceTools(
        source_tools=s.base_source_tools + (explorer_tool,)
    )

class SourceEnvironment(ToolEnvironment, SourceTools, Protocol):
    pass


class SourceParams(RAGInputs):
    root: str
    forbidden_read: str

def build_source_env(
    **params: Unpack[SourceParams]
) -> SourceEnvironment:
    rag_env = build_rag_tool_env(**params)

    basic_source = build_basic_source_tools(
        root=params["root"],
        forbidden_read=params["forbidden_read"]
    )

    full_source = build_source_tools(basic_source, rag_env)

    @dataclass(frozen=True)
    class ToRet(_SourceTools, _BaseTools):
        rag_tools: tuple[BaseTool, ...]
        
        @property
        def cvl_authorship_tools(self) -> tuple[BaseTool, ...]:
            return self.source_tools + self.rag_tools
    
        @property
        def feedback_tools(self) -> tuple[BaseTool, ...]:
            return self.cvl_authorship_tools
        
        @property
        def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
            return self.source_tools
        
        @property
        def system_analysis_tools(self) -> tuple[BaseTool, ...]:
            return self.source_tools

    return ToRet(
        builder=rag_env.builder,
        has_source=True,
        rag_tools=rag_env.rag_tools,
        source_tools=full_source.source_tools
    )
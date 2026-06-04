"""ServiceHost — the per-agent environment passed to the shared
analysis machinery.

Replaces the family of per-agent ``Env`` protocols (``AnalysisEnv``,
``BugEnvironment``, ``FeedbackEnv``, ...) that each layered a different
tool-shape on top of ``BasicAgentTools``. Concrete consumers used to
satisfy several of these protocols at once; the wrapping function
picked whichever subset it wanted. That worked but it spread the
"what mode are we in?" axis across every protocol — every callsite
had to know to ask ``has_source``.

``ServiceHost`` collapses that into a single concrete dataclass: a
fixed set of tool slots plus a tri-state ``sort`` describing the
relationship between the workflow and the underlying source tree:

  - ``greenfield`` — no Solidity exists yet; everything is stubs.
  - ``update``     — pre-existing codebase being extended with new
                      contracts/edits.
  - ``existing``   — pre-existing codebase being verified as-is.

The natspec pipeline picks ``greenfield`` / ``update`` depending on
whether the design doc describes new work or an extension. The
autoprover pipeline always runs with ``sort="existing"``.
"""

from typing import Literal, Sequence, Protocol, Callable
from dataclasses import dataclass

from graphcore.graph import Builder

from langchain_core.language_models.chat_models import BaseChatModel as LLM
from langchain_core.tools import BaseTool


Sort = Literal["greenfield", "existing", "update"]


@dataclass
class PureServiceHost:
    """``ServiceHost`` without source tools — the pre-stub natspec phase
    uses this directly. Call :meth:`bind_source_tools` once a usable
    source-layer materializer exists to get a full ``ServiceHost``."""

    llm: LLM
    builder: Builder[None, None, None]
    rag_tools: tuple[BaseTool, ...]
    sort: Sort

    def bind_source_tools(self, tools: Sequence[BaseTool]) -> "ServiceHost":
        return ServiceHost(
            llm=self.llm,
            builder=self.builder,
            rag_tools=self.rag_tools,
            source_tools=tuple(tools),
            sort=self.sort,
        )


@dataclass
class ServiceHost(PureServiceHost):
    """Concrete per-agent environment carrying both tool tuples and the
    workflow ``sort``. The single env type for all per-agent callsites —
    replaces the family of per-role ``Env`` protocols.

    Four tool entry points:

    - :attr:`rag_tools` — CVL manual / cvl_researcher / kb tools.
    - :attr:`source_tools` — raw fs/explorer over the layered VFS.
    - :attr:`all_tools` — ``source_tools + rag_tools`` (authoring / feedback).
    - :attr:`analysis_tools` — :attr:`source_tools` when ``sort != "greenfield"``
      else empty; the right surface for component / bug analysis, which run
      before any meaningful source backends are layered into the VFS.
    """

    source_tools: tuple[BaseTool, ...]

    @property
    def all_tools(self) -> tuple[BaseTool, ...]:
        return self.source_tools + self.rag_tools

    @property
    def analysis_tools(self) -> tuple[BaseTool, ...]:
        """``source_tools`` in ``update`` / ``existing`` mode; empty in
        ``greenfield``. Component analysis and bug analysis run before any
        domain-specific backends are layered into the VFS, so in greenfield
        the source-tool tuple would be non-empty but point at an empty
        tree — exposing it just tempts the agent to fish around. Use
        this property to encapsulate the gate so consumers don't need to
        check ``sort`` themselves."""
        return self.source_tools if self.sort != "greenfield" else ()

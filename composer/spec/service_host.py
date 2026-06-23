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

from typing import Literal, Sequence
from dataclasses import dataclass

from langgraph.types import Checkpointer
from graphcore.graph import Builder

from langchain_core.language_models.chat_models import BaseChatModel as LLM
from langchain_core.tools import BaseTool

from composer.templates.loader import load_jinja_template
from composer.workflow.services import CacheLevel, LLMFactory


Sort = Literal["greenfield", "existing", "update"]


@dataclass(frozen=True)
class ModelProvider:
    """Mints tiered LLMs / Builders on demand, deferring the model / cache /
    thinking choice to the call site.

    ``heavy_model`` and ``lite_model`` are model names fed to the same
    :class:`~composer.workflow.services.LLMFactory`. A workflow that does not
    support model-swapping (natspec) opts out simply by constructing this with
    ``heavy_model == lite_model``: ``llm_lite`` then returns the same model as
    ``llm_heavy``. ``cache_level`` / ``disable_thinking`` are honoured either
    way, so a collapsed provider still gives access to those knobs."""

    factory: LLMFactory
    heavy_model: str
    lite_model: str
    checkpointer: Checkpointer

    def llm_heavy(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> LLM:
        return self.factory(self.heavy_model, cache_level=cache_level, disable_thinking=disable_thinking)

    def llm_lite(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> LLM:
        return self.factory(self.lite_model, cache_level=cache_level, disable_thinking=disable_thinking)

    def builder_heavy(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> Builder[None, None, None]:
        return self._builder(self.llm_heavy(cache_level=cache_level, disable_thinking=disable_thinking))

    def builder_lite(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> Builder[None, None, None]:
        return self._builder(self.llm_lite(cache_level=cache_level, disable_thinking=disable_thinking))

    def _builder(self, llm: LLM) -> Builder[None, None, None]:
        return Builder[None, None, None]().with_llm(
            llm
        ).with_loader(load_jinja_template).with_checkpointer(self.checkpointer)


@dataclass
class PureServiceHost:
    """``ServiceHost`` without source tools — the pre-stub natspec phase
    uses this directly. Call :meth:`bind_source_tools` once a usable
    source-layer materializer exists to get a full ``ServiceHost``.

    The model entry points (:meth:`llm_heavy` etc.) forward to :attr:`models`;
    consumers pick a tier per pass rather than sharing one fixed model."""

    models: ModelProvider
    rag_tools: tuple[BaseTool, ...]
    sort: Sort

    def llm_heavy(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> LLM:
        return self.models.llm_heavy(cache_level=cache_level, disable_thinking=disable_thinking)

    def llm_lite(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> LLM:
        return self.models.llm_lite(cache_level=cache_level, disable_thinking=disable_thinking)

    def builder_heavy(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> Builder[None, None, None]:
        return self.models.builder_heavy(cache_level=cache_level, disable_thinking=disable_thinking)

    def builder_lite(
        self, *, cache_level: CacheLevel = CacheLevel.SHORT, disable_thinking: bool = False
    ) -> Builder[None, None, None]:
        return self.models.builder_lite(cache_level=cache_level, disable_thinking=disable_thinking)

    def bind_source_tools(self, tools: Sequence[BaseTool]) -> "ServiceHost":
        return ServiceHost(
            models=self.models,
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

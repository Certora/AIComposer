"""Entry point for the NatSpec multi-agent pipeline TUI.

This driver covers the ``greenfield`` and ``update`` natspec workflows.
``existing`` (verify-as-is from source) lives in ``console_autoprove``.
"""

import composer.bind as _

import argparse
import asyncio
import json
import pathlib
import uuid
from typing import Any, cast, Protocol

from langchain_core.tools import BaseTool

from graphcore.tools.memory import async_memory_tool
from graphcore.tools.vfs import DirBackend, FSBackend, Materializer, fs_tools_layered

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import create_llm, standard_connections
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.spec._env_common import build_rag_tool_env
from composer.spec.service_host import PureServiceHost
from composer.spec.gen_types import TypedTemplate
from composer.spec.system_model import Application, FromSourceApplication
from composer.spec.natspec.models import (
    InterfaceResult,
    LocatedInterfaceDecl, AutoInterfaceDecl,
    LocatedStubDeclaration, AutoStubDeclaration,
)
from composer.spec.natspec.task_description import (
    MentalModel, AgentDescription, InterfaceGenCallParams, StubGenCallParams,
)

from composer.spec.context import (
    WorkflowContext, SystemDoc, get_document_input,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline
from composer.spec.util import string_hash, FS_FORBIDDEN_READ
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.ui.tool_display import async_tool_context

from composer.ui.pipeline_app import NatspecPipelineApp


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

class PipelineArgs(ModelOptions, RAGDBOptions, Protocol):
    input_file: str
    contract_name: str
    solc_version: str
    max_concurrent: int
    cache_ns: str | None
    memory_ns: str | None
    source_root: str | None
    forbidden_read: str | None
    prover_conf: str | None


# ---------------------------------------------------------------------------
# MentalModel construction
# ---------------------------------------------------------------------------

_InterfaceTemplate = TypedTemplate[dict[str, Any]]("interface_generation_prompt.j2")
_StubTemplate = TypedTemplate[dict[str, Any]]("stub_generation_prompt.j2")


def _build_mental_model(
    *,
    source_root: pathlib.Path | None,
    config_init: dict | None,
) -> MentalModel:
    # In from-source (update) mode the agent picks file locations to fit the
    # existing project layout. In greenfield there is no layout to conform to,
    # so paths are derived automatically from the solidity identifier.
    agent_chooses_path = source_root is not None
    interface_prompt = _InterfaceTemplate.bind(
        {"agent_chooses_path": agent_chooses_path}
    ).depends(InterfaceGenCallParams)
    stub_prompt = _StubTemplate.bind(
        {"agent_chooses_path": agent_chooses_path}
    ).depends(StubGenCallParams)

    if source_root is not None:
        return MentalModel(
            model_ty=FromSourceApplication,
            interface_desc=AgentDescription(
                output_ty=InterfaceResult[LocatedInterfaceDecl],
                prompt=interface_prompt,
            ),
            stub_desc=AgentDescription(
                output_ty=LocatedStubDeclaration,
                prompt=stub_prompt,
            ),
            source_root=source_root,
            config_init=config_init,
        )
    return MentalModel(
        model_ty=Application,
        interface_desc=AgentDescription(
            output_ty=InterfaceResult[AutoInterfaceDecl],
            prompt=interface_prompt,
        ),
        stub_desc=AgentDescription(
            output_ty=AutoStubDeclaration,
            prompt=stub_prompt,
        ),
        source_root=None,
        config_init=config_init,
    )


# ---------------------------------------------------------------------------
# Source-tool factory (phase-dispatched)
# ---------------------------------------------------------------------------


def _make_source_factory(
    source_root: pathlib.Path | None,
    forbidden_read: str | None,
):
    """Build the ``source_factory`` closure the pipeline calls at each phase.

    The factory layers any extra backends the pipeline supplies (generated
    interfaces, the stub registry, etc.) over the on-disk source root, with
    the extras winning on collision. In greenfield mode there is no source
    root, so the factory just wires the extras.
    """
    base_layer: list[FSBackend] = []
    if source_root is not None:
        base_layer.append(DirBackend(source_root))

    def factory(extra_backends: list[FSBackend]) -> tuple[list[BaseTool], Materializer]:
        # Extras first — first-hit reads and reverse-order dumps mean the
        # extras' content wins over the source tree on collision.
        return fs_tools_layered(
            [*extra_backends, *base_layer],
            forbidden_read=forbidden_read,
        )

    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="NatSpec multi-agent pipeline TUI (greenfield / update)"
    )
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument("input_file", help="Path to the design document (text or PDF)")
    parser.add_argument("--solc-version", default="8.29", help="Solidity compiler version (default: 8.29)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument(
        "--source-root", default=None,
        help="Path to an existing codebase root. When set, natspec runs in `update` mode: "
             "contracts are tagged unchanged/edited/new, and specs are generated only for "
             "the new contracts. When unset, natspec runs in `greenfield` mode.",
    )
    parser.add_argument(
        "--forbidden-read", default=None,
        help="Regex of paths source tools may not read. Defaults to FS_FORBIDDEN_READ "
             "when source-root is set.",
    )
    parser.add_argument(
        "--prover-conf", default=None,
        help="Path to a Certora config JSON file whose keys (packages, link, solc_args, etc.) "
             "are merged into every typecheck invocation. Dynamic keys (files, verify, solc, "
             "compilation_steps_only) are always set by the pipeline.",
    )

    args = cast(PipelineArgs, parser.parse_args())

    # Read input document (handles both text and PDF)
    input_path = pathlib.Path(args.input_file)
    content = get_document_input(input_path)
    if content is None:
        print(f"Error: cannot read {input_path}")
        return 1
    system_doc = SystemDoc(content=content)

    # Set up services
    llm = create_llm(args)
    model = get_model()

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conn,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag,
        async_tool_context()
    ):
        source_root_path: pathlib.Path | None = None
        if args.source_root:
            source_root_path = pathlib.Path(args.source_root).resolve()

        sort = "update" if source_root_path is not None else "greenfield"
        forbidden_read = (
            args.forbidden_read or (FS_FORBIDDEN_READ if source_root_path else None)
        )

        rag_env = build_rag_tool_env(
            llm=llm,
            checkpoint=conn.checkpointer,
            db=rag,
            cvl_cache_ns=DEFAULT_CVL_AGENT_INDEX_NS,
            kb_ns=DEFAULT_KB_NS,
            store=conn.indexed_store,
        )

        start_env = PureServiceHost(
            llm=rag_env.llm,
            builder=rag_env.builder,
            cvl_tools=tuple(rag_env.rag_tools),
            sort=sort,
        )

        source_factory = _make_source_factory(source_root_path, forbidden_read)

        config_init: dict | None = None
        if args.prover_conf:
            config_init = json.loads(pathlib.Path(args.prover_conf).read_text())

        mental_model = _build_mental_model(
            source_root=source_root_path,
            config_init=config_init,
        )

        cache_root = (args.cache_ns, string_hash(str(system_doc.content))) if args.cache_ns else None

        thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        ctx = WorkflowContext.create(
            services=lambda ns: async_memory_tool(conn.memory(ns)),
            thread_id=thread_id,
            store=conn.store,
            cache_namespace=cache_root,
            memory_namespace=args.memory_ns,
        )

        # Set up TUI
        app = NatspecPipelineApp()

        async def work():
            try:
                result = await run_natspec_pipeline(
                    system_doc=system_doc,
                    solc_version=args.solc_version,
                    start_env=start_env,
                    ctx=ctx,
                    store=conn.store,
                    handler_factory=app.make_handler,
                    mental_model=mental_model,
                    source_factory=source_factory,
                    max_concurrent=args.max_concurrent,
                )
                await app.on_pipeline_done(result)
            except Exception as exc:
                app.notify(f"Pipeline failed: {exc}", severity="error")
                app._pipeline_done = True

        app.set_work(work)
        await app.run_async()
        return 0


def main() -> int:
    return asyncio.run(_main())

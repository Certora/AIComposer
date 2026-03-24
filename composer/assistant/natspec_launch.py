import traceback
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from graphcore.graph import Builder
from graphcore.tools.memory import memory_tool as make_memory_tool

from composer.assistant.launch_args import LaunchNatSpecArgs
from composer.assistant.types import OrchestratorContext
from composer.ui.pipeline_app import PipelineApp
from composer.kb.knowledge_base import kb_tools
from composer.rag.db import create_rag_db, ComposerRAGDB
from composer.spec.context import (
    WorkflowContext, SystemDoc, PlainBuilder, CVLOnlyBuilder, get_system_doc,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline, PipelineResult
from composer.spec.util import string_hash
from composer.templates.loader import load_jinja_template
from composer.tools.search import cvl_manual_tools
from composer.workflow.services import create_llm, get_checkpointer, get_store, get_memory


class _PipelineServices:
    """Concrete WorkflowServices for the orchestrator-launched pipeline."""

    def __init__(self, checkpointer: Checkpointer, store: BaseStore):
        self._checkpointer = checkpointer
        self._store = store

    def kb_tools(self, read_only: bool) -> list[BaseTool]:
        return kb_tools(self._store, ("natspec_pipeline", "kb"), read_only)

    def memory_tool(self, namespace: str) -> BaseTool:
        return make_memory_tool(get_memory(namespace))

    @property
    def checkpointer(self) -> Checkpointer:
        return self._checkpointer


def _make_builders(
    llm: BaseChatModel, rag_db: ComposerRAGDB,
) -> tuple[PlainBuilder, CVLOnlyBuilder, CVLOnlyBuilder]:
    base = Builder().with_llm(llm).with_loader(load_jinja_template)
    cvl_tools = cvl_manual_tools(rag_db)
    cvl = base.with_tools(cvl_tools)
    return base, cvl, cvl


async def launch_natspec_workflow(
    args: LaunchNatSpecArgs,
    ctx: OrchestratorContext,
) -> str:
    input_path = ctx.workspace / args.input_file
    content = get_system_doc(input_path)
    if content is None:
        return f"Error: cannot read {input_path}"
    system_doc = SystemDoc(content=content)

    pipeline_llm = create_llm(ctx.config)
    checkpointer = get_checkpointer()
    store = get_store()
    rag_db = create_rag_db(ctx.config.rag_db)

    services = _PipelineServices(checkpointer, store)
    analysis_builder, cvl_authorship, cvl_research = _make_builders(pipeline_llm, rag_db)

    thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
    cache_root = (args.cache_namespace, string_hash(str(system_doc.content))) if args.cache_namespace else None

    wf_ctx = WorkflowContext.create(
        services=services,
        thread_id=thread_id,
        store=store,
        cache_namespace=cache_root,
        memory_namespace=args.memory_namespace or None,
    )

    app = PipelineApp(ide=ctx.ide)
    pipeline_result: PipelineResult | None = None
    captured_error: Exception | None = None

    async def work() -> None:
        nonlocal pipeline_result, captured_error
        try:
            pipeline_result = await run_natspec_pipeline(
                system_doc=system_doc,
                solc_version=args.solc_version,
                analysis_builder=analysis_builder,
                cvl_research=cvl_research,
                ctx=wf_ctx,
                store=store,
                handler_factory=app.make_handler,
            )
            await app.on_pipeline_done(pipeline_result)
        except Exception as exc:
            captured_error = exc
            app.notify(f"Pipeline failed: {exc}", severity="error")
            app._pipeline_done = True

    app.set_work(work)
    await app.run_async()

    if captured_error is not None:
        tb = "".join(traceback.format_exception(captured_error))
        return (
            f"NatSpec pipeline crashed with "
            f"{type(captured_error).__name__}: {captured_error}\n"
            f"Traceback:\n{tb}"
        )
    if pipeline_result is not None:
        n_fail = len(pipeline_result.failures)
        if n_fail == 0:
            return "NatSpec pipeline completed successfully. All properties formalized."
        failures = "; ".join(
            f"{f.prop.description}: {f.reason}"
            for f in pipeline_result.failures
        )
        return f"NatSpec pipeline completed with {n_fail} failure(s): {failures}"
    return "NatSpec pipeline finished without producing a result."

import traceback
import uuid


from graphcore.tools.memory import memory_tool as make_memory_tool

from composer.assistant.launch_args import LaunchNatSpecArgs
from composer.assistant.types import OrchestratorContext
from composer.ui.pipeline_app import PipelineApp
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.spec.context import (
    WorkflowContext, SystemDoc, get_system_doc,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline, PipelineResult
from composer.spec.util import string_hash
from composer.workflow.services import create_llm, get_checkpointer, get_store, get_memory, get_indexed_store
from composer.spec.services import build_natspec_env

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
    the_model = get_model()
    rag_db = PostgreSQLRAGDatabase(
        conn_string=ctx.config.rag_db, model=get_model(),
    )

    thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
    cache_root = (args.cache_namespace, string_hash(str(system_doc.content))) if args.cache_namespace else None

    wf_ctx = WorkflowContext.create(
        services=lambda ns: make_memory_tool(get_memory(ns)),
        thread_id=thread_id,
        store=store,
        cache_namespace=cache_root,
        memory_namespace=args.memory_namespace or None,
    )

    env = build_natspec_env(
        llm=pipeline_llm,
        db=rag_db,
        checkpoint=checkpointer,
        kb_ns=("cvl",),
        store=get_indexed_store(DefaultEmbedder(the_model))
    )

    app = PipelineApp(ide=ctx.ide)
    pipeline_result: PipelineResult | None = None
    captured_error: Exception | None = None

    async def work() -> None:
        nonlocal pipeline_result, captured_error
        try:
            pipeline_result = await run_natspec_pipeline(
                system_doc=system_doc,
                tool_env=env,
                solc_version=args.solc_version,
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

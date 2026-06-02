import pathlib
import traceback
import uuid

from typing import cast


from graphcore.tools.memory import async_memory_tool as make_memory_tool

from composer.assistant.launch_args import LaunchNatSpecArgs
from composer.assistant.types import OrchestratorContext
from composer.ui.pipeline_app import PipelineApp
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.spec.context import (
    WorkflowContext, SystemDoc,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline, PipelineResult, ComponentGenerationFailure
from composer.spec.util import string_hash
from composer.workflow.services import create_llm, standard_connections
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.spec.services import build_rag_tool_env
from composer.spec.agent_index import agent_index_config_from_env
from composer.cli.natspec_startup import make_source_factory, build_mental_model
from composer.ui.tool_display import async_tool_context

async def launch_natspec_workflow(
    args: LaunchNatSpecArgs,
    ctx: OrchestratorContext,
) -> str:
    input_path = ctx.workspace / args.input_file

    pipeline_llm = create_llm(ctx.config)
    the_model = get_model()
    async with (
        standard_connections(embedder=DefaultEmbedder(the_model)) as conn,
        PostgreSQLRAGDatabase.rag_context(
            the_model, ctx.config.rag_db
        ) as rag_db,
        async_tool_context()
    ):
        content = await conn.uploader.get_document(input_path)
        if content is None:
            return f"Error: cannot read {input_path}"
        system_doc = SystemDoc(content=content)

        thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        cache_root = (args.cache_namespace, system_doc.content.to_digest()) if args.cache_namespace else None


        wf_ctx = WorkflowContext.create(
            services=lambda ns: make_memory_tool(conn.memory(ns)),
            thread_id=thread_id,
            store=conn.store,
            recursion_limit=ctx.config.recursion_limit,
            cache_namespace=cache_root,
            memory_namespace=args.memory_namespace or None,
        )

        service_env = build_rag_tool_env(
            sort="update" if args.source_root is not None else "greenfield",
            llm=pipeline_llm,
            checkpoint=conn.checkpointer,
            cvl_index_config=agent_index_config_from_env(DEFAULT_CVL_AGENT_INDEX_NS),
            db=rag_db,
            kb_ns=DEFAULT_KB_NS,
            store=conn.indexed_store,
            recursion_limit=ctx.config.recursion_limit,
        )


        source_root_path: pathlib.Path | None = None
        if args.source_root:
            source_root_path = (ctx.workspace / args.source_root).resolve()

        # Resolve output_root: explicit > derived from cache_namespace > generic.
        if args.output_root:
            output_root_path = (ctx.workspace / args.output_root).resolve()
        elif args.cache_namespace:
            output_root_path = (
                ctx.workspace / "natspec_output" / args.cache_namespace
            ).resolve()
        else:
            output_root_path = (ctx.workspace / "natspec_output").resolve()

        app = PipelineApp(
            ide=ctx.ide,
            system_doc_path=input_path.resolve(),
            source_root=source_root_path,
            prover_conf=args.prover_conf,
            output_root=output_root_path,
        )
        # don't let pyright conclude these variables remain None (they are mutated in work)
        pipeline_result: PipelineResult | None = cast(PipelineResult | None, None)
        captured_error: Exception | None = cast(Exception | None, None)

        async def work() -> None:
            nonlocal pipeline_result, captured_error
            try:
                pipeline_result = await run_natspec_pipeline(
                    system_doc=system_doc,
                    start_env=service_env,
                    solc_version=args.solc_version,
                    ctx=wf_ctx,
                    store=conn.store,
                    handler_factory=app.make_handler,
                    mental_model=build_mental_model(
                        source_root=source_root_path,
                        config_init=args.prover_conf
                    ),
                    source_factory=make_source_factory(
                        source_root=source_root_path,
                        forbidden_read=args.forbidden_read
                    ),
                    interactive=args.interactive,
                )
                await app.on_pipeline_done(pipeline_result)
            except Exception as exc:
                captured_error = exc
                app.notify(f"Pipeline failed: {exc}", severity="error", markup=False)
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
        plan_path = output_root_path / "implementation_plan.json"
        plan_info = (
            f" Plan written to: {plan_path}"
            if plan_path.is_file()
            else " (plan was not persisted; check pipeline logs)"
        )

        if pipeline_result is not None:
            failures_obj: list[ComponentGenerationFailure] = []
            for rc in pipeline_result.contracts:
                for f in rc.spec_results.failures:
                    failures_obj.append(f)
            if len(failures_obj) == 0:
                return (
                    "NatSpec pipeline completed successfully. All properties "
                    f"formalized.{plan_info}"
                )
            failures = "; ".join(
                f"{f.component}: {f.reason}" for f in failures_obj
            )
            return (
                f"NatSpec pipeline completed with {len(failures_obj)} failure(s): "
                f"{failures}.{plan_info}"
            )
        return "NatSpec pipeline finished without producing a result."

"""Properties-driven auto-prove pipeline (single conf).

Three sequential phases:

0. AutoSetup — if ``config_path`` is ``None``, invoke ``run_autosetup`` for the
   user's main contract; the result carries the single ``prover_config`` dict
   and the path to AutoSetup's generated summaries spec. If ``config_path`` is
   supplied, skip this phase entirely, parse the .conf, and proceed with no
   summaries resource.
1. Property formulation — one LLM call turns the user's plain-English property
   list into ``list[PropertyFormulation]`` for this conf.
2. CVL generation — one ``batch_cvl_generation`` call against the formulated
   property list; writes ``certora/autospec_<stem>.spec`` plus a companion
   commentary file when the generation succeeded.
"""

import json
import pathlib

from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import HandlerFactory, TaskInfo, run_task
from composer.prover.core import ProverOptions
from composer.spec.context import (
    CacheKey,
    Properties,
    SourceCode,
    WorkflowContext,
)
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.gen_types import CVLResource
from composer.spec.source.author import GaveUp
from composer.spec.source.autosetup import SetupFailure, run_autosetup
from composer.spec.source.common_pipeline import (
    AutoProveResult, autosetup_summaries_resource, run_cached_batch_cvl,
)
from composer.spec.source.direct_property import (
    main_contract_from_config,
    run_direct_property_formulation,
)
from composer.spec.source.prover import get_prover_tool
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.util import string_hash
from composer.ui.autoprove_app import AutoProvePhase


PROPERTIES_KEY = CacheKey[None, Properties]("properties")


def _conf_cache_key(
    conf_path: str, config: dict
) -> CacheKey[Properties, GeneratedCVL]:
    combined = "|".join([conf_path, json.dumps(config, sort_keys=True)])
    return CacheKey(string_hash(combined))


async def run_properties_pipeline(
    llm: BaseChatModel,
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    *,
    properties: str | dict,
    config_path: str | None,
    prover_opts: ProverOptions,
    max_concurrent: int = 4,
    interactive: bool,
) -> AutoProveResult:
    """Run the properties-driven auto-prove pipeline against a single conf."""
    project_root = pathlib.Path(source_input.project_root)

    # ------------------------------------------------------------------
    # Phase 0: AutoSetup (skipped if --config-path is supplied)
    # ------------------------------------------------------------------
    resources: list[CVLResource]
    config: dict
    contract_name: str
    conf_stem: str
    conf_cache_id: str

    if config_path is None:
        setup_result = await run_task(
            handler_factory,
            TaskInfo("autosetup", "Auto Setup", AutoProvePhase.HARNESS),
            lambda: run_autosetup(
                project_root,
                source_input.relative_path,
                source_input.contract_name,
                prover_opts,
            ),
        )
        if isinstance(setup_result, SetupFailure):
            raise RuntimeError(
                f"Auto setup failed: {setup_result.error}\nProc stderr:\n{setup_result.stderr}"
            )
        config = setup_result.prover_config
        contract_name = source_input.contract_name
        conf_stem = contract_name
        conf_cache_id = f"autosetup:{contract_name}"
        resources = [autosetup_summaries_resource(setup_result.summaries_path)]
    else:
        conf_path = pathlib.Path(config_path)
        config = json.loads(conf_path.read_text())
        derived = main_contract_from_config(config)
        if derived is None:
            raise ValueError(
                f"Could not determine a main contract from conf file {config_path}"
            )
        contract_name = derived
        conf_stem = conf_path.stem
        conf_cache_id = str(conf_path)
        resources = []

    # ------------------------------------------------------------------
    # Phase 1: Property formulation (single LLM call)
    # ------------------------------------------------------------------
    prop_context = ctx.child(PROPERTIES_KEY)

    props = await run_task(
        handler_factory,
        TaskInfo(
            "props",
            f"Property formulation for {conf_stem}",
            AutoProvePhase.BUG_ANALYSIS,
        ),
        lambda: run_direct_property_formulation(
            ctx=prop_context,
            env=env,
            conf_path=conf_cache_id,
            config=config,
            contract_name=contract_name,
            properties=properties,
            system_doc=source_input.content,
        ),
    )

    if not props:
        return AutoProveResult(n_components=1, n_properties=0, failures=[])

    # ------------------------------------------------------------------
    # Phase 2: CVL generation
    # ------------------------------------------------------------------
    per_conf_source = SourceCode(
        content=source_input.content,
        project_root=source_input.project_root,
        contract_name=contract_name,
        relative_path=source_input.relative_path,
        forbidden_read=source_input.forbidden_read,
    )

    conf_ctx_child = await prop_context.child(
        _conf_cache_key(conf_cache_id, config),
        {"conf": conf_cache_id, "contract_name": contract_name},
    )

    prover_tool = get_prover_tool(
        llm, contract_name, source_input.project_root, prover_opts=prover_opts,
    )
    label = f"{conf_stem} ({len(props)} properties)"
    gen_result = await run_cached_batch_cvl(
        cache_ctx=conf_ctx_child,
        handler_factory=handler_factory,
        task_info=TaskInfo("cvl", label, AutoProvePhase.CVL_GEN),
        init_config=config,
        props=props,
        component=None,
        resources=resources,
        prover_tool=prover_tool,
        env=env,
        source=per_conf_source,
    )
    if isinstance(gen_result, GaveUp):
        return AutoProveResult(
            n_components=1,
            n_properties=len(props),
            failures=[f"{conf_stem}: GAVE_UP: {gen_result.reason}"],
        )
    cvl_result = gen_result

    certora_dir = project_root / "certora"
    certora_dir.mkdir(exist_ok=True, parents=True)
    (certora_dir / f"autospec_{conf_stem}.spec").write_text(cvl_result.cvl)
    (certora_dir / f"autospec_{conf_stem}.commentary.md").write_text(cvl_result.commentary)

    return AutoProveResult(
        n_components=1,
        n_properties=len(props),
        failures=[],
    )

"""
Direct auto-prove pipeline orchestration (skip_setup mode).

Unlike the standard pipeline, the user has already run setup, so we don't
re-derive components or summaries — instead we take a list of pre-existing
Certora `.conf` files together with plain-English properties and an optional
threat model.

Phases:
1. In a single LLM call that sees every conf at once, map every user
   property to exactly one conf, producing
   `dict[conf_path, list[PropertyFormulation]]`.
2. For each conf, run `batch_cvl_generation` over its formulated properties
   to produce a CVL spec.
"""

import asyncio
import json
import pathlib
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.ui.autoprove_app import AutoProvePhase

from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties, ComponentGroup, CVLGeneration,
)
from composer.spec.util import string_hash
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import CVLResource
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.prover import CloudConfig, get_prover_tool
from composer.spec.source.common_pipeline import AutoProveResult
from composer.spec.source.author import batch_cvl_generation
from composer.spec.source.direct_property import (
    run_direct_property_formulation_all, main_contract_from_config,
)


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[None, Properties]("properties")


def _conf_cache_key(conf_path: str, config: dict) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([conf_path, json.dumps(config, sort_keys=True)])
    return CacheKey(string_hash(combined))


def _batch_cache_key(props: list[PropertyFormulation]) -> CacheKey[ComponentGroup, GeneratedCVL]:
    combined = "|".join(p.model_dump_json() for p in props)
    return CacheKey(string_hash(combined))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_autoprove_pipeline(
    llm: BaseChatModel,
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    custom_summary_path: str | None,
    standard_summary_path: str | None,
    config_paths: list[str],
    threat_model: str | dict | None = None,
    properties: str | dict | None = None,
    *,
    cloud: CloudConfig | None = None,
    max_concurrent: int = 4,
    interactive: bool,
) -> AutoProveResult:
    """Run the direct (skip_setup) auto-prove pipeline."""
    if properties is None:
        raise ValueError("--properties_path is required for the direct pipeline")
    if not config_paths:
        raise ValueError("At least one .conf path is required for the direct pipeline")

    semaphore = asyncio.Semaphore(max_concurrent)
    project_root = pathlib.Path(source_input.project_root)

    # Resources shared by every conf's CVL generation.
    resources: list[CVLResource] = []
    if standard_summary_path:
        resources.append(CVLResource(
            import_path=standard_summary_path,
            required=True,
            description="PreAudit-generated summaries",
            sort="import",
        ))
    if custom_summary_path:
        resources.append(CVLResource(
            import_path=custom_summary_path,
            required=False,
            description="Custom CVL summaries specific to the contract",
            sort="import",
        ))

    prop_context = ctx.child(PROPERTIES_KEY)

    # ------------------------------------------------------------------
    # Phase 1: A single LLM call sees every conf file at once and produces
    # a mapping conf_path => list[PropertyFormulation], so every user
    # property is assigned to exactly one conf with full coverage.
    # ------------------------------------------------------------------

    @dataclass
    class _ConfBatch:
        conf_path: str
        config: dict
        contract_name: str
        props: list[PropertyFormulation]
        conf_ctx: WorkflowContext[ComponentGroup]
        source: SourceCode

    parsed_confs: list[tuple[str, dict, str]] = []  # (conf_path, config, contract_name)
    for conf_path_str in config_paths:
        config = json.loads(pathlib.Path(conf_path_str).read_text())
        contract_name = main_contract_from_config(config)
        if not contract_name:
            continue
        parsed_confs.append((conf_path_str, config, contract_name))

    if not parsed_confs:
        raise ValueError(
            "Could not determine a main contract from any of the provided conf files."
        )

    mapping = await run_task(
        handler_factory,
        TaskInfo(
            "props-all",
            f"Property formulation across {len(parsed_confs)} conf(s)",
            AutoProvePhase.BUG_ANALYSIS,
        ),
        lambda: run_direct_property_formulation_all(
            ctx=prop_context,
            env=env,
            confs=[(p, cfg) for p, cfg, _ in parsed_confs],
            properties=properties,
            threat_model=threat_model,
        ),
        semaphore,
    )

    conf_batches: list[_ConfBatch] = []
    for conf_path_str, config, contract_name in parsed_confs:
        props = mapping.get(conf_path_str, [])
        if not props:
            continue
        conf_ctx = await prop_context.child(
            _conf_cache_key(conf_path_str, config),
            {
                "conf_path": conf_path_str,
                "contract_name": contract_name,
            },
        )
        per_conf_source = SourceCode(
            content=source_input.content,
            project_root=source_input.project_root,
            contract_name=contract_name,
            relative_path=source_input.relative_path,
            forbidden_read=source_input.forbidden_read,
        )
        conf_batches.append(_ConfBatch(
            conf_path=conf_path_str,
            config=config,
            contract_name=contract_name,
            props=props,
            conf_ctx=conf_ctx,
            source=per_conf_source,
        ))

    if not conf_batches:
        raise ValueError("No properties were formulated for any conf file.")

    # ------------------------------------------------------------------
    # Phase 2: For the mapping from phase 1, iterate over all conf files
    # and generate the CVL specification using the list[PropertyFormulation]
    # ------------------------------------------------------------------

    async def _generate_batch(batch_idx: int, batch: _ConfBatch) -> GeneratedCVL:
        batch_child = await batch.conf_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        )
        if (cached := await batch_child.cache_get(GeneratedCVL)) is not None:
            return cached
        batch_ctx = batch_child.abstract(CVLGeneration)

        prover_tool = get_prover_tool(
            llm,
            batch.contract_name,
            source_input.project_root,
            cloud=cloud,
        )

        label = f"{pathlib.Path(batch.conf_path).stem} ({len(batch.props)} properties)"
        res = await run_task(
            handler_factory,
            TaskInfo(f"cvl-{batch_idx}", label, AutoProvePhase.CVL_GEN),
            lambda: batch_cvl_generation(
                ctx=batch_ctx,
                init_config=batch.config,
                component=None,
                env=env,
                props=batch.props,
                prover_tool=prover_tool,
                resources=resources,
                description=label,
                source=batch.source,
            ),
            semaphore,
        )
        await batch_child.cache_put(res)
        return res

    async def _generate_and_write_batch(
        i: int, batch: _ConfBatch,
    ) -> GeneratedCVL:
        res = await _generate_batch(i, batch)
        if res.commentary.startswith("GAVE_UP:"):
            return res
        certora_dir = project_root / "certora"
        certora_dir.mkdir(exist_ok=True, parents=True)
        stem = pathlib.Path(batch.conf_path).stem
        (certora_dir / f"autospec_{stem}.spec").write_text(res.cvl)
        (certora_dir / f"autospec_{stem}.commentary.md").write_text(res.commentary)
        return res

    generation_results = await asyncio.gather(
        *[
            _generate_and_write_batch(i, batch)
            for i, batch in enumerate(conf_batches)
        ],
        return_exceptions=True,
    )

    failures: list[str] = []
    n_properties = 0
    for batch, result in zip(conf_batches, generation_results):
        n_properties += len(batch.props)
        if isinstance(result, BaseException):
            failures.append(f"{pathlib.Path(batch.conf_path).stem}: {result}")
        elif isinstance(result, GeneratedCVL) and result.commentary.startswith("GAVE_UP:"):
            failures.append(f"{pathlib.Path(batch.conf_path).stem}: {result.commentary}")

    return AutoProveResult(
        n_components=len(conf_batches),
        n_properties=n_properties,
        failures=failures,
    )

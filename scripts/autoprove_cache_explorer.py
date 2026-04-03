"""
Cache & Memory Explorer for the Auto-Prove pipeline.

Usage:
    python scripts/autoprove_cache_explorer.py <project_root> <main_contract> <system_doc> --cache-ns <ns> [--memory-ns <ns>]
"""

import argparse
import hashlib
import pathlib
import sys
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, AsyncGenerator

_repo_root = str(pathlib.Path(__file__).parent.parent.absolute())
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from composer.ui.cache_explorer import CacheNode, OrgNode, CacheTreeNode, CacheExplorerApp, DummyServices
from composer.spec.context import WorkflowContext, SourceCode, CacheKey, CacheTypes, get_system_doc, CVLGeneration, Marker, CVLJudge
from composer.spec.source.system_analysis import SOURCE_ANALYSIS_KEY
from composer.spec.source.harness import (
    config_key,
    system_setup_key,
    harness_generation_key,
    HARNESS_ANALYSIS_KEY,
    ContractSetup,
    SystemDescriptionHarnessed,
    AgentSystemDescription,
    HarnessResult,
)
from composer.spec.source.summarizer import _summary_key, _SummaryCache
from composer.spec.source.struct_invariant import STRUCTURAL_INV_KEY, Invariants
from composer.spec.source.common_pipeline import PROPERTIES_KEY, INV_CVL_KEY, _component_cache_key, _batch_cache_key
from composer.spec.bug import _BugAnalysisCache, BUG_ANALYSIS_KEY
from composer.spec.cvl_generation import GeneratedCVL, _LastAttemptCache, LAST_ATTEMPT_KEY, CVL_JUDGE_KEY
from composer.spec.system_model import (
    SourceApplication, SourceExplicitContract, SourceExternalActor,
    HarnessedApplication, HarnessedExplicitContract, HarnessDefinition,
    ContractInstance, ContractComponentInstance,
)


# ---------------------------------------------------------------------------
# Cache value type
# ---------------------------------------------------------------------------

type AutoProveCachedValue = (
    SourceApplication
    | ContractSetup
    | SystemDescriptionHarnessed
    | AgentSystemDescription
    | HarnessResult
    | _SummaryCache
    | Invariants
    | GeneratedCVL
    | _LastAttemptCache
    | _BugAnalysisCache
    | CVLJudge
)


# ---------------------------------------------------------------------------
# Tree construction helpers
# ---------------------------------------------------------------------------

_node_context: ContextVar[CacheTreeNode[AutoProveCachedValue] | None] = ContextVar(
    "_node_context", default=None
)


@contextmanager
def node(c: CacheTreeNode[AutoProveCachedValue]):
    prev = _node_context.get()
    if prev is not None:
        prev.children.append(c)
    tok = _node_context.set(c)
    try:
        yield
    finally:
        _node_context.reset(tok)


@asynccontextmanager
async def node_for[T: CacheTypes, S: CacheTypes](
    ctx: WorkflowContext[T],
    child: CacheKey[T, S],
    label: str,
    ty: type[S] | None = None,
) -> AsyncIterator[tuple[WorkflowContext[S], S | None]]:
    child_ctx = ctx.child(child)
    value: S | None = await child_ctx.cache_get(ty) if ty is not None else None  # type: ignore[arg-type]
    new_node: CacheNode[AutoProveCachedValue] = CacheNode(label=label, ctx=child_ctx, value=value)  # type: ignore[arg-type]
    with node(new_node):
        yield child_ctx, value


async def leaf[T: CacheTypes, S: AutoProveCachedValue](
    ctx: WorkflowContext[T], child: CacheKey[T, S], label: str, ty: type[S]
) -> CacheNode[S]:
    child_ctx = ctx.child(child)
    value: S | None = await child_ctx.cache_get(ty)
    return CacheNode(label=label, ctx=child_ctx, value=value)

def memory[T: CacheTypes, S: Marker](ctx: WorkflowContext[T], child: CacheKey[T, S], label: str):
    return CacheNode[S](label=label, value=None, ctx=ctx.child(child))

# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_harnessed_app(
    sa: SourceApplication,
    config_val: ContractSetup | None,
) -> HarnessedApplication:
    """Reconstruct the HarnessedApplication the same way pipeline.py does."""
    contract_to_harness: dict[str, list[HarnessDefinition]] = {}
    if config_val is not None:
        for c in config_val.system_description.transitive_closure:
            if not c.harness_definition:
                continue
            contract_to_harness.setdefault(c.harness_definition.harness_of, []).append(
                HarnessDefinition(name=c.name, path=c.path)
            )

    comp: list[SourceExternalActor | HarnessedExplicitContract] = []
    for c in sa.components:
        if not isinstance(c, SourceExplicitContract):
            comp.append(c)
            continue
        comp.append(HarnessedExplicitContract(
            sort=c.sort,
            name=c.name,
            components=c.components,
            description=c.description,
            path=c.path,
            harnesses=contract_to_harness.get(c.name, []),
        ))

    return HarnessedApplication(
        application_type=sa.application_type,
        description=sa.description,
        components=comp,
    )


async def _build_cvl_gen_nodes(
    ctx: WorkflowContext[CVLGeneration],
) -> AsyncGenerator[CacheTreeNode[AutoProveCachedValue], None]:
    yield memory(ctx, child=CVL_JUDGE_KEY, label="Feedback")
    yield await leaf(ctx, LAST_ATTEMPT_KEY, "Last Attempt", _LastAttemptCache)


async def _build_component_nodes(
    prop_ctx: WorkflowContext,
    feat: ContractComponentInstance,
) -> AsyncGenerator[CacheTreeNode[AutoProveCachedValue], None]:
    comp_key = _component_cache_key(feat)
    async with node_for(prop_ctx, comp_key, feat.component.name) as (feat_ctx, _):
        bug_leaf = await leaf(feat_ctx, BUG_ANALYSIS_KEY, "Bug Analysis", _BugAnalysisCache)
        yield bug_leaf
        if bug_leaf.value is None:
            return
        batch_key = _batch_cache_key(bug_leaf.value.items)
        async with node_for(feat_ctx, batch_key, "CVL Generation", GeneratedCVL) as (cvl_ctx, _):
            async for n in _build_cvl_gen_nodes(cvl_ctx.abstract(CVLGeneration)):
                yield n


async def build_tree_inner(
    root_ctx: WorkflowContext[None],
    contract_name: str,
) -> AsyncGenerator[CacheTreeNode[AutoProveCachedValue], None]:
    sa_leaf = await leaf(root_ctx, SOURCE_ANALYSIS_KEY, "source-analysis", SourceApplication)
    yield sa_leaf

    # Read config value upfront so we can derive the summary key outside the with block
    config_val = await root_ctx.child(config_key).cache_get(ContractSetup)

    async with node_for(root_ctx, config_key, "config", ContractSetup) as (config_ctx, _):
        if sa_leaf.value is not None:
            async with node_for(config_ctx, system_setup_key(sa_leaf.value), "setup", SystemDescriptionHarnessed) as (setup_ctx, _):
                ha_leaf = await leaf(setup_ctx, HARNESS_ANALYSIS_KEY, "harness-analysis", AgentSystemDescription)
                yield ha_leaf
                if ha_leaf.value is not None and ha_leaf.value.needs_harnessing():
                    yield await leaf(
                        setup_ctx,
                        harness_generation_key(ha_leaf.value),
                        "harness-generation",
                        HarnessResult,
                    )

    # Summary — key derivable only once ContractSetup is cached
    if config_val is not None:
        yield await leaf(root_ctx, _summary_key(config_val), "summary", _SummaryCache)

    yield await leaf(root_ctx, STRUCTURAL_INV_KEY, "structural-inv", Invariants)
    async with node_for(root_ctx, INV_CVL_KEY, "invariant-cvl", GeneratedCVL) as (inv_cvl_ctx, _):
        gen_ctx = inv_cvl_ctx.abstract(CVLGeneration)
        yield await leaf(
            gen_ctx, LAST_ATTEMPT_KEY, "Last Attempt", _LastAttemptCache
        )
        yield memory(
            gen_ctx, child=CVL_JUDGE_KEY, label="Feedback"
        )

    # Properties — per-component bug analysis + CVL generation
    if sa_leaf.value is None:
        with node(OrgNode(label="properties (no source analysis)")):
            pass
        return

    harnessed_app = _build_harnessed_app(sa_leaf.value, config_val)

    # Find the main contract
    contract_ind = -1
    for i, c in enumerate(harnessed_app.contract_components):
        if c.name == contract_name:
            contract_ind = i
            break

    if contract_ind == -1:
        with node(OrgNode(label=f"properties (contract '{contract_name}' not found)")):
            pass
        return

    contract_instance = ContractInstance(contract_ind, app=harnessed_app)
    prop_ctx = root_ctx.child(PROPERTIES_KEY)

    with node(OrgNode(label="properties")):
        for comp_idx in range(len(contract_instance.contract.components)):
            feat = ContractComponentInstance(_contract=contract_instance, ind=comp_idx)
            async for n in _build_component_nodes(prop_ctx, feat):
                yield n


async def build_tree(root_ctx: WorkflowContext[None], contract_name: str) -> CacheNode[AutoProveCachedValue]:
    root: CacheNode[AutoProveCachedValue] = CacheNode(label="root", ctx=root_ctx)
    with node(root):
        async for n in build_tree_inner(root_ctx, contract_name):
            curr = _node_context.get()
            assert curr is not None
            curr.children.append(n) #type: ignore
    return root


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_value(val: AutoProveCachedValue) -> list[str]:
    lines: list[str] = []

    match val:
        case SourceApplication(application_type=app_type, description=desc, components=comps):
            lines.append(f"Type: {app_type}")
            lines.append(f"Description: {desc}")
            lines.append("")
            for c in comps:
                match c:
                    case SourceExplicitContract(name=name, sort=sort, path=path, description=cdesc):
                        lines.append(f"[{sort}] {name}  ({path})")
                        lines.append(f"  {cdesc}")
                    case SourceExternalActor(name=name, path=path, description=cdesc):
                        loc = f"  ({path})" if path else ""
                        lines.append(f"[external] {name}{loc}")
                        lines.append(f"  {cdesc}")

        case ContractSetup(system_description=sys_desc, config=cfg):
            lines.append("Pre-audit setup: OK")
            lines.append(f"Summaries path: {cfg.summaries_path}")
            lines.append(f"User types: {len(cfg.user_types)}")
            lines.append(f"Closure contracts: {len(sys_desc.transitive_closure)}")

        case AgentSystemDescription(
            non_trivial_state=nts,
            erc20_contracts=erc20s,
            external_interfaces=ext_ifaces,
            transitive_closure=closure,
        ):
            lines.append(f"Non-trivial state: {nts}")
            lines.append(f"ERC20 contracts: {', '.join(erc20s) if erc20s else 'none'}")
            lines.append(f"Needs harnessing: {val.needs_harnessing()}")
            lines.append("")
            lines.append(f"Transitive closure ({len(closure)}):")
            for c in closure:
                instances = f"  x{c.num_instances}" if c.num_instances else ""
                lines.append(f"  {c.name}{instances}")
                for lf in c.link_fields:
                    lines.append(f"    links → {', '.join(lf.target)}")
            if ext_ifaces:
                lines.append("")
                lines.append(f"External interfaces ({len(ext_ifaces)}):")
                for ei in ext_ifaces:
                    lines.append(f"  {ei.name}: {ei.behavioral_spec}")

        case SystemDescriptionHarnessed(
            non_trivial_state=nts,
            erc20_contracts=erc20s,
            external_interfaces=ext_ifaces,
            transitive_closure=closure,
        ):
            lines.append(f"Non-trivial state: {nts}")
            lines.append(f"ERC20 contracts: {', '.join(erc20s) if erc20s else 'none'}")
            lines.append("")
            lines.append(f"Transitive closure ({len(closure)}):")
            for c in closure:
                harnessed = " [harnessed]" if c.harness_definition else ""
                lines.append(f"  {c.name}  ({c.path}){harnessed}")
                if c.harness_definition:
                    lines.append(f"    harness of: {c.harness_definition.harness_of}")
                for lf in c.link_fields:
                    lines.append(f"    links → {', '.join(lf.target)} via {", ".join(lf.link_paths)}")
            if ext_ifaces:
                lines.append("")
                lines.append(f"External interfaces ({len(ext_ifaces)}):")
                for ei in ext_ifaces:
                    lines.append(f"  {ei.name}: {ei.behavioral_spec}")

        case HarnessResult(name_to_source=name_to_source):
            for contract, harnesses in name_to_source.items():
                lines.append(f"{contract}:")
                for h in harnesses:
                    lines.append(f"  {h.harness_name}  →  {h.path}")

        case _SummaryCache(content=content):
            for line in content.splitlines()[:40]:
                lines.append(line)
            if len(content.splitlines()) > 40:
                lines.append(f"... ({len(content.splitlines()) - 40} more lines)")

        case Invariants(inv=invs):
            lines.append(f"Invariants ({len(invs)}):")
            for inv in invs:
                lines.append(f"  {inv.description}")

        case GeneratedCVL(commentary=commentary, cvl=cvl, skipped=skipped):
            lines.append(f"Commentary: {commentary}")
            if skipped:
                lines.append(f"Skipped: {len(skipped)}")
            lines.append("")
            for line in cvl.splitlines()[:40]:
                lines.append(line)
            if len(cvl.splitlines()) > 40:
                lines.append(f"... ({len(cvl.splitlines()) - 40} more lines)")
        
        case _BugAnalysisCache(items=items):
            lines.append(f"Properties ({len(items)}):")
            for p in items:
                methods = p.methods
                lines.append(f"  - [{p.sort}] {p.description}")
                if isinstance(methods, list):
                    lines.append(f"    methods: {', '.join(methods)}")
                else:
                    lines.append(f"    methods: {methods}")

        case _LastAttemptCache(cvl=cvl):
            lines.append("LAST ATTEMPT")
            lines.append(cvl)

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _root_cache_key(
    project_root: str,
    system_doc_path: pathlib.Path,
    relative_path: str,
    contract_name: str,
) -> str:
    """Same logic as tui_autoprove._root_cache_key."""
    doc_hash = hashlib.sha256(system_doc_path.read_bytes()).hexdigest()
    combined = "|".join([project_root, doc_hash, relative_path, contract_name])
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache & Memory Explorer for Auto-Prove pipeline"
    )
    parser.add_argument("project_root", help="Root directory of the Solidity project")
    parser.add_argument("main_contract", help="Main contract as path:ContractName")
    parser.add_argument("system_doc", help="Path to the design document (text or PDF)")
    parser.add_argument("--cache-ns", required=True, dest="cache_ns",
                        help="Cache namespace (same as passed to tui_autoprove.py)")
    parser.add_argument("--memory-ns", dest="memory_ns", default=None,
                        help="Memory namespace (enables memory browsing)")

    args = parser.parse_args()

    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)
    full_contract_path = pathlib.Path(main_contract_path).resolve()
    relative_path = str(full_contract_path.relative_to(project_root))

    sys_path = pathlib.Path(args.system_doc)
    content = get_system_doc(sys_path)
    if content is None:
        print(f"Error: cannot read {sys_path}")
        return 1

    from composer.workflow.services import get_store
    store = get_store()

    root_ns = (args.cache_ns, _root_cache_key(
        str(project_root), sys_path, relative_path, contract_name,
    ))
    print(f"Root namespace: {root_ns}")

    root_ctx: WorkflowContext[None] = WorkflowContext.create(
        services=DummyServices(),  # type: ignore[arg-type]
        thread_id="explorer",
        store=store,
        memory_namespace=args.memory_ns,
        cache_namespace=root_ns,
    )

    status = f"Cache NS: {root_ns}"
    if args.memory_ns:
        status += f"  |  Memory NS: {args.memory_ns}"

    app = CacheExplorerApp(
        build_tree=lambda: build_tree(root_ctx, contract_name),
        format_value=format_value,
        store=store,
        status=status,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Cache & Memory Explorer for the NatSpec pipeline.

Usage:
    python scripts/cache_explorer.py <input_file> --cache-ns <ns> [--memory-ns <ns>]
"""

import argparse
import sys
from pathlib import Path
from contextvars import ContextVar
from contextlib import contextmanager, asynccontextmanager
from typing import Awaitable

_repo_root = str(Path(__file__).parent.parent.absolute())
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from composer.ui.cache_explorer import CacheNode, CacheExplorerApp, DummyServices, CacheTreeNode, OrgNode
from composer.spec.context import (
    WorkflowContext, CacheKey, CVLGeneration, get_system_doc, 
    Contract, CacheTypes, Marker, ComponentGroup, Properties
)
from composer.spec.system_model import Application, ExplicitContract, ContractComponent, ExternalActor
from composer.spec.natspec.interface_gen import InterfaceResult
from composer.spec.natspec.system_analysis import SOURCE_ANALYSIS_KEY
from composer.spec.natspec.stub_gen import StubDeclaration
from composer.spec.bug import _BugAnalysisCache, BUG_ANALYSIS_KEY
from composer.spec.cvl_generation import GeneratedCVL, _LastAttemptCache, CVL_JUDGE_KEY, LAST_ATTEMPT_KEY
from composer.spec.natspec.pipeline import PROPERTIES_KEY, _component_cache_key, _batch_cache_key
from composer.spec.util import string_hash



# ---------------------------------------------------------------------------
# NatSpec cache value type
# ---------------------------------------------------------------------------

type NatSpecCachedValue = (
    Application | InterfaceResult | StubDeclaration
    | _BugAnalysisCache | GeneratedCVL | _LastAttemptCache
)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

_node_context = ContextVar[CacheTreeNode[NatSpecCachedValue] | None]("_node_context", default=None)

@contextmanager
def node(c: CacheTreeNode[NatSpecCachedValue]):
    prev = _node_context.get()
    if prev is not None:
        prev.children.append(c)
    tok = _node_context.set(c)
    try:
        yield
    finally:
        _node_context.reset(tok)

@contextmanager
def section(s: str):
    with node(OrgNode(s)):
        yield

@asynccontextmanager
async def node_for[T : CacheTypes, S : CacheTypes](ctx: WorkflowContext[T], child: CacheKey[T, S], label: str, ty: type[S] | None = None):
    child_ctx = ctx.child(child)
    value : S | None = None
    if ty is not None:
        value = await child_ctx.cache_get(ty)
    new_node = CacheNode(
        label=label,
        ctx=child_ctx,
        value = value
    )
    with node(new_node): #type: ignore
        yield child_ctx

async def leaf[T : CacheTypes, S : NatSpecCachedValue](ctx: WorkflowContext[T], child: CacheKey[T, S], label: str, ty: type[S]) -> CacheNode[S]:
    child_ctx = ctx.child(child)
    value : S | None = await child_ctx.cache_get(ty)
    return CacheNode[S](label=label, value=value, ctx=child_ctx)

def memory[T: CacheTypes, S: Marker](ctx: WorkflowContext[T], child: CacheKey[T, S], label: str):
    return CacheNode[S](label=label, value=None, ctx=ctx.child(child))

async def build_cvl_generation_node(ctx: WorkflowContext[CVLGeneration]):
    yield memory(ctx, CVL_JUDGE_KEY, "Feedback judge")
    yield (await leaf(ctx, LAST_ATTEMPT_KEY, "Last attempt", _LastAttemptCache))

async def build_component_tree(contract_ctx: WorkflowContext[Properties], key: CacheKey[Properties, ComponentGroup], comp: ContractComponent):
    async with node_for(contract_ctx, key, comp.name) as feat_ctx:
        d = await leaf(feat_ctx, BUG_ANALYSIS_KEY, "Bug Analysis", _BugAnalysisCache)
        yield d
        if d.value is None:
            return
        async with node_for(
            feat_ctx,
            _batch_cache_key(d.value.items),
            "CVL Generation",
            GeneratedCVL
        ) as cvl_ctx:
            async for t in build_cvl_generation_node(
                cvl_ctx.abstract(CVLGeneration)
            ): yield t

async def build_contract_tree(contract_ctx: WorkflowContext[Contract], contract: ExplicitContract, summ: Application):
    async with node_for(contract_ctx, PROPERTIES_KEY, "properties") as prop_ctx:
        for comp in contract.components:
            comp_key = _component_cache_key(comp, summ.application_type)
            async for t in build_component_tree(prop_ctx, comp_key, comp): yield t

async def build_tree_inner(root_ctx: WorkflowContext[None]):
    sa_ctx = root_ctx.child(SOURCE_ANALYSIS_KEY)
    summary = await sa_ctx.cache_get(Application)
    yield CacheNode(
        label="source-analysis", ctx=sa_ctx, value=summary,
    )

    cached_intf: InterfaceResult | None = None
    if summary is None:
        return
    intf_key = CacheKey[None, InterfaceResult](
        f"interface-{string_hash(summary.model_dump_json())}"
    )
    intf_ctx = root_ctx.child(intf_key)
    cached_intf = await intf_ctx.cache_get(InterfaceResult)
    yield CacheNode(
        label="interface", ctx=intf_ctx, value=cached_intf,
    )

    if cached_intf is not None:
        with section("Stubs"):
            cache_prefix = f"stub-for-{string_hash(cached_intf.model_dump_json())}-"
            for c in summary.contract_components:    
                stub_key = CacheKey[None, StubDeclaration](
                    cache_prefix + c.name
                )
                yield leaf(
                    root_ctx, stub_key, f"Stub: {c.name}", StubDeclaration
                )
    for c in summary.contract_components:
        contract_key = CacheKey[None, Contract](string_hash(c.model_dump_json()))
        contract_ctx = root_ctx.child(contract_key)
        async with node_for(root_ctx, contract_key, f"Contract: {c.name}") as contract_ctx:
            async for t in build_contract_tree(contract_ctx, c, summary): yield t


async def build_tree(root_ctx: WorkflowContext) -> CacheNode[NatSpecCachedValue]:
    """Build the NatSpec pipeline cache tree by reading the store."""

    root: CacheNode[NatSpecCachedValue] = CacheNode(label="root", ctx=root_ctx)

    with node(root):
        async for n in build_tree_inner(root_ctx):
            curr_node = _node_context.get()
            assert curr_node is not None
            curr_node.children.append(n) #type: ignore

    return root



# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_value(val: NatSpecCachedValue) -> list[str]:
    """Format a NatSpec cached value for the detail pane."""
    lines: list[str] = []

    match val:
        case GeneratedCVL(commentary=commentary, cvl=cvl, skipped=skipped):
            lines.append("")
            lines.append("--- Commentary ---")
            lines.append(commentary)
            lines.append("")
            lines.append("--- CVL ---")
            lines.append(cvl)
            if skipped:
                lines.append("")
                lines.append(f"--- Skipped ({len(skipped)}) ---")
                for s in skipped:
                    lines.append(f"  Property {s.property_index}: {s.reason}")

        case Application(application_type=app_type, components=comps):
            lines.append(f"Application type: {app_type}")
            lines.append(f"Components ({len(comps)}):")
            for c in comps:
                if isinstance(c, ExternalActor):
                    lines.append(f"## External Actor: {c.name}")
                    lines.append(f"    {c.description}")
                else:
                    lines.append(f"## Contract: {c.name}")
                    lines.append("Contract components:")
                    for cc in c.components:
                        lines.append(f"- {cc.name}: {cc.description}")

        case InterfaceResult():
            lines.append("")
            for (nm, decl) in val.name_to_interface.items():
                lines.append(f"--- Interface {nm}---")
                lines.append(decl.content)

        case StubDeclaration():
            lines.append("")
            lines.append(f"--- Stub {val.solidity_identifier} ---")
            lines.append(val.content)

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
            lines.append("--- Last attempt CVL ---")
            lines.append(cvl)

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cache & Memory Explorer for NatSpec pipeline"
    )
    parser.add_argument("input_file", help="Path to the design document (text or PDF)")
    parser.add_argument("--cache-ns", required=True, dest="cache_ns",
                        help="Cache namespace (same as passed to tui_pipeline.py)")
    parser.add_argument("--memory-ns", dest="memory_ns", default=None,
                        help="Memory namespace (enables memory browsing)")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    content = get_system_doc(input_path)
    if content is None:
        print(f"Error: cannot read {input_path}")
        return 1

    from composer.workflow.services import get_store
    store = get_store()

    root_ns = (args.cache_ns, string_hash(str(content)))
    print(f"Root namespace: {root_ns}")

    root_ctx: WorkflowContext = WorkflowContext.create(
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
        build_tree=lambda: build_tree(root_ctx),
        format_value=format_value,
        store=store,
        status=status,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

_repo_root = str(Path(__file__).parent.parent.absolute())
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from composer.ui.cache_explorer import CacheNode, CacheExplorerApp, DummyServices, CacheTreeNode, OrgNode
from composer.spec.context import (
    WorkflowContext, CacheKey, CVLGeneration, get_document_input,
    Contract, CacheTypes, Marker, ComponentGroup, Properties,
)
from composer.spec.system_model import (
    Application, FromSourceApplication, NatspecApplication,
    ExplicitContract, ExternalActor, FromSourceContract, ExistingFromSource,
)
from composer.spec.natspec.models import (
    InterfaceResult,
    LocatedInterfaceDecl, AutoInterfaceDecl,
    LocatedStubDeclaration, AutoStubDeclaration,
    StubDeclarationModel,
)
from composer.spec.bug import _BugAnalysisCache, _AgentHistory, bug_analysis_key, AGENT_RESULT_KEY
from composer.spec.cvl_generation import (
    _LastAttemptCache, CVL_JUDGE_KEY, LAST_ATTEMPT_KEY,
)
from composer.spec.natspec.pipeline import (
    PROPERTIES_KEY, _component_cache_key, _batch_cache_key,
)
from composer.spec.natspec.author import AuthorResult, GenerationSuccess, GaveUp
from composer.spec.util import string_hash


# ---------------------------------------------------------------------------
# NatSpec cache value type
# ---------------------------------------------------------------------------

type NatSpecCachedValue = (
    NatspecApplication
    | InterfaceResult
    | StubDeclarationModel
    | _BugAnalysisCache
    | _AgentHistory
    | AuthorResult
    | _LastAttemptCache
)

# Cache-key literal used by source_analysis — same across mental models.
SOURCE_ANALYSIS_KEY = CacheKey[None, NatspecApplication]("source-analysis")

# Both interface-decl variants may have been used; the cache key encodes the
# decl subtype name, so we try each.
_INTERFACE_DECL_TYPES: tuple[type, ...] = (LocatedInterfaceDecl, AutoInterfaceDecl)
_STUB_DECL_TYPES: tuple[type, ...] = (LocatedStubDeclaration, AutoStubDeclaration)


async def _cache_get_first[T](ctx: WorkflowContext, key: CacheKey, types: tuple[type[T], ...]) -> T | None:
    """Try each candidate value type against ``key`` under ``ctx``; return the
    first hit. Used when a cache key was parameterized on the concrete type
    name at write time (interface / stub)."""
    child = ctx.child(key)
    for ty in types:
        v = await child.cache_get(ty)
        if v is not None:
            return v
    return None


def _is_generated(c: ExplicitContract) -> bool:
    """The pipeline allocates per-contract cache subtrees only for contracts
    that are generated — every contract in greenfield; only FreshFromSource
    in from-source (``unchanged``/``edited`` are excluded)."""
    return not isinstance(c, ExistingFromSource)


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
async def node_for[T: CacheTypes, S: CacheTypes](
    ctx: WorkflowContext[T],
    child: CacheKey[T, S],
    label: str,
    ty: type[S] | None = None,
):
    child_ctx = ctx.child(child)
    value: S | None = None
    if ty is not None:
        value = await child_ctx.cache_get(ty)
    new_node = CacheNode(label=label, ctx=child_ctx, value=value)
    with node(new_node):  # type: ignore
        yield child_ctx


async def leaf[T: CacheTypes, S: NatSpecCachedValue](
    ctx: WorkflowContext[T],
    child: CacheKey[T, S],
    label: str,
    ty: type[S],
) -> CacheNode[S]:
    child_ctx = ctx.child(child)
    value: S | None = await child_ctx.cache_get(ty)
    return CacheNode[S](label=label, value=value, ctx=child_ctx)


def memory[T: CacheTypes, S: Marker](ctx: WorkflowContext[T], child: CacheKey[T, S], label: str):
    return CacheNode[S](label=label, value=None, ctx=ctx.child(child))


async def build_cvl_generation_node(ctx: WorkflowContext[CVLGeneration]):
    """Children of the per-batch CVL-generation subtree.

    The batch root stores an ``AuthorResult`` (via ``_batch_cache_key``); the
    judge and last-attempt children live under that same context, abstracted
    as ``CVLGeneration``.
    """
    yield memory(ctx, CVL_JUDGE_KEY, "Feedback judge")
    yield (await leaf(ctx, LAST_ATTEMPT_KEY, "Last attempt", _LastAttemptCache))


async def build_component_tree(
    contract_ctx: WorkflowContext[Properties],
    key: CacheKey[Properties, ComponentGroup],
    comp,
):
    async with node_for(contract_ctx, key, comp.name) as feat_ctx:
        # Bug analysis cache (with nested agent history).
        bug_key = bug_analysis_key(None)
        async with node_for(feat_ctx, bug_key, "Bug Analysis", _BugAnalysisCache) as bug_ctx:
            yield (await leaf(bug_ctx, AGENT_RESULT_KEY, "Agent history", _AgentHistory))

        # If bug analysis is cached, its items drive the per-batch cache key.
        bug_cache = await feat_ctx.child(bug_key).cache_get(_BugAnalysisCache)
        if bug_cache is None:
            return
        async with node_for(
            feat_ctx,
            _batch_cache_key(bug_cache.items),
            "CVL Generation",
            AuthorResult,
        ) as cvl_ctx:
            async for t in build_cvl_generation_node(cvl_ctx.abstract(CVLGeneration)):
                yield t


async def build_contract_tree(
    contract_ctx: WorkflowContext[Contract],
    contract: ExplicitContract,
    summ: NatspecApplication,
):
    async with node_for(contract_ctx, PROPERTIES_KEY, "properties") as prop_ctx:
        for comp in contract.components:
            comp_key = _component_cache_key(comp, summ.application_type)
            async for t in build_component_tree(prop_ctx, comp_key, comp):
                yield t


async def build_tree_inner(root_ctx: WorkflowContext[None]):
    # Source analysis: try Application first (greenfield), then
    # FromSourceApplication (update). Whichever hits is the run's flavour.
    sa_ctx = root_ctx.child(SOURCE_ANALYSIS_KEY)
    summary: NatspecApplication | None = await sa_ctx.cache_get(Application)
    if summary is None:
        summary = await sa_ctx.cache_get(FromSourceApplication)
    yield CacheNode(label="source-analysis", ctx=sa_ctx, value=summary)

    if summary is None:
        return

    # Interfaces: the cache key suffix encodes the decl subtype, so try both.
    cached_intf: InterfaceResult | None = None
    intf_ctx = None
    for decl_ty in _INTERFACE_DECL_TYPES:
        intf_key = CacheKey[None, InterfaceResult](
            f"interface-{string_hash(summary.model_dump_json())}-{decl_ty.__name__}"
        )
        probe_ctx = root_ctx.child(intf_key)
        cached = await probe_ctx.cache_get(InterfaceResult)
        if cached is not None:
            cached_intf = cached
            intf_ctx = probe_ctx
            break
    if intf_ctx is None:
        # No interface cache — still show a placeholder so the user can see.
        intf_key = CacheKey[None, InterfaceResult](
            f"interface-{string_hash(summary.model_dump_json())}-<no-cache-hit>"
        )
        intf_ctx = root_ctx.child(intf_key)
    yield CacheNode(label="interface", ctx=intf_ctx, value=cached_intf)

    # Stubs — per-contract, also keyed on decl subtype suffix.
    if cached_intf is not None:
        with section("Stubs"):
            intf_hash = string_hash(cached_intf.model_dump_json())
            for c in summary.contract_components:
                if not _is_generated(c):
                    continue
                found = False
                for decl_ty in _STUB_DECL_TYPES:
                    key = CacheKey[None, StubDeclarationModel](
                        f"stub-for-{intf_hash}-{c.name}-{decl_ty.__name__}"
                    )
                    child_ctx = root_ctx.child(key)
                    value = await child_ctx.cache_get(decl_ty)
                    if value is not None:
                        yield CacheNode[StubDeclarationModel](
                            label=f"Stub: {c.name}", value=value, ctx=child_ctx,
                        )
                        found = True
                        break
                if not found:
                    # Show a miss so the user can see the contract exists.
                    key = CacheKey[None, StubDeclarationModel](
                        f"stub-for-{intf_hash}-{c.name}-<no-cache-hit>"
                    )
                    yield CacheNode[StubDeclarationModel](
                        label=f"Stub: {c.name}",
                        value=None,
                        ctx=root_ctx.child(key),
                    )

    # Per-contract Contract subtree (properties / components / batches).
    for c in summary.contract_components:
        if not _is_generated(c):
            continue
        contract_key = CacheKey[None, Contract](string_hash(c.model_dump_json()))
        async with node_for(root_ctx, contract_key, f"Contract: {c.name}") as contract_ctx:
            async for t in build_contract_tree(contract_ctx, c, summary):
                yield t


async def build_tree(root_ctx: WorkflowContext) -> CacheNode[NatSpecCachedValue]:
    """Build the NatSpec pipeline cache tree by reading the store."""
    root: CacheNode[NatSpecCachedValue] = CacheNode(label="root", ctx=root_ctx)
    with node(root):
        async for n in build_tree_inner(root_ctx):
            curr_node = _node_context.get()
            assert curr_node is not None
            curr_node.children.append(n)  # type: ignore
    return root


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_value(val: NatSpecCachedValue) -> list[str]:
    """Format a NatSpec cached value for the detail pane."""
    lines: list[str] = []

    match val:
        case AuthorResult(result_wrapped=inner):
            match inner:
                case GenerationSuccess(commentary=commentary, spec=spec, skipped=skipped, suggested_path=suggested_path):
                    lines.append(f"Suggested path: {suggested_path}")
                    lines.append("")
                    lines.append("--- Commentary ---")
                    lines.append(commentary)
                    lines.append("")
                    lines.append("--- CVL ---")
                    lines.append(spec)
                    if skipped:
                        lines.append("")
                        lines.append(f"--- Skipped ({len(skipped)}) ---")
                        for s in skipped:
                            lines.append(f"  Property {s.property_index}: {s.reason}")
                case GaveUp(reason=reason):
                    lines.append("--- Gave up ---")
                    lines.append(reason)

        case Application(application_type=app_type, components=comps):
            lines.append(f"Application type: {app_type}")
            lines.append(f"Components ({len(comps)}):")
            for c in comps:
                if isinstance(c, ExternalActor):
                    lines.append(f"## External Actor: {c.name}")
                    lines.append(f"    {c.description}")
                else:
                    lines.append(f"## Contract: {c.name}")
                    for cc in c.components:
                        lines.append(f"- {cc.name}: {cc.description}")

        case FromSourceApplication(application_type=app_type, components=comps):
            lines.append(f"Application type: {app_type}  (from-source mode)")
            lines.append(f"Components ({len(comps)}):")
            for c in comps:
                if isinstance(c, ExternalActor):
                    lines.append(f"## External Actor: {c.name}")
                    lines.append(f"    {c.description}")
                    continue
                if isinstance(c, FromSourceContract):
                    tag = getattr(c, "tag", "?")
                    path = getattr(c, "path", None)
                    head = f"## Contract: {c.name}  [tag: {tag}]"
                    if path:
                        head += f"  @ {path}"
                    lines.append(head)
                    for cc in c.components:
                        lines.append(f"- {cc.name}: {cc.description}")

        case InterfaceResult():
            lines.append("")
            for (nm, decl) in val.name_to_interface.items():
                lines.append(f"--- Interface {nm} (path: {decl.path}) ---")
                lines.append(decl.content)

        case StubDeclarationModel():
            lines.append("")
            head = f"--- Stub {val.solidity_identifier}"
            path = getattr(val, "path", None)
            if path:
                head += f" (path: {path})"
            head += " ---"
            lines.append(head)
            lines.append(val.content)

        case _AgentHistory(items=items, agent_conversation=history):
            lines.append(f"Properties ({len(items)}):")
            for p in items:
                lines.append(f"  - [{p.sort}] {p.description}")
            lines.append("")
            lines.append(f"Agent history: {len(history)} message(s)")

        case _BugAnalysisCache(items=items):
            lines.append(f"Properties ({len(items)}):")
            for p in items:
                lines.append(f"  - [{p.sort}] {p.description}")

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
                        help="Cache namespace (same as passed to tui_pipeline)")
    parser.add_argument("--memory-ns", dest="memory_ns", default=None,
                        help="Memory namespace (enables memory browsing)")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    content = get_document_input(input_path)
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

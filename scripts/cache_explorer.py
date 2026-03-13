"""
Cache & Memory Explorer for the NatSpec pipeline.

Usage:
    python scripts/cache_explorer.py <input_file> --cache-ns <ns> [--memory-ns <ns>]
"""

import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).parent.parent.absolute())
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from composer.io.cache_explorer import CacheNode, CacheExplorerApp, DummyServices
from composer.spec.context import WorkflowContext, CacheKey, CVLGeneration, get_system_doc
from composer.spec.component import ApplicationSummary, SOURCE_ANALYSIS_KEY
from composer.spec.interface_gen import _CachedInterface
from composer.spec.stub_gen import _CachedStub
from composer.spec.bug import _BugAnalysisCache, BUG_ANALYSIS_KEY
from composer.spec.cvl_generation import GeneratedCVL, _LastAttemptCache, CVL_JUDGE_KEY, LAST_ATTEMPT_KEY, FEEDBACK_KEY
from composer.spec.pipeline import PROPERTIES_KEY, _component_cache_key, _batch_cache_key
from composer.spec.util import string_hash


# ---------------------------------------------------------------------------
# NatSpec cache value type
# ---------------------------------------------------------------------------

type NatSpecCachedValue = (
    ApplicationSummary | _CachedInterface | _CachedStub
    | _BugAnalysisCache | GeneratedCVL | _LastAttemptCache
)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree(root_ctx: WorkflowContext) -> CacheNode[NatSpecCachedValue]:
    """Build the NatSpec pipeline cache tree by reading the store."""

    root: CacheNode[NatSpecCachedValue] = CacheNode(label="root", ctx=root_ctx)

    # --- source-analysis -> ApplicationSummary ---
    sa_ctx = root_ctx.child(SOURCE_ANALYSIS_KEY)
    summary = sa_ctx.cache_get(ApplicationSummary)
    root.children.append(CacheNode(
        label="source-analysis", ctx=sa_ctx, value=summary,
    ))

    # --- interface (key depends on summary hash) ---
    cached_intf: _CachedInterface | None = None
    if summary is not None:
        intf_key = CacheKey[None, _CachedInterface](
            f"interface-{string_hash(summary.model_dump_json())}"
        )
        intf_ctx = root_ctx.child(intf_key)
        cached_intf = intf_ctx.cache_get(_CachedInterface)
        root.children.append(CacheNode(
            label="interface", ctx=intf_ctx, value=cached_intf,
        ))

        # --- stub (key depends on interface hash) ---
        if cached_intf is not None:
            stub_key = CacheKey[None, _CachedStub](
                f"stub-for-{string_hash(cached_intf.intf)}"
            )
            stub_ctx = root_ctx.child(stub_key)
            root.children.append(CacheNode(
                label="stub", ctx=stub_ctx, value=stub_ctx.cache_get(_CachedStub),
            ))

    # --- properties (parent for component analysis children) ---
    props_ctx = root_ctx.child(PROPERTIES_KEY)
    props_node: CacheNode[NatSpecCachedValue] = CacheNode(label="properties", ctx=props_ctx)
    root.children.append(props_node)

    if summary is not None:
        for comp in summary.components:
            comp_ctx = props_ctx.child(
                _component_cache_key(comp, summary.application_type)
            )

            comp_node: CacheNode[NatSpecCachedValue] = CacheNode(label=comp.name, ctx=comp_ctx)
            props_node.children.append(comp_node)

            # bug_analysis under component
            ba_ctx = comp_ctx.child(BUG_ANALYSIS_KEY)
            bug_analysis = ba_ctx.cache_get(_BugAnalysisCache)
            comp_node.children.append(CacheNode(
                label="bug_analysis", ctx=ba_ctx, value=bug_analysis,
            ))

            if bug_analysis is not None:
                batch_ctx = comp_ctx.child(
                    _batch_cache_key(bug_analysis.items)
                )
                batch_node: CacheNode[NatSpecCachedValue] = CacheNode(
                    label=f"batch ({len(bug_analysis.items)} properties)",
                    ctx=batch_ctx,
                    value=batch_ctx.cache_get(GeneratedCVL),
                )
                comp_node.children.append(batch_node)

                cvl_ctx = batch_ctx.abstract(CVLGeneration)
                judge_ctx = cvl_ctx.child(CVL_JUDGE_KEY)
                judge_node: CacheNode[NatSpecCachedValue] = CacheNode(
                    label="judge", ctx=judge_ctx,
                )
                judge_node.children.append(CacheNode(
                    label="feedback", ctx=judge_ctx.child(FEEDBACK_KEY),
                ))
                batch_node.children.append(judge_node)
                la_ctx = cvl_ctx.child(LAST_ATTEMPT_KEY)
                batch_node.children.append(CacheNode(
                    label="last_attempt", ctx=la_ctx,
                    value=la_ctx.cache_get(_LastAttemptCache),
                ))

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

        case ApplicationSummary(application_type=app_type, components=comps):
            lines.append(f"Application type: {app_type}")
            lines.append(f"Components ({len(comps)}):")
            for c in comps:
                lines.append(f"  - {c.name}")
                lines.append(f"    {c.description}")

        case _CachedInterface(intf=intf):
            lines.append("")
            lines.append("--- Interface ---")
            lines.append(intf)

        case _CachedStub(stub=stub):
            lines.append("")
            lines.append("--- Stub ---")
            lines.append(stub)

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

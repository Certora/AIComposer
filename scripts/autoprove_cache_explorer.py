"""
Cache & Memory Explorer for the Auto-Prove pipeline.

Usage:
    python scripts/autoprove_cache_explorer.py <project_root> <main_contract> <system_doc> --cache-ns <ns> [--memory-ns <ns>]
"""

import argparse
import hashlib
import pathlib
import sys

_repo_root = str(pathlib.Path(__file__).parent.parent.absolute())
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from composer.io.cache_explorer import CacheNode, CacheExplorerApp, DummyServices
from composer.spec.context import (
    WorkflowContext, CacheKey, CVLGeneration, Properties, ComponentGroup, get_system_doc,
)
from composer.spec.component import ApplicationSummary, SOURCE_ANALYSIS_KEY
from composer.spec.harness import Configuration, HarnessSetup, SETUP_KEY, HARNESS_KEY
from composer.spec.struct_invariant import Invariants, STRUCTURAL_INV_KEY, INV_JUDGE_KEY
from composer.spec.bug import _BugAnalysisCache, BUG_ANALYSIS_KEY
from composer.spec.summarizer import _SummaryCache, _summary_key
from composer.spec.cvl_generation import (
    GeneratedCVL, _LastAttemptCache, CVL_JUDGE_KEY, LAST_ATTEMPT_KEY, FEEDBACK_KEY,
)
from composer.spec.autoprove_pipeline import (
    PROPERTIES_KEY, INV_CVL_KEY, _component_cache_key, _batch_cache_key,
)
from composer.spec.util import string_hash


# ---------------------------------------------------------------------------
# Cache value type
# ---------------------------------------------------------------------------

type AutoProveCachedValue = (
    Configuration | HarnessSetup | _SummaryCache
    | Invariants | ApplicationSummary
    | _BugAnalysisCache | GeneratedCVL | _LastAttemptCache
)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree(root_ctx: WorkflowContext) -> CacheNode[AutoProveCachedValue]:
    """Build the auto-prove pipeline cache tree by reading the store."""

    root: CacheNode[AutoProveCachedValue] = CacheNode(label="root", ctx=root_ctx)

    # --- Phase 1: setup → Configuration ---
    setup_ctx = root_ctx.child(SETUP_KEY)
    config = setup_ctx.cache_get(Configuration)
    setup_node: CacheNode[AutoProveCachedValue] = CacheNode(
        label="setup", ctx=setup_ctx, value=config,
    )
    root.children.append(setup_node)

    # --- Phase 1: harnessing → HarnessSetup (child of setup) ---
    if config is not None:
        harness_ctx = setup_ctx.child(HARNESS_KEY)
        setup_node.children.append(CacheNode(
            label="harnessing", ctx=harness_ctx,
            value=harness_ctx.cache_get(HarnessSetup),
        ))

        # --- Phase 2: summary (key depends on config hash) ---
        summary_key = _summary_key(config)
        summary_ctx = root_ctx.child(summary_key)
        root.children.append(CacheNode(
            label="summary", ctx=summary_ctx,
            value=summary_ctx.cache_get(_SummaryCache),
        ))

    # --- Phase 3: structural invariants ---
    inv_ctx = root_ctx.child(STRUCTURAL_INV_KEY)
    invariants = inv_ctx.cache_get(Invariants)
    inv_node: CacheNode[AutoProveCachedValue] = CacheNode(
        label="structural-invariants", ctx=inv_ctx, value=invariants,
    )
    root.children.append(inv_node)

    # invariant judge (child of structural-inv)
    judge_ctx = inv_ctx.child(INV_JUDGE_KEY)
    inv_node.children.append(CacheNode(label="judge", ctx=judge_ctx))

    # --- Phase 3: invariant CVL generation ---
    inv_cvl_ctx = root_ctx.child(INV_CVL_KEY)
    inv_cvl = inv_cvl_ctx.cache_get(GeneratedCVL)
    inv_cvl_node: CacheNode[AutoProveCachedValue] = CacheNode(
        label="invariant-cvl", ctx=inv_cvl_ctx, value=inv_cvl,
    )
    root.children.append(inv_cvl_node)

    # CVL generation sub-tree for invariant CVL
    _add_cvl_gen_children(inv_cvl_node, inv_cvl_ctx)

    # --- Phase 4: component analysis ---
    sa_ctx = root_ctx.child(SOURCE_ANALYSIS_KEY)
    summary = sa_ctx.cache_get(ApplicationSummary)
    root.children.append(CacheNode(
        label="component-analysis", ctx=sa_ctx, value=summary,
    ))

    # --- Phases 5+6: properties → per-component ---
    props_ctx = root_ctx.child(PROPERTIES_KEY)
    props_node: CacheNode[AutoProveCachedValue] = CacheNode(label="properties", ctx=props_ctx)
    root.children.append(props_node)

    if summary is not None:
        for comp in summary.components:
            comp_ctx = props_ctx.child(
                _component_cache_key(comp, summary.application_type)
            )
            comp_node: CacheNode[AutoProveCachedValue] = CacheNode(
                label=comp.name, ctx=comp_ctx,
            )
            props_node.children.append(comp_node)

            # bug analysis under component
            ba_ctx = comp_ctx.child(BUG_ANALYSIS_KEY)
            bug_analysis = ba_ctx.cache_get(_BugAnalysisCache)
            comp_node.children.append(CacheNode(
                label="bug_analysis", ctx=ba_ctx, value=bug_analysis,
            ))

            # batch CVL generation
            if bug_analysis is not None:
                batch_ctx = comp_ctx.child(
                    _batch_cache_key(bug_analysis.items)
                )
                batch_node: CacheNode[AutoProveCachedValue] = CacheNode(
                    label=f"batch ({len(bug_analysis.items)} properties)",
                    ctx=batch_ctx,
                    value=batch_ctx.cache_get(GeneratedCVL),
                )
                comp_node.children.append(batch_node)

                _add_cvl_gen_children(batch_node, batch_ctx)

    return root


def _add_cvl_gen_children(
    parent: CacheNode[AutoProveCachedValue],
    batch_ctx: WorkflowContext,
) -> None:
    """Add judge/feedback/last_attempt children for a CVL generation context."""
    cvl_ctx = batch_ctx.abstract(CVLGeneration)
    judge_ctx = cvl_ctx.child(CVL_JUDGE_KEY)
    judge_node: CacheNode[AutoProveCachedValue] = CacheNode(
        label="judge", ctx=judge_ctx,
    )
    judge_node.children.append(CacheNode(
        label="feedback", ctx=judge_ctx.child(FEEDBACK_KEY),
    ))
    parent.children.append(judge_node)

    la_ctx = cvl_ctx.child(LAST_ATTEMPT_KEY)
    parent.children.append(CacheNode(
        label="last_attempt", ctx=la_ctx,
        value=la_ctx.cache_get(_LastAttemptCache),
    ))


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def format_value(val: AutoProveCachedValue) -> list[str]:
    """Format a cached value for the detail pane."""
    lines: list[str] = []

    match val:
        case Configuration() as cfg:
            lines.append(f"Primary entity: {cfg.primary_entity}")
            lines.append(f"Non-trivial state: {cfg.non_trivial_state}")
            lines.append(f"Summaries path: {cfg.summaries_path}")
            lines.append(f"External contracts ({len(cfg.external_contracts)}):")
            for ec in cfg.external_contracts:
                lines.append(f"  - [{ec.l}] {ec.name}")
            lines.append("")
            lines.append("--- Config ---")
            import json
            lines.append(json.dumps(cfg.config, indent=2))

        case HarnessSetup(setup=setup, vfs=vfs):
            lines.append(f"Primary entity: {setup.primary_entity}")
            lines.append(f"Non-trivial state: {setup.non_trivial_state}")
            lines.append(f"External contracts ({len(setup.external_contracts)}):")
            for ec in setup.external_contracts:
                lines.append(f"  - [{ec.l}] {ec.name}")
            lines.append("")
            lines.append(f"VFS files ({len(vfs)}):")
            for path in sorted(vfs):
                lines.append(f"  {path} ({len(vfs[path])} chars)")

        case _SummaryCache(content=content):
            lines.append("--- Summary CVL ---")
            lines.append(content)

        case Invariants(inv=invs):
            lines.append(f"Invariants ({len(invs)}):")
            for inv in invs:
                lines.append(f"  - {inv.name}: {inv.description}")

        case ApplicationSummary(application_type=app_type, components=comps):
            lines.append(f"Application type: {app_type}")
            lines.append(f"Components ({len(comps)}):")
            for c in comps:
                lines.append(f"  - {c.name}")
                lines.append(f"    {c.description}")

        case _BugAnalysisCache(items=items):
            lines.append(f"Properties ({len(items)}):")
            for p in items:
                methods = p.methods
                lines.append(f"  - [{p.sort}] {p.description}")
                if isinstance(methods, list):
                    lines.append(f"    methods: {', '.join(methods)}")
                else:
                    lines.append(f"    methods: {methods}")

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

        case _LastAttemptCache(cvl=cvl):
            lines.append("--- Last attempt CVL ---")
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

    # Parse main_contract
    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)
    full_contract_path = pathlib.Path(main_contract_path).resolve()
    relative_path = str(full_contract_path.relative_to(project_root))

    # Read system doc
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

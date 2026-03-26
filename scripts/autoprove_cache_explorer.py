"""
Cache & Memory Explorer for the Auto-Prove pipeline.

Usage:
    python scripts/autoprove_cache_explorer.py <project_root> <main_contract> <system_doc> --cache-ns <ns> [--memory-ns <ns>]
"""

import argparse
import hashlib
import pathlib
import sys
from contextlib import contextmanager
from contextvars import ContextVar

_repo_root = str(pathlib.Path(__file__).parent.parent.absolute())
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from composer.ui.cache_explorer import CacheNode, OrgNode, CacheTreeNode, CacheExplorerApp, DummyServices
from composer.spec.context import WorkflowContext, CacheKey, CacheTypes, get_system_doc
from composer.spec.source.system_analysis import SOURCE_ANALYSIS_KEY
from composer.spec.source.harness import (
    system_setup_key, HARNESS_ANALYSIS_KEY,
    SystemDescriptionHarnessed, AgentSystemDescription,
)
from composer.spec.system_model import SourceApplication, SourceExplicitContract, SourceExternalActor


# ---------------------------------------------------------------------------
# Cache value type
# ---------------------------------------------------------------------------

type AutoProveCachedValue = SourceApplication | SystemDescriptionHarnessed | AgentSystemDescription


# ---------------------------------------------------------------------------
# Tree construction helpers (same pattern as scripts/cache_explorer.py)
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


@contextmanager
def node_for[T: CacheTypes, S: CacheTypes](
    ctx: WorkflowContext[T], child: CacheKey[T, S], label: str, ty: type[S] | None = None
):
    child_ctx = ctx.child(child)
    value: S | None = child_ctx.cache_get(ty) if ty is not None else None  # type: ignore[arg-type]
    new_node: CacheNode[AutoProveCachedValue] = CacheNode(label=label, ctx=child_ctx, value=value)  # type: ignore[arg-type]
    with node(new_node):
        yield child_ctx


def leaf[T: CacheTypes, S: AutoProveCachedValue](
    ctx: WorkflowContext[T], child: CacheKey[T, S], label: str, ty: type[S]
) -> CacheNode[AutoProveCachedValue]:
    child_ctx = ctx.child(child)
    value: S | None = child_ctx.cache_get(ty)
    return CacheNode(label=label, ctx=child_ctx, value=value)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree_inner(root_ctx: WorkflowContext):
    sa = leaf(root_ctx, SOURCE_ANALYSIS_KEY, "source-analysis", SourceApplication)
    yield sa

    if sa.value is None:
        return

    with node_for(root_ctx, system_setup_key(sa.value), "setup", SystemDescriptionHarnessed) as setup_ctx:
        yield leaf(setup_ctx, HARNESS_ANALYSIS_KEY, "harness-analysis", AgentSystemDescription)


def build_tree(root_ctx: WorkflowContext) -> CacheNode[AutoProveCachedValue]:
    root: CacheNode[AutoProveCachedValue] = CacheNode(label="root", ctx=root_ctx)
    with node(root):
        for n in build_tree_inner(root_ctx):
            curr = _node_context.get()
            assert curr is not None
            curr.children.append(n)
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
                    lines.append(f"    links → {', '.join(lf.target)}")
            if ext_ifaces:
                lines.append("")
                lines.append(f"External interfaces ({len(ext_ifaces)}):")
                for ei in ext_ifaces:
                    lines.append(f"  {ei.name}: {ei.behavioral_spec}")

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

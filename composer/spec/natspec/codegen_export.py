"""Export a natspec pipeline result as an ``ImplementationPlan`` — the
authoritative handoff artifact from natspec to codegen.

The plan bundles, in one JSON:

* The application-level context the downstream driver needs (app type,
  ``source_root``, ``prover_conf``, system-doc pointer).
* One ``ContractPlan`` per contract that needs codegen, emitted in
  dependency order (leaves first, dependents last). Order is derived by
  topological sort of the contract-interaction graph extracted from the
  application model.
* Each contract's interface source, stub source, specs, the per-contract
  stub fields that were requested during generation, and the list of
  intra-application contracts it depends on.
* Any dependency cycles detected during topological sort, so the codegen
  driver can surface them instead of silently picking an order.

The emitted ``implementation_plan.json`` is self-contained: it carries
the textual content of interfaces, stubs, and specs directly rather than
referencing paths that the consumer would have to resolve. This makes
the handoff robust to layout changes between natspec and codegen and
keeps the artifact trivially relocatable.

The consumer (codegen driver / CLI) iterates ``contracts`` in order,
materializes each contract's artifacts to disk on its side, and invokes
``python main.py --input-json <per-contract json>`` for each.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from composer.spec.natspec.registry import FieldSpec
from composer.spec.system_model import (
    ComponentInteraction,
    ExplicitContract,
    FreshFromSource,
    FromSourceApplication,
)

if TYPE_CHECKING:
    from composer.spec.natspec.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------


@dataclass
class SpecEntry:
    """One spec file produced for a contract."""
    filename: str
    content: str


@dataclass
class ContractPlan:
    """Everything the codegen driver needs to produce one contract."""
    name: str
    # From the from-source workflow; ``None`` in greenfield. When present,
    # indicates the contract is being freshly introduced into an existing
    # source tree.
    tag: Literal["new"] | None
    # Generated interface (path is agent-chosen, workspace-relative in
    # from-source mode and self-contained-relative in greenfield).
    interface_path: str
    interface_source: str
    # Generated stub — the file codegen's implementation should replace.
    stub_path: str
    stub_source: str
    # Fields the CVL authors requested on this contract's stub during
    # generation. Codegen must ensure the final implementation carries these
    # as storage, matching name + type, or the specs will not typecheck.
    required_stub_fields: list[FieldSpec] = field(default_factory=list)
    # Specs produced for this contract.
    specs: list[SpecEntry] = field(default_factory=list)
    # Intra-application contracts this contract interacts with (by name).
    # Drives the topological ordering.
    depends_on: list[str] = field(default_factory=list)


@dataclass
class ExternalActorSpec:
    """Surfaced for the driver's awareness; not consumed per-contract."""
    name: str
    description: str
    path: str | None


@dataclass
class ImplementationPlan:
    """Authoritative natspec → codegen handoff artifact."""
    application_type: str
    application_description: str
    # Workspace-relative reference back to the originating system doc.
    # String path (absolute or relative to wherever the plan is loaded).
    system_doc_path: str
    # The existing source tree, if any. ``None`` in greenfield.
    source_root: str | None
    # Opaque per-task prover-config overrides. Passed through verbatim to
    # every per-contract codegen invocation.
    prover_conf: dict | None
    # Contracts in dependency order. Leaves first; a contract appears after
    # every contract it depends on (within the subset of contracts that
    # were actually generated — external actors and pre-existing
    # ``unchanged``/``edited`` dependencies are not entries in this list).
    contracts: list[ContractPlan] = field(default_factory=list)
    # Dependency cycles detected during topological sort. Each entry is a
    # list of contract names forming one cycle. Empty in the DAG case (the
    # expected case for NEST-shaped apps). Non-empty entries indicate the
    # driver must decide how to break the cycle (e.g., by processing one
    # contract against its neighbours' stubs and backfilling later).
    cycles: list[list[str]] = field(default_factory=list)
    # External actors the application interacts with — surfaced here as
    # context for the codegen driver's mocking strategies. Not per-contract
    # because multiple contracts may interact with the same actor.
    external_actors: list[ExternalActorSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dependency graph construction + topological sort
# ---------------------------------------------------------------------------


def _build_dep_graph(
    result: "PipelineResult",
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Construct ``(forward, reverse)`` adjacency maps restricted to the
    contracts present in ``result.contracts`` (the ones that need codegen).

    ``forward[X]`` is the set of generated contracts X depends on.
    ``reverse[X]`` is the set of generated contracts that depend on X.

    Interactions with pre-existing ``unchanged``/``edited`` contracts and
    external actors are intentionally dropped — those aren't nodes in
    *this* plan, they're fixed dependencies on the driver's side.
    """
    generated_names = {c.name for c in result.contracts}

    # Build a name -> ExplicitContract lookup from the application model so
    # we can read each generated contract's interactions back out.
    by_name = {
        c.name: c for c in result.app.contract_components
    }

    forward: dict[str, set[str]] = {name: set() for name in generated_names}
    reverse: dict[str, set[str]] = {name: set() for name in generated_names}

    for name in generated_names:
        c = by_name.get(name)
        if c is None:
            continue
        for comp in c.components:
            for inter in comp.interactions:
                if not isinstance(inter, ComponentInteraction):
                    continue
                target = inter.contract_name
                if target == name:
                    continue  # self-interaction, ignore
                if target not in generated_names:
                    continue  # pre-existing or external, not in plan
                forward[name].add(target)
                reverse[target].add(name)

    return forward, reverse


def _topo_sort(
    names: list[str],
    forward: dict[str, set[str]],
    reverse: dict[str, set[str]],
) -> tuple[list[str], list[list[str]]]:
    """Kahn's algorithm with cycle extraction.

    Returns ``(ordered, cycles)``.
    - ``ordered`` is every name that could be drained (dependencies first).
    - ``cycles`` contains the strongly-connected remnant as separate lists
      (one per cycle). Cycle names are NOT included in ``ordered``; the
      caller decides what to do with them.
    """
    indeg = {n: len(forward[n]) for n in names}
    ready = sorted(n for n in names if indeg[n] == 0)
    ordered: list[str] = []
    while ready:
        n = ready.pop(0)
        ordered.append(n)
        for dependent in sorted(reverse[n]):
            indeg[dependent] -= 1
            if indeg[dependent] == 0:
                ready.append(dependent)
        ready.sort()

    remaining = [n for n in names if n not in set(ordered)]
    cycles = _extract_sccs(remaining, forward)
    return ordered, cycles


def _extract_sccs(
    remaining: list[str],
    forward: dict[str, set[str]],
) -> list[list[str]]:
    """Tarjan-lite SCC extraction over the sub-graph induced by ``remaining``.

    Every node in ``remaining`` is part of at least one cycle (since Kahn
    couldn't drain it). We return each strongly-connected component as a
    sorted list of names. A self-loop-free node that merely depends on a
    cycle but isn't in one would have been drained by Kahn, so anything
    still present is cyclic.
    """
    if not remaining:
        return []

    remaining_set = set(remaining)
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in forward.get(v, set()):
            if w not in remaining_set:
                continue
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1 or v in forward.get(v, set()):
                sccs.append(sorted(scc))

    for n in remaining:
        if n not in indices:
            strongconnect(n)

    return sccs


# ---------------------------------------------------------------------------
# Spec + tag extraction
# ---------------------------------------------------------------------------


def _spec_entries_for(formulation) -> list[SpecEntry]:
    """Pull specs out of a ContractFormulation's spec_results."""
    out: list[SpecEntry] = []
    spec_results = getattr(formulation, "spec_results", None)
    if spec_results is None:
        return out
    specs_list = getattr(spec_results, "specs", None)
    if not specs_list:
        return out

    for i, success in enumerate(specs_list):
        content = getattr(success, "spec", None)
        if not content:
            continue
        suggested = getattr(success, "suggested_path", None)
        basename = suggested or f"{formulation.name}_{i}.spec"
        if not basename.endswith(".spec"):
            basename = f"{basename}.spec"
        out.append(SpecEntry(filename=basename, content=content))
    return out


def _tag_for(name: str, app) -> Literal["new"] | None:
    """Return ``'new'`` for FreshFromSource contracts, ``None`` otherwise.

    Greenfield contracts have no tag semantics; only the from-source
    workflow carries ``new``/``edited``/``unchanged`` tags.
    """
    if not isinstance(app, FromSourceApplication):
        return None
    for c in app.contract_components:
        if c.name == name and isinstance(c, FreshFromSource):
            return "new"
    return None


def _external_actors(app) -> list[ExternalActorSpec]:
    out: list[ExternalActorSpec] = []
    for c in app.components:
        if isinstance(c, ExplicitContract):
            continue
        out.append(
            ExternalActorSpec(
                name=c.name,
                description=getattr(c, "description", ""),
                path=getattr(c, "path", None),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_implementation_plan(
    result: "PipelineResult",
    *,
    system_doc_path: pathlib.Path | str,
    source_root: pathlib.Path | str | None = None,
    prover_conf: dict | None = None,
) -> ImplementationPlan:
    """Assemble a self-contained ``ImplementationPlan`` from a
    ``PipelineResult``.

    This is a pure function — it reads the pipeline result (including the
    stub-field snapshot already on ``result.stub_fields``) and returns a
    structured plan. It does not touch the filesystem.

    Args:
        result: Output of ``run_natspec_pipeline``.
        system_doc_path: Pointer back to the originating system document.
            Stored verbatim in the plan; the consumer is responsible for
            resolving it.
        source_root: The existing source tree for the from-source workflow.
            ``None`` in greenfield.
        prover_conf: Optional per-task prover-config overrides, passed
            through verbatim to every per-contract codegen invocation.

    Returns:
        An ``ImplementationPlan`` with ``contracts`` in topological order
        and ``cycles`` populated only if the interaction graph is not a
        DAG.
    """
    forward, reverse = _build_dep_graph(result)
    ordered_names, cycles = _topo_sort(
        [c.name for c in result.contracts], forward, reverse
    )

    # Drop any cycle-participating contracts from the ordered list before
    # materializing ContractPlan entries; they still exist as raw pipeline
    # output but should not be surfaced as orderable codegen tasks until
    # the driver resolves the cycle.
    in_cycle: set[str] = set()
    for cyc in cycles:
        in_cycle.update(cyc)
    ordered_for_plan = [n for n in ordered_names if n not in in_cycle]

    by_name = {c.name: c for c in result.contracts}

    contract_plans: list[ContractPlan] = []
    for name in ordered_for_plan:
        formulation = by_name[name]
        contract_plans.append(
            ContractPlan(
                name=name,
                tag=_tag_for(name, result.app),
                interface_path=formulation.interface.path,
                interface_source=formulation.interface.content,
                stub_path=formulation.stub.path,
                stub_source=formulation.stub.content,
                required_stub_fields=list(result.stub_fields.get(name, [])),
                specs=_spec_entries_for(formulation),
                depends_on=sorted(forward.get(name, set())),
            )
        )

    return ImplementationPlan(
        application_type=result.app.application_type,
        application_description=result.app.description,
        system_doc_path=str(system_doc_path),
        source_root=str(source_root) if source_root is not None else None,
        prover_conf=prover_conf,
        contracts=contract_plans,
        cycles=cycles,
        external_actors=_external_actors(result.app),
    )


def plan_to_json(plan: ImplementationPlan) -> dict:
    """Serialize a plan to a JSON-ready dict.

    Hand-walks the structure rather than using ``dataclasses.asdict`` because
    ``FieldSpec`` is a pydantic model (not a dataclass), and ``asdict``'s
    recursion does not know how to flatten it.
    """
    return {
        "application_type": plan.application_type,
        "application_description": plan.application_description,
        "system_doc_path": plan.system_doc_path,
        "source_root": plan.source_root,
        "prover_conf": plan.prover_conf,
        "contracts": [
            {
                "name": c.name,
                "tag": c.tag,
                "interface_path": c.interface_path,
                "interface_source": c.interface_source,
                "stub_path": c.stub_path,
                "stub_source": c.stub_source,
                "required_stub_fields": [
                    fs.model_dump() for fs in c.required_stub_fields
                ],
                "specs": [
                    {"filename": s.filename, "content": s.content}
                    for s in c.specs
                ],
                "depends_on": list(c.depends_on),
            }
            for c in plan.contracts
        ],
        "cycles": [list(cyc) for cyc in plan.cycles],
        "external_actors": [
            {"name": a.name, "description": a.description, "path": a.path}
            for a in plan.external_actors
        ],
    }


def export_implementation_plan(
    result: "PipelineResult",
    *,
    output_root: pathlib.Path | str,
    system_doc_path: pathlib.Path | str,
    source_root: pathlib.Path | str | None = None,
    prover_conf: dict | None = None,
) -> tuple[ImplementationPlan, pathlib.Path]:
    """Build the plan and write it to ``<output_root>/implementation_plan.json``.

    Returns the plan and the absolute path it was written to.
    """
    out = pathlib.Path(output_root).absolute()
    out.mkdir(parents=True, exist_ok=True)

    plan = build_implementation_plan(
        result,
        system_doc_path=system_doc_path,
        source_root=source_root,
        prover_conf=prover_conf,
    )

    json_path = out / "implementation_plan.json"
    json_path.write_text(json.dumps(plan_to_json(plan), indent=2, default=str))
    return plan, json_path

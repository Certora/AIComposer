"""Per-thread persistence for ``cex_remediation``'s full-text spec
proposals.

The remediator returns a *diff* to the codegen author so the spec text
doesn't crowd out the rationale / addendum / Solidity-side instructions
in the agent's attention budget. The full proposed CVL is stored here
under an opaque ``proposal_key``; ``apply_remediation_proposal`` resolves
the key at staging time so the working-spec slot gets the complete text
without the codegen author having to re-emit it.

The codegen author still has ``write_working_spec`` (raw-text path) for
cases where they need to tweak the proposal — e.g. fix a typecheck
error in the proposed CVL. The proposal-key path is the encouraged
default; raw-text is the escape hatch.

Default storage namespace is flat (``("cex_proposals",)``) keyed by
uuid. Stale keys are harmless (uuids don't collide); we don't
aggressively GC.
"""

from dataclasses import dataclass

from langgraph.store.base import BaseStore


_DEFAULT_NAMESPACE: tuple[str, ...] = ("cex_proposals",)


@dataclass(frozen=True)
class ProposalStore:
    """Typed wrapper around a ``BaseStore`` for full-text spec proposals.
    Construct once at workflow setup; pass through ``AIComposerContext``.
    Callers don't reach for ``langgraph.config.get_store`` themselves."""

    store: BaseStore
    namespace: tuple[str, ...] = _DEFAULT_NAMESPACE

    async def record(self, key: str, full_cvl: str) -> None:
        """Persist a full proposed-CVL text under ``key``."""
        await self.store.aput(self.namespace, key, {"full_cvl": full_cvl})

    async def lookup(self, key: str) -> str | None:
        """Return the full proposed-CVL text for ``key`` or ``None``."""
        item = await self.store.aget(self.namespace, key)
        if item is None:
            return None
        return item.value["full_cvl"]

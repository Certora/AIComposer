"""Per-thread persistence for ``AnalyzedDiagnosis`` records produced by the
CEX analyzer.

Reports get written here after the analyzer runs (in the codegen prover
tool wrapper) and read here by ``cex_remediation`` when the codegen
author hands it a ``report_key``. This indirection prevents the codegen
agent from corrupting a diagnosis by paraphrasing it before forwarding
to the remediator: the agent only ever passes the opaque key, and the
remediator looks up the original text itself.

The default storage namespace is flat (``("cex_reports",)``) keyed by
uuid. Reports are not aggressively GC'd — a stale key from a prior
prover run won't collide with a fresh one (uuid), and the storage cost
is small.
"""

from dataclasses import dataclass

from langgraph.store.base import BaseStore

from composer.prover.ptypes import AnalyzedDiagnosis


_DEFAULT_NAMESPACE: tuple[str, ...] = ("cex_reports",)


@dataclass(frozen=True)
class ReportStore:
    """Typed wrapper around a ``BaseStore`` for ``AnalyzedDiagnosis``
    persistence. Construct once at workflow setup with the run's
    ``BaseStore``; pass the wrapper through ``AIComposerContext``.
    Callers don't reach for ``langgraph.config.get_store`` themselves —
    the wrapper is the boundary."""

    store: BaseStore
    namespace: tuple[str, ...] = _DEFAULT_NAMESPACE

    async def record(self, diagnoses: list[AnalyzedDiagnosis]) -> None:
        """Persist each diagnosis under its ``report_key``."""
        for diag in diagnoses:
            await self.store.aput(
                self.namespace,
                diag.report_key,
                diag.model_dump(mode="json"),
            )

    async def lookup(self, report_key: str) -> AnalyzedDiagnosis | None:
        """Return the diagnosis matching ``report_key`` or ``None``."""
        item = await self.store.aget(self.namespace, report_key)
        if item is None:
            return None
        return AnalyzedDiagnosis.model_validate(item.value)

from dataclasses import dataclass, field
import hashlib

from graphcore.graph import BoundLLM
from graphcore.tools.vfs import VFSAccessor

from composer.core.state import AIComposerState
from composer.core.validation import Validation
from composer.rag.db import ComposerRAGDB
from composer.prover.core import CexHandler, CloudConfig
from composer.prover.report_store import ReportStore
from composer.spec.proposal_store import ProposalStore

@dataclass
class ProverOptions:
    capture_output: bool
    keep_folder: bool
    cloud: CloudConfig | None = None

@dataclass
class AIComposerContext:
    """Per-run context for the codegen workflow.

    ``required_validations`` is a flat list of typed validation keys (see
    :class:`composer.core.validation.Validation`). One ``prover_validation``
    entry per registered spec (each spec must be independently verified)
    plus ``REQS_VALIDATION`` if requirements were extracted. The set is
    built in the executor from ``InputData.specs`` + whether reqs were
    extracted.

    ``cex_handler`` is the strategy used to turn prover-returned CEXes into
    diagnoses; constructed by the executor once and reused across every
    prover invocation in this run. Codegen wires the agentic handler;
    other workflows could substitute a trivial fanout.

    ``report_store`` and ``proposal_store`` are typed wrappers around the
    run's ``BaseStore`` for ``AnalyzedDiagnosis`` records and full-text
    spec proposals respectively. Constructed once in the executor; the
    prover tool, ``cex_remediation``, and ``apply_remediation_proposal``
    all reach for them through this context rather than each calling
    ``langgraph.config.get_store`` directly.
    """
    llm: BoundLLM
    rag_db: ComposerRAGDB
    prover_opts: ProverOptions
    vfs_materializer: VFSAccessor[AIComposerState]
    cex_handler: CexHandler
    required_validations: list[Validation] = field(default_factory=list)
    prover_conf_overrides: dict | None = None

def compute_state_digest(state: AIComposerState) -> str:
    # not interested in cryptographic bulletproofing, just need *some* digest
    digester = hashlib.md5()
    for (_, cont) in sorted(state["vfs"].items(), key = lambda x: x[0]):
        digester.update(cont.encode("utf-8"))
    return digester.hexdigest()

from dataclasses import dataclass, field
import hashlib

from graphcore.graph import BoundLLM
from graphcore.tools.vfs import VFSAccessor

from composer.core.state import AIComposerState
from composer.rag.db import ComposerRAGDB
from composer.prover.core import CloudConfig

@dataclass
class ProverOptions:
    capture_output: bool
    keep_folder: bool
    cloud: CloudConfig | None = None

@dataclass
class AIComposerContext:
    """Per-run context for the codegen workflow.

    ``required_validations`` is a flat list of validation keys. A prover-stamp
    entry looks like ``"prover:<spec_vfs_path>"`` (one per registered spec so
    each spec must be independently verified); the natural-language
    requirements entry is ``"natural language requirements"``. The set is
    built in the executor from ``InputData.specs`` + whether reqs were
    extracted.
    """
    llm: BoundLLM
    rag_db: ComposerRAGDB
    prover_opts: ProverOptions
    vfs_materializer: VFSAccessor[AIComposerState]
    required_validations: list[str] = field(default_factory=list)
    prover_conf_overrides: dict | None = None

def compute_state_digest(c: AIComposerContext, state: AIComposerState) -> str:
    # not interested in cryptographic bulletproofing, just need *some* digest
    digester = hashlib.md5()
    for (_, cont) in sorted(c.vfs_materializer.iterate(state), key = lambda x: x[0]):
        digester.update(cont)
    return digester.hexdigest()

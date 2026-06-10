from dataclasses import dataclass, field
import hashlib

from graphcore.graph import BoundLLM
from graphcore.tools.vfs import VFSAccessor

from composer.core.state import AIComposerState
from composer.rag.db import ComposerRAGDB
from composer.core.validation import ValidationType, prover
from composer.prover.core import DEFAULT_GLOBAL_TIMEOUT

@dataclass
class ProverOptions:
    capture_output: bool
    keep_folder: bool
    extra_args: list[str] = field(default_factory=list)
    #: Explicit cloud flag (from --cloud / --local_prover), not inferred from a
    #: --server flag; the cloud server is env-derived at launch time.
    cloud: bool = False

    @property
    def global_timeout(self) -> float:
        if "--global_timeout" not in self.extra_args:
            return DEFAULT_GLOBAL_TIMEOUT
        idx = self.extra_args.index("--global_timeout")
        return float(self.extra_args[idx + 1])

@dataclass
class AIComposerContext:
    llm: BoundLLM
    rag_db: ComposerRAGDB
    prover_opts: ProverOptions
    vfs_materializer: VFSAccessor[AIComposerState]
    required_validations: list[ValidationType] = field(default_factory=lambda: [prover])

def compute_state_digest(c: AIComposerContext, state: AIComposerState) -> str:
    # not interested in cryptographic bulletproofing, just need *some* digest
    digester = hashlib.md5()
    for (_, cont) in sorted(c.vfs_materializer.iterate(state), key = lambda x: x[0]):
        digester.update(cont)
    return digester.hexdigest()

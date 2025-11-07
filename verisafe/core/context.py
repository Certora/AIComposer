from dataclasses import dataclass, field
import hashlib

from graphcore.graph import BoundLLM
from graphcore.tools.vfs import VFSAccessor

from verisafe.core.state import CryptoStateGen
from verisafe.rag.db import PostgreSQLRAGDatabase
from verisafe.core.validation import ValidationType, prover

@dataclass
class ProverOptions:
    capture_output: bool
    keep_folder: bool

@dataclass
class CryptoContext:
    llm: BoundLLM
    rag_db: PostgreSQLRAGDatabase
    prover_opts: ProverOptions
    vfs_materializer: VFSAccessor[CryptoStateGen]
    required_validations: list[ValidationType] = field(default_factory=lambda: [prover])

def compute_state_digest(c: CryptoContext, state: CryptoStateGen) -> str:
    # not interested in cryptographic bulletproofing, just need *some* digest
    digester = hashlib.md5()
    for (_, cont) in sorted(c.vfs_materializer.iterate(state), key = lambda x: x[0]):
        digester.update(cont)
    return digester.hexdigest()

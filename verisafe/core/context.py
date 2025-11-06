from dataclasses import dataclass, field
import hashlib

from graphcore.graph import BoundLLM
from graphcore.tools.vfs import VFSAccessor

from verisafe.core.state import CryptoStateGen
from verisafe.rag.db import PostgreSQLRAGDatabase
from verisafe.core.validation import prover

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
    required_validations: list[str] = field(default_factory=lambda: [prover])

def compute_state_digest(c: CryptoContext, state: CryptoStateGen) -> str:
    files = {}
    file_names = []
    for (p, cont) in c.vfs_materializer.iterate(state):
        files[p] = cont
        file_names.append(p)
    file_names.sort()
    # not interested in cryptographic bulletproofing, just need *some* digest
    digester = hashlib.md5()
    for nm in file_names:
        digester.update(files[nm])
    return digester.hexdigest()

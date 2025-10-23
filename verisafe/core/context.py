from dataclasses import dataclass

from graphcore.graph import BoundLLM
from graphcore.tools.vfs import VFSAccessor

from verisafe.core.state import CryptoStateGen
from verisafe.rag.db import PostgreSQLRAGDatabase


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
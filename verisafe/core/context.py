from dataclasses import dataclass
from graphcore.graph import BoundLLM
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

from dataclasses import dataclass, field
from typing import Optional
from graphcore.graph import BoundLLM
from verisafe.rag.db import PostgreSQLRAGDatabase
from graphcore.summary import SummaryConfig

@dataclass
class ProverOptions:
    capture_output: bool
    keep_folder: bool

@dataclass
class CryptoContext:
    llm: BoundLLM
    rag_db: PostgreSQLRAGDatabase
    prover_opts: ProverOptions
    summarizer: Optional[SummaryConfig] = field(init=False)

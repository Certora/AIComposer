"""
CLI entry point for mutation testing of corpus-extracted specifications.

Reads pipeline state produced by corpus_ingest.py, generates source-code
mutations targeting each extracted CVL rule, and reports which mutations
the rules caught vs missed.

Usage:
    python corpus_mutate.py ./pdfs
    python corpus_mutate.py ./pdfs --concurrency 5 --model claude-sonnet-4-6
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import composer.bind as _

from composer.corpus.mutate import try_mutate, RuleValidationCache
from composer.corpus.pipeline import StateCache
from composer.workflow.services import create_llm_base, store_context, indexed_store_context
from composer.io.context import with_handler
from composer.io.event_handler import NullEventHandler
from composer.io.protocol import NullIOHandler
from composer.ui.simple_console_handler import SimpleConsoleHandler
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.models import get_model


@dataclass
class LLMConfig:
    """Satisfies ModelOptionsBase protocol for create_llm_base."""
    model: str
    tokens: int
    thinking_tokens: int | None
    memory_tool: bool = False
    interleaved_thinking: bool = False

class DebuggingHandler(NullIOHandler[Any]):
    async def log_state_update(self, path: list[str], st: dict):
        print(f"{path} -> {st}")

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None):
        print("Job Start")

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mutation testing against corpus-extracted CVL specifications",
    )
    parser.add_argument(
        "pdf_dir", type=Path,
        help="Directory containing the same PDFs used for corpus ingestion",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("./corpus_work"),
        help="Directory for downloaded sources (same as corpus_ingest)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("./corpus_cache"),
        help="Directory for local debug cache (same as corpus_ingest)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./mutation_output"),
        help="Directory for mutation testing results",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Model for mutation generation (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--tokens", type=int, default=16_000,
        help="Max output tokens (default: 16000)",
    )
    parser.add_argument(
        "--thinking-tokens", type=int, default=10_000,
        help="Thinking budget (default: 10000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent mutation agents (default: 5)",
    )
    parser.add_argument(
        "--cache-namespace", default="corpus",
        help="Store namespace prefix — must match corpus_ingest (default: corpus)",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        print(f"Error: {args.pdf_dir} is not a directory")
        return 1

    mutation_llm = create_llm_base(LLMConfig(
        model=args.model,
        tokens=args.tokens,
        thinking_tokens=args.thinking_tokens,
    ))

    model = get_model()


    async with store_context() as store, indexed_store_context(embedder=DefaultEmbedder(model)) as ind_store, with_handler(
        SimpleConsoleHandler(), NullEventHandler()
    ):
        cache = StateCache(store, args.cache_dir, namespace=args.cache_namespace)
        sem = asyncio.Semaphore(args.concurrency)

        await try_mutate(
            work_dir=args.work_dir,
            pdf_dir=args.pdf_dir,
            output_dir=args.output_dir,
            mutation_llm=mutation_llm,
            state_cache=cache,
            ind_store=ind_store,
            store_ns=("mutation",),
            sem=sem,
            rule_cache=RuleValidationCache(store, ("rule_validation", args.cache_namespace))
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

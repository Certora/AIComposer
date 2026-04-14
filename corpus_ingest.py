"""
CLI entry point for the security report corpus ingestion pipeline.

Usage:
    python -m scripts.corpus_ingest ./pdfs
    python -m scripts.corpus_ingest ./pdfs --concurrency 10 --model-analysis claude-sonnet-4-6
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import composer.bind as _

from composer.corpus.pipeline import STAGE_NAMES, StateCache, process_all
from composer.workflow.services import create_llm_base, store_context
from composer.io.context import with_handler
from composer.io.event_handler import NullEventHandler
from composer.io.protocol import NullIOHandler

@dataclass
class LLMConfig:
    """Satisfies ModelOptionsBase protocol for create_llm_base."""
    model: str
    tokens: int
    thinking_tokens: int | None
    memory_tool: bool = False
    interleaved_thinking: bool = False

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest security report PDFs into structured corpus",
    )
    parser.add_argument(
        "pdf_dir", type=Path,
        help="Directory containing security report PDFs",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("./corpus_work"),
        help="Directory for downloaded sources",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("./corpus_cache"),
        help="Directory for local debug cache",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./corpus_output"),
        help="Directory for output JSONs",
    )
    parser.add_argument(
        "--model-triage", default="claude-sonnet-4-6",
        help="Model for triage stage (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--model-extraction", default="claude-sonnet-4-6",
        help="Model for extraction stage (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--model-analysis", default="claude-opus-4-6",
        help="Model for analysis stage (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--tokens", type=int, default=16_000,
        help="Max output tokens for all LLM calls (default: 16000)",
    )
    parser.add_argument(
        "--thinking-tokens", type=int, default=10_000,
        help="Thinking budget for analysis model (default: 10000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent LLM operations (default: 5)",
    )
    parser.add_argument(
        "--cache-namespace", default="corpus",
        help="Store namespace prefix for cache keys (default: corpus)",
    )
    parser.add_argument(
        "--force", nargs="+", choices=STAGE_NAMES, default=[],
        metavar="STAGE",
        help=f"Force re-run of specific stages (choices: {', '.join(STAGE_NAMES)}). "
             "Forcing a stage also invalidates all downstream stages.",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        print(f"Error: {args.pdf_dir} is not a directory")
        return 1

    # Triage + extraction: no thinking (single-shot structured output)
    llm_triage = create_llm_base(LLMConfig(
        model=args.model_triage,
        tokens=args.tokens,
        thinking_tokens=None,
    ))
    llm_extraction = create_llm_base(LLMConfig(
        model=args.model_extraction,
        tokens=args.tokens,
        thinking_tokens=None,
    ))
    # Analysis: thinking enabled (agentic, needs reasoning)
    llm_analysis = create_llm_base(LLMConfig(
        model=args.model_analysis,
        tokens=args.tokens,
        thinking_tokens=args.thinking_tokens,
    ))


    async with store_context() as store, with_handler(
        NullIOHandler(), NullEventHandler()
    ):
        cache = StateCache(store, args.cache_dir, namespace=args.cache_namespace)
        sem = asyncio.Semaphore(args.concurrency)
        report_sem = asyncio.Semaphore(10)

        await process_all(
            cache, pdf_dir=args.pdf_dir, work_dir=args.work_dir, output_dir=args.output_dir,
            llm_triage=llm_triage, llm_extraction=llm_extraction, llm_analysis=llm_analysis,
            sem=sem, report_sem=report_sem,
            force_stages=frozenset(args.force)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

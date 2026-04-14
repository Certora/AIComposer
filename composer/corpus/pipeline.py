"""
Pipeline orchestrator — stages, caching, and concurrent execution.

Processes security report PDFs through: preprocess → triage → resolve links →
extract → download → analyze. Uses dual-write caching (LangGraph store +
local JSON) for resumability and concurrent execution bounded by a global
semaphore.
"""

import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from composer.corpus.models import (
    CorpusEntry, PipelineState, ProcessedReport, RuleRef, UnmatchedRule,
)
from composer.corpus.preprocess import preprocess_pdf
from composer.corpus.triage import triage_report
from composer.corpus.extraction import extract_report
from composer.corpus.cloud_run import download_sources, resolve_cloud_links
from composer.corpus.analysis import analyze_source_tree

STAGE_NAMES = ("triage", "extraction", "analysis")

# Ordered list: forcing a stage invalidates it and everything after it.
_STAGE_ORDER = {name: i for i, name in enumerate(STAGE_NAMES)}


def _apply_force(state: PipelineState, force_stages: frozenset[str]) -> None:
    """Clear cached results for forced stages and all downstream stages."""
    if not force_stages:
        return

    # Find the earliest forced stage
    earliest = min(_STAGE_ORDER[s] for s in force_stages)

    # Always clear skipped_reason so the pipeline re-evaluates
    state.skipped_reason = None

    if earliest <= _STAGE_ORDER["triage"]:
        state.triage = None

    if earliest <= _STAGE_ORDER["extraction"]:
        state.extraction = None

    if earliest <= _STAGE_ORDER["analysis"]:
        state.analyzed_trees = {}
        state.unmatched = []


# ---------------------------------------------------------------------------
# State cache — dual-write: LangGraph store + local JSON
# ---------------------------------------------------------------------------

_CACHE_VERSION = 3

class StateCache:
    """Dual-write cache: LangGraph store (authoritative) + local JSON (debug)."""

    def __init__(
        self, store: BaseStore, cache_dir: Path, namespace: str = "corpus",
    ) -> None:
        self._store = store
        self._cache_dir = cache_dir
        self._namespace_prefix = namespace
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _namespace(self, pdf_hash: str) -> tuple[str, ...]:
        return (self._namespace_prefix, str(_CACHE_VERSION), pdf_hash)

    async def load(self, pdf_hash: str) -> PipelineState | None:
        """Load pipeline state from the store. Returns None if not found."""
        item = await self._store.aget(self._namespace(pdf_hash), "state")
        if item is None:
            return None
        return PipelineState.model_validate(item.value)

    async def save(self, state: PipelineState) -> None:
        """Write state to both store and local JSON."""
        dumped = state.model_dump()
        await self._store.aput(self._namespace(state.pdf_hash), "state", dumped)
        json_path = self._cache_dir / f"{state.pdf_hash}.json"
        json_path.write_text(json.dumps(dumped, indent=2))


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _assemble_result(state: PipelineState) -> ProcessedReport | None:
    """Assemble a ProcessedReport from the current pipeline state.

    Returns None if triage hasn't completed (no metadata to report).
    """
    if state.triage is None:
        return None

    all_entries: list[CorpusEntry] = []
    for entries in state.analyzed_trees.values():
        all_entries.extend(entries)

    return ProcessedReport(
        source_pdf=Path(state.pdf_path).name,
        metadata=state.triage,
        entries=all_entries,
        unmatched=state.unmatched,
        skipped_reason=state.skipped_reason,
    )


def _url_dir_name(url: str) -> str:
    """Derive a short, filesystem-safe directory name from a URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]

async def ensure_report(
    work_dir: Path,
    state: PipelineState,
    url: str,
    report_sem: asyncio.Semaphore
) -> Path | None:
    if url in state.source_dirs and (dir := (work_dir / state.source_dirs[url])).exists():
        return dir
    url_dir = _url_dir_name(url)
    res_path = work_dir / state.pdf_hash / url_dir
    result = await download_sources(url, res_path, report_sem)
    if result is not None:
        return None
    state.source_dirs[url] = str(res_path.relative_to(work_dir))
    return res_path


# ---------------------------------------------------------------------------
# Single PDF processing
# ---------------------------------------------------------------------------

async def process_single_pdf(
    pdf_dir: Path,
    pdf_path: Path,
    work_dir: Path,
    cache: StateCache,
    llm_triage: BaseChatModel,
    llm_extraction: BaseChatModel,
    llm_analysis: BaseChatModel,
    llm_sem: asyncio.Semaphore,
    report_sem: asyncio.Semaphore,
    extra_tools: list[BaseTool] | None = None,
    force_stages: frozenset[str] = frozenset(),
) -> ProcessedReport | None:
    """Process a single PDF through all pipeline stages.

    Each LLM-calling stage acquires the semaphore to bound total
    concurrent API requests across all PDFs and property groups.
    """
    # Stage 0: Preprocess (deterministic, no semaphore needed)
    pdf = preprocess_pdf(pdf_path)

    # Load or create state
    state = await cache.load(pdf.content_hash) or PipelineState(
        pdf_hash=pdf.content_hash,
        pdf_path=str(pdf_path.relative_to(pdf_dir)),
    )

    _apply_force(state, force_stages)

    if state.skipped_reason is not None:
        return _assemble_result(state)

    # Stage 1: Triage
    if state.triage is None:
        async with llm_sem:
            state.triage = await triage_report(pdf, llm_triage)
        await cache.save(state)

    triage = state.triage

    if triage.report_type != "formal_verification":
        state.skipped_reason = f"Report type: {triage.report_type}"
        await cache.save(state)
        return _assemble_result(state)

    # Stage 2: Resolve cloud run links from PDF
    prover_urls = [
        url for url in pdf.links
        if "prover.certora.com/output/" in url
    ]
    if not prover_urls:
        state.skipped_reason = "No cloud run URLs found in PDF"
        await cache.save(state)
        return _assemble_result(state)

    resolved_links = await resolve_cloud_links(prover_urls, llm_sem)

    # Stage 3: Extraction (with resolved link context)
    if state.extraction is None:
        async with llm_sem:
            state.extraction = await extract_report(
                pdf, triage, resolved_links, llm_extraction,
            )
        await cache.save(state)

    extraction = state.extraction

    # Stage 4: Download sources — one directory per cloud run URL
    cloud_urls = list({link.output_url for link in resolved_links})
    pdf_work_dir = work_dir / pdf.content_hash

    download_jobs = [
        ensure_report(
            work_dir, state, url, report_sem
        ) for url in cloud_urls
    ]
    await asyncio.gather(*download_jobs)
    await cache.save(state)

    if not state.source_dirs:
        state.skipped_reason = "No sources downloaded (all URLs failed)"
        await cache.save(state)
        return _assemble_result(state)

    # Stage 5: Analysis — one agent per source tree
    # Group rules by cloud_run_url
    rules_by_url: dict[str, list[RuleRef]] = defaultdict(list)
    for group in extraction.properties:
        for i, rule in enumerate(group.rules):
            if rule.cloud_run_url is not None and rule.cloud_run_url in state.source_dirs:
                rules_by_url[rule.cloud_run_url].append(
                    RuleRef(group=group, rule_index=i)
                )
            else:
                # No source tree for this rule
                state.unmatched.append(UnmatchedRule(
                    rule=rule,
                    property_id=group.id,
                    reason="no associated cloud run URL" if rule.cloud_run_url is None
                        else f"source download failed for {rule.cloud_run_url}",
                ))

    pending_urls = [
        url for url in rules_by_url
        if url not in state.analyzed_trees
    ]

    protocol_desc = triage.protocol_description

    if pending_urls:
        state_lock = asyncio.Lock()

        async def _analyze_one(url: str) -> None:
            async with llm_sem:
                entries, unmatched = await analyze_source_tree(
                    rules_by_url[url],
                    state.source_dirs[url],
                    protocol_desc,
                    llm_analysis,
                    extra_tools=extra_tools,
                )
            async with state_lock:
                state.analyzed_trees[url] = entries
                state.unmatched.extend(unmatched)
                await cache.save(state)

        await asyncio.gather(
            *(_analyze_one(u) for u in pending_urls)
        )

    return _assemble_result(state)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

async def process_all(
    cache: StateCache,
    *,
    pdf_dir: Path,
    work_dir: Path,
    output_dir: Path,
    llm_triage: BaseChatModel,
    llm_extraction: BaseChatModel,
    llm_analysis: BaseChatModel,
    sem: asyncio.Semaphore,
    report_sem: asyncio.Semaphore,
    extra_tools: list[BaseTool] | None = None,
    force_stages: frozenset[str] = frozenset(),
) -> None:
    """Process all PDFs in a directory concurrently.

    Concurrency across both PDFs and source trees is bounded by ``sem``.
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found.")
        return

    total = len(pdfs)
    print(f"Found {total} PDFs to process.")

    async def _process_one(idx: int, path: Path) -> ProcessedReport | None:
        print(f"[{idx + 1}/{total}] Processing {path.name}...")
        try:
            result = await process_single_pdf(
                pdf_dir,
                path, work_dir, cache,
                llm_triage, llm_extraction, llm_analysis,
                sem, extra_tools=extra_tools,
                force_stages=force_stages, report_sem=report_sem
            )
            if result is not None and result.skipped_reason:
                print(f"  Skipped: {result.skipped_reason}")
            elif result is not None:
                print(
                    f"  Done: {len(result.entries)} entries, "
                    f"{len(result.unmatched)} unmatched"
                )
            return result
        except Exception as e:
            print(f"  Error processing {path.name}: {e}")
            return None

    results = await asyncio.gather(
        *(_process_one(i, p) for i, p in enumerate(pdfs))
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for result in results:
        if result is None:
            continue
        slug = result.source_pdf.removesuffix(".pdf").replace(" ", "_")
        out_path = output_dir / f"{slug}.json"
        out_path.write_text(result.model_dump_json(indent=2))
        manifest.append({
            "source_pdf": result.source_pdf,
            "output_file": out_path.name,
            "entries": len(result.entries),
            "unmatched": len(result.unmatched),
            "skipped": result.skipped_reason,
        })

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} reports written. Manifest: {manifest_path}")

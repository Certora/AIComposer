"""Foundry cheatcode RAG builder — sketch.

The foundry cheatcode docs are HTML fragments (no ``<html>``/``<body>``/
``<section>`` wrappers) — they're a flat sequence of ``h2``/``h3``
headers interleaved with ``<pre>``, ``<p>``, ``<table>``, ``<ul>`` and
``<div class="admonition">`` blocks.

Each page documents exactly one cheatcode and has a stable structure::

    <h2><code>NAME</code></h2>
    <h3>Signature</h3>      <pre> ... </pre>
    <h3>Description</h3>    <p> ... </p>
    <h3>Parameters</h3>     <table> ... </table>   (optional)
    <h3>Returns</h3>        <table> ... </table>   (optional)
    <h3>Examples</h3>       <pre>...</pre>  |  <div class="admonition code-group"><pre>..</pre>..</div>
    <h3>Gotchas</h3>        <div class="admonition warning|note">...</div>   (optional)
    <h3>Related Cheatcodes</h3>  <ul> ... </ul>

This script reuses the shared streaming/chunking machinery in
``text_processors.py`` but the *boundary detection* and *table
translation* are cheatcode-specific:

- ``<h2>`` / ``<h3>`` define implicit section boundaries (no
  ``<section>`` containers exist, so we walk children linearly and
  start a new logical section every time we hit a header).
- Parameter / Returns ``<table>`` elements are translated into a
  parameter-list-style text format (``- name (type): description``)
  rather than markdown tables — easier for the LLM to read after
  embedding.
- ``<div class="admonition code-group">`` containers wrap multiple
  ``<pre>`` examples; each ``<pre>`` becomes its own code ref.
- Other ``<div class="admonition ...">`` blocks (warning / note) are
  surfaced with a leading ``"Warning:"`` / ``"Note:"`` marker.

The script accepts one or more HTML files on the command line and either
prints the resulting chunks to stdout (``--print``) or writes them to a
RAG database via the same ``add_chunks_batch`` / ``add_manual_section``
path ``ragbuild.py`` uses for the CVL manual.
"""

import argparse
import asyncio
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator

import spacy
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from composer.rag.db import FOUNDRY_DEFAULT_CONNECTION, rag_context
from composer.rag.models import get_model
from composer.rag.types import BlockChunk
from composer.scripts.text_processors import (
    BlockBuilder, BuilderConfig, TextCollector, TextStreamer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table translation
# ---------------------------------------------------------------------------


@dataclass
class _TableRow:
    cells: list[str]


def _extract_table(tbl: Tag) -> tuple[list[str], list[_TableRow]]:
    """Return ``(headers, rows)`` for a ``<table>``. Headers come from the
    ``<thead>``'s ``<th>`` cells; rows from the ``<tbody>``'s ``<tr>``."""
    headers: list[str] = []
    thead = tbl.find("thead")
    if isinstance(thead, Tag):
        for th in thead.find_all("th"):
            headers.append(th.get_text(" ", strip=True))

    rows: list[_TableRow] = []
    tbody = tbl.find("tbody")
    if isinstance(tbody, Tag):
        for tr in tbody.find_all("tr"):
            if not isinstance(tr, Tag):
                continue
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            rows.append(_TableRow(cells=cells))
    return headers, rows


def _format_param_table(tbl: Tag) -> str:
    """Translate a Parameters/Returns table into a parameter-list-style
    string. Handles the two column shapes the cheatcode docs use:

    * ``Parameter | Type | Description``
    * ``Type | Description``                  (anonymous returns)
    """
    headers, rows = _extract_table(tbl)
    if not rows:
        return ""

    # Lowercase the headers we recognize so capitalization differences
    # between pages don't matter.
    norm = [h.strip().lower() for h in headers]

    lines: list[str] = []
    if norm[:3] == ["parameter", "type", "description"]:
        for r in rows:
            if len(r.cells) < 3:
                continue
            name, ty, desc = r.cells[0], r.cells[1], r.cells[2]
            lines.append(f"- {name} ({ty}): {desc}")
    elif norm[:2] == ["type", "description"]:
        for r in rows:
            if len(r.cells) < 2:
                continue
            ty, desc = r.cells[0], r.cells[1]
            lines.append(f"- ({ty}) {desc}")
    else:
        # Unknown shape — fall back to a generic key=value join. Surface
        # the shape in the output so reviewers can decide whether to
        # special-case it.
        lines.append(f"(unrecognized table headers: {headers!r})")
        for r in rows:
            lines.append("- " + " | ".join(r.cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block-level helpers
# ---------------------------------------------------------------------------


def _extract_code(pre: Tag) -> str:
    """Pull the text content of a ``<pre><code>...</code></pre>`` block."""
    code = pre.find("code")
    if isinstance(code, Tag):
        return code.get_text("").rstrip("\n")
    return pre.get_text("").rstrip("\n")


def _extract_admonition_text(div: Tag) -> str:
    """Render an admonition (warning / note / etc.) as a single paragraph
    with a leading marker — preserves the semantic emphasis without the
    HTML structure."""
    classes = div.get("class") or []
    marker = "Note"
    for c in classes:
        if c == "admonition":
            continue
        marker = c.capitalize()
        break
    # Concatenate the inner paragraphs with a blank line between them.
    parts = []
    for p in div.find_all("p"):
        parts.append(p.get_text(" ", strip=True))
    body = "\n\n".join(parts) or div.get_text(" ", strip=True)
    return f"{marker}: {body}"


def _is_code_group(div: Tag) -> bool:
    classes = div.get("class") or []
    return "admonition" in classes and "code-group" in classes


def _convert_ul(ul: Tag, depth: int = 0) -> str:
    out: list[str] = []
    for li in ul.find_all("li", recursive=False):
        if not isinstance(li, Tag):
            continue
        indent = "  " * depth + "- "
        # Strip nested <ul>/<ol> first, render them after with extra indent.
        nested = []
        for child in list(li.children):
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                nested.append(child)
                child.extract()
        out.append(indent + li.get_text(" ", strip=True))
        for n in nested:
            out.append(_convert_ul(n, depth + 1))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Header-driven section walker
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    """A logical section keyed off an ``h3`` (or deeper) header. Contains
    a contiguous run of children up to the next same-or-shallower header."""
    name: str
    level: int  # the H-level (2/3/4/...)
    children: list[Tag | NavigableString]


def _split_sections(root_children: Iterable[Tag | NavigableString]) -> tuple[str | None, list[_Section]]:
    """Walk the flat children of the document and group them by ``h2``/``h3``
    boundaries. Returns ``(cheatcode_name, sections)`` — the name comes from
    the first ``<h2>``; sections are everything after, keyed by ``<h3>``.

    A section's ``children`` list excludes the header tag itself.
    """
    cheatcode_name: str | None = None
    sections: list[_Section] = []
    current: _Section | None = None

    for ch in root_children:
        if isinstance(ch, NavigableString):
            if current is not None and ch.strip():
                current.children.append(ch)
            continue
        if not isinstance(ch, Tag):
            continue

        if ch.name == "h2":
            cheatcode_name = ch.get_text(" ", strip=True)
            continue

        if ch.name and ch.name.startswith("h") and len(ch.name) == 2 and ch.name[1].isdigit():
            level = int(ch.name[1])
            current = _Section(
                name=ch.get_text(" ", strip=True),
                level=level,
                children=[],
            )
            sections.append(current)
            continue

        if current is None:
            # Stray content before the first header — synthesize an
            # "intro" bucket so we don't drop it.
            current = _Section(name="", level=3, children=[])
            sections.append(current)
        current.children.append(ch)

    return cheatcode_name, sections


# ---------------------------------------------------------------------------
# Section translation
# ---------------------------------------------------------------------------


def _translate_section(
    streamer: TextStreamer,
    builder: BlockBuilder,
    section: _Section,
) -> None:
    """Convert the contents of one cheatcode subsection into streamed text
    plus chunk-builder operations. Mutates ``builder`` in place; the caller
    decides when to push siblings.

    The dispatch leans on the section name where it helps (Parameters /
    Returns get the table-as-parameter-list translation) but otherwise
    handles any tag inside any section in a uniform way.
    """
    def emit_block(txt: str, *, structured: bool, unbreakable: bool) -> None:
        """Append a block-level item with a paragraph break after it.

        ``BlockBuilder.append_text`` concatenates without inserting any
        separator, so consecutive block-level items (two admonitions, a
        paragraph then a list, etc.) would otherwise run together
        ("...calls.Warning: ..."). The trailing ``\\n\\n`` is stripped by
        ``_push``'s ``strip()`` on the final chunk so it doesn't leak.
        """
        if not txt:
            return
        streamer.stream_text(txt)
        builder.append_text(
            txt, is_structured_boundary=structured, unbreakable=unbreakable,
        )
        streamer.stream_text("\n")
        builder.append_text("\n", is_structured_boundary=False, unbreakable=False)

    for ch in section.children:
        if isinstance(ch, NavigableString):
            txt = str(ch).strip()
            if txt:
                streamer.stream_text(txt)
                builder.append_text(txt, is_structured_boundary=False, unbreakable=False)
            continue
        if not isinstance(ch, Tag):
            continue

        match ch.name:
            case "p":
                emit_block(ch.get_text(" ", strip=True), structured=True, unbreakable=False)
            case "pre":
                code = _extract_code(ch)
                streamer.stream_code(code)
                builder.add_code(code)
            case "table":
                # Parameters / Returns sections: translate to a typed list.
                # If a table shows up elsewhere, the same translation is
                # still sensible — they all describe (name, type, desc).
                emit_block(_format_param_table(ch), structured=True, unbreakable=True)
            case "ul" | "ol":
                emit_block(_convert_ul(ch), structured=True, unbreakable=True)
            case "div":
                if _is_code_group(ch):
                    # Multiple <pre> blocks tabbed together — emit each
                    # as a separate code ref so the embedding sees them
                    # individually.
                    for pre in ch.find_all("pre"):
                        if not isinstance(pre, Tag):
                            continue
                        code = _extract_code(pre)
                        streamer.stream_code(code)
                        builder.add_code(code)
                elif "admonition" in (ch.get("class") or []):
                    emit_block(
                        _extract_admonition_text(ch),
                        structured=True, unbreakable=True,
                    )
                else:
                    emit_block(
                        ch.get_text(" ", strip=True),
                        structured=False, unbreakable=False,
                    )
            case _:
                txt = ch.get_text(" ", strip=True)
                if txt:
                    print(
                        f"[foundry_ragbuild] unhandled tag <{ch.name}> in "
                        f"section {section.name!r}; falling back to text",
                        file=sys.stderr,
                    )
                    emit_block(txt, structured=False, unbreakable=False)


# ---------------------------------------------------------------------------
# Top-level: file → chunks
# ---------------------------------------------------------------------------


# Sections that are short, structured, and conceptually one "function
# summary" — these get collapsed into a single chunk per cheatcode (with
# the cheatcode name as the sole header, and section names emitted inline
# as "Signature:" / "Description:" / ... labels). Code refs (signature +
# any inline pre blocks) compress to one tag each, so even after merging
# we stay well under ``max_length``.
_SUMMARY_SECTIONS = {"signature", "description", "parameters", "returns"}


def chunk_cheatcode_html(
    html: str,
    *,
    config: BuilderConfig,
) -> Iterator[BlockChunk]:
    """Translate a single cheatcode HTML fragment into ``BlockChunk``s.

    Summary sections (signature / description / parameters / returns) collapse
    into a single chunk keyed by the cheatcode name. Examples, Gotchas, and
    any other ``h3`` get their own chunk so an example block doesn't drag a
    description into a sentence split. Related Cheatcodes is dropped (nav
    metadata).
    """
    soup = BeautifulSoup(html, "html.parser")
    top_children: list[Tag | NavigableString] = [
        c for c in soup.children if isinstance(c, (Tag, NavigableString))
    ]
    cheatcode_name, sections = _split_sections(top_children)
    if cheatcode_name is None:
        return iter(())

    # Drop the "Related Cheatcodes" section — it's nav metadata for the
    # docs site, not content the model needs to recall. Keep this filter
    # narrow; everything else is fair game.
    sections = [s for s in sections if s.name.strip().lower() != "related cheatcodes"]

    sink = TextCollector()
    root_streamer = TextStreamer(sink, min_depth=0, parent=None, header=[cheatcode_name])

    summary_chunks: list[BlockChunk] = []
    rest_chunks: list[BlockChunk] = []

    summary_headers = ["Cheatcodes", cheatcode_name]
    summary_builder = BlockBuilder(header=summary_headers, config=config)
    summary_streamer = root_streamer.child(summary_headers)
    summary_started = False

    for section in sections:
        if section.name.strip().lower() in _SUMMARY_SECTIONS:
            # Prefix this sub-section's content with a label so the merged
            # chunk still preserves the original structure for the LLM.
            if section.name:
                label = f"\n\n{section.name}:\n"
                summary_streamer.stream_text(label)
                summary_builder.append_text(
                    label, is_structured_boundary=True, unbreakable=False,
                )
            _translate_section(summary_streamer, summary_builder, section)
            summary_started = True
            continue
        if section.name.lower() in {"see also", "related cheatcodes"}:
            continue
        # Non-summary section — its own chunk.
        headers = [*summary_headers, section.name]
        section_streamer = root_streamer.child(headers)
        builder = BlockBuilder(header=headers, config=config)
        _translate_section(section_streamer, builder, section)
        rest_chunks.extend(builder.finish())

    if summary_started:
        summary_chunks.extend(summary_builder.finish())

    return iter(summary_chunks + rest_chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_chunk_for_review(c: BlockChunk) -> str:
    lines = [
        "=" * 78,
        f"headers : {' / '.join(h for h in c.headers if h)}",
        f"part    : {c.part}",
        f"code_refs: {len(c.code_refs)}",
        "-" * 78,
        c.chunk,
    ]
    for i, code in enumerate(c.code_refs):
        lines.append("-" * 78)
        lines.append(f"[code_ref #{i}]")
        lines.append(code)
    return "\n".join(lines)


_BATCH_SIZE = 50


async def _async_main(args: argparse.Namespace) -> int:
    files: list[pathlib.Path] = list(args.html_files)
    missing = [f for f in files if not f.is_file()]
    if missing:
        for f in missing:
            print(f"Error: not a file: {f}", file=sys.stderr)
        return 1

    nlp = spacy.load("en_core_web_sm")
    config = BuilderConfig(nlp=nlp, max_length=args.max_length)

    if args.print_only:
        for f in files:
            for c in chunk_cheatcode_html(f.read_text(), config=config):
                print(_format_chunk_for_review(c))
        print("=" * 78)
        return 0

    output = args.output or FOUNDRY_DEFAULT_CONNECTION
    async with rag_context(output, get_model()) as db:
        buffer: list[BlockChunk] = []
        n_chunks = 0
        for f in files:
            logger.info("Processing %s", f)
            for c in chunk_cheatcode_html(f.read_text(), config=config):
                buffer.append(c)
                n_chunks += 1
                if len(buffer) >= _BATCH_SIZE:
                    await db.add_chunks_batch(buffer)
                    buffer = []
        if buffer:
            await db.add_chunks_batch(buffer)

    logger.info(
        "Wrote %d chunks from %d file(s) to %s", n_chunks, len(files), output,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "html_files", nargs="+", type=pathlib.Path, metavar="HTML_FILE",
        help="One or more foundry cheatcode HTML files to ingest.",
    )
    parser.add_argument(
        "--max-length", type=int, default=2000,
        help="Soft cap on chunk length in characters (default: 2000).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="ChromaDB directory or PostgreSQL connection string. "
             f"Defaults to {FOUNDRY_DEFAULT_CONNECTION}.",
    )
    parser.add_argument(
        "--print", dest="print_only", action="store_true",
        help="Don't touch the database — just print chunks to stdout for "
             "manual review (matches the original spot-test mode).",
    )
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())

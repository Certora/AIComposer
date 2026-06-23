#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional, Generator, cast

from dataclasses import dataclass
import logging
import argparse
import contextvars
import pathlib

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
import spacy
from composer.rag.db import get_rag_db, DEFAULT_CONNECTION
from composer.rag.types import BlockChunk
from composer.rag.text import get_code_refs
from composer.rag.models import get_model
from composer.scripts.text_processors import (
    BlockBuilder, BuilderConfig, TextCollector, TextStreamer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Header:
    head: str
    level: int

def get_section_header(s: Tag) -> Optional[Header]:
    head_tag : Optional[Tag] = None
    for ch in s.children:
        match ch:
            case Tag():
                if ch.name == "span":
                    if ch.text.strip():
                        return None
                elif ch.name.startswith("h"):
                    head_tag = ch
                    break
                else:
                    return None
            case NavigableString():
                if ch.text.strip():
                    return None
            case _:
                return None
    if head_tag is None:
        return None
    target = head_tag
    header = target.getText()
    level = int(target.name[1:])
    return Header(head=header, level=level)


max_length = 2000
nlp = spacy.load("en_core_web_sm")
main_body_ctx: contextvars.ContextVar[Tag] = contextvars.ContextVar('main_body')
section_label_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar('section_label', default=None)

# Shared text-processing config; passed into every ``BlockBuilder`` so the
# chunker has access to the spaCy model and the soft length cap. Kept as a
# module-level singleton because ``spacy.load`` is expensive and we don't
# want to re-load the model per call.
_builder_config = BuilderConfig(nlp=nlp, max_length=max_length)

def extract_code(s: Tag) -> str:
    assert s.name == "pre"
    block = ""
    for ch in s.children:
        match ch:
            case Tag():
                if ch.name != "span":
                    assert False
                block += ch.text
            case NavigableString():
                block += ch.text
            case _:
                # Other PageElement kinds (comments, etc.) contribute no code.
                pass
    return block.strip("\n")


def translate_text_block(s: Tag) -> str:
    assert s.name == "p"
    return s.get_text("")

def class_or_empty(s: Tag) -> list[str]:
    return cast(list[str], s.attrs.get("class", []))

def convert_li(s: Tag, depth: int) -> str:
    ident = (" " * depth) + " * "
    elem = ident
    for c in s.children:
        match c:
            case Tag(name="ul") | Tag(name="ol"):
                elem += "\n"
                elem += convert_ul(c, depth + 1) + "\n"
            case _:
                elem += c.getText("")
    return elem


def convert_ul(s: Tag, depth: int = 0) -> str:
    elems = []
    for l in s.find_all("li"):
        assert isinstance(l, Tag)
        elems.append(convert_li(l, depth))
    return "\n".join(elems)

def convert_table(s: Tag) -> str:
    """Render a docutils <table> as a markdown-style table so the row/column
    structure survives into the chunk text the LLM sees."""
    assert s.name == "table"
    rows: list[list[str]] = []
    header_row: int | None = None
    for tr in s.find_all("tr"):
        assert isinstance(tr, Tag)
        cells: list[str] = []
        is_header = False
        for cell in tr.find_all(["th", "td"], recursive=False):
            assert isinstance(cell, Tag)
            if cell.name == "th":
                is_header = True
            cells.append(" ".join(cell.get_text(" ").split()))
        if not cells:
            continue
        if is_header and header_row is None:
            header_row = len(rows)
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = []
    for i, r in enumerate(rows):
        lines.append("| " + " | ".join(r) + " |")
        if i == header_row:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)

def convert_aside(s: Tag) -> str:
    """Render an <aside> (docutils footnotes) as plain prose. Strip the "[n]"
    label/backlink span, which is just noise once the cross-reference is gone."""
    assert s.name == "aside"
    for label in s.find_all("span", {"class": "label"}):
        assert isinstance(label, Tag)
        label.decompose()
    return " ".join(s.get_text(" ").split())

def skip_class(s: Tag) -> bool:
    cl = class_or_empty(s)
    return "versionchanged" in cl or "versionadded" in cl or "math" in cl

def translate_block(streamer: TextStreamer, s: Tag, headers: list[str]) -> Generator[BlockChunk, None, None]:
    assert s.name == "section"
    builder = BlockBuilder(
        header=headers, config=_builder_config,
    )
    for ch in s.children:
        match ch:
            case Tag(name="nav"):
                continue
            case Tag(name="div") if skip_class(ch):
                continue
            case Tag(name="p"):
                block = translate_text_block(ch)
                streamer.stream_text(block)
                builder.append_text(block, True, False)
            case Tag(name="div") if "admonition" in ch.attrs.get("class", []):
                txt = ch.getText("")
                streamer.stream_text(txt)
                builder.append_text(txt, is_structured_boundary=True, unbreakable=True)
            case Tag(name="div") if isinstance(ch.find("pre"), Tag):
                code = extract_code(cast(Tag, ch.find("pre")))
                streamer.stream_code(code)
                builder.add_code(code)
            case Tag(name="ul") | Tag(name="ol"):
                ul = convert_ul(ch)
                streamer.stream_text(ul)
                builder.append_text(ul, is_structured_boundary=True, unbreakable=True)
            case Tag(name="table"):
                tbl = convert_table(ch)
                if tbl:
                    streamer.stream_text(tbl)
                    builder.append_text(tbl, is_structured_boundary=True, unbreakable=True)
            case Tag(name="aside"):
                aside = convert_aside(ch)
                if aside:
                    streamer.stream_text(aside)
                    builder.append_text(aside, is_structured_boundary=True, unbreakable=True)
            case Tag(name="span") if ch.getText() == "":
                continue
            case NavigableString():
                streamer.stream_text(ch.text)
                builder.append_text(ch.text, False, False)
            case Tag(name="section"):
                head = get_block_header(ch)
                sec = [ h for h in head if h ]
                streamer.section_start(sec)
                if "Example" in sec[-1]:
                    child_streamer = streamer
                else:
                    child_streamer = streamer.child(sec)
                first = True
                for child_block in translate_block(child_streamer, ch, head):
                    if first:
                        builder.push_child(child_block)
                        first = False
                    yield child_block
                streamer.section_end()
            case Tag(name=nm) if nm.startswith("h"):
                continue
            case _:
                pp = ch.name if isinstance(ch, Tag) else str(type(ch))
                print(f"Have unhandled element {pp} in {' '.join(headers)}")
    for x in builder.finish():
        yield x

def get_block_header(s: Tag) -> list[str]:
    assert isinstance(s, Tag)
    main_body = main_body_ctx.get()
    section_label = section_label_ctx.get()
    h = get_section_header(s)
    assert h is not None
    offset = 1 if section_label else 0
    headers = [""] * 6
    if section_label:
        headers[0] = section_label
    headers[min(h.level - 1 + offset, 5)] = h.head
    for p in s.parents:
        if p == main_body:
            break
        if p.name == "section":
            head = get_section_header(p)
            assert head is not None
            idx = min(head.level - 1 + offset, 5)
            if not headers[idx]:
                headers[idx] = head.head
    return headers

def sanity_checker(s: BlockChunk) -> None:
    seen = set()
    for (_, ref) in get_code_refs(s.chunk):
        if ref in seen:
            print(f"Duplicated code-ref {ref} in {s.chunk}")
        seen.add(ref)
        if ref >= len(s.code_refs):
            print(f"Orphan ref {ref} in {s.chunk}")

async def main() -> None:
    parser = argparse.ArgumentParser(description='Build RAG database from HTML documentation')
    parser.add_argument('files', nargs='+', metavar='HTML_FILE',
                        help='One or more HTML files to process directly')
    parser.add_argument('--output', '-o',
        help='Output directory for ChromaDB, or PostgreSQL connection string. '
             f'Defaults to PostgreSQL ({DEFAULT_CONNECTION})')
    args = parser.parse_args()

    file_entries = [{"file": (path:=pathlib.Path(f)), "section": path.stem} for f in args.files]

    output = args.output or DEFAULT_CONNECTION
    db = await get_rag_db(output, get_model())

    buffer: list[BlockChunk] = []

    for entry in file_entries:
        section_label_ctx.set(entry["section"])

        with open(entry["file"], "r") as f:
            manual = f.read()

        m = BeautifulSoup(manual, "html.parser")

        for s in m.find_all("a", {"class": "headerlink"}):
            s.decompose()

        # delete documentation of changes, not interesting to the LLM
        for section in m.find_all("section"):
            if not isinstance(section, Tag):
                continue
            sid = (section.attrs or {}).get("id", "")
            h = get_section_header(section)
            heading = h.head if h else ""
            if any(kw in sid or kw in heading for kw in (
                "changelog", "release-note", "release note", "changes-since", "changes since",
                "changes-introduced", "changes introduced", "changes to"
            )):
                section.decompose()

        main_body = m.find("div", {"itemprop": "articleBody"})
        assert isinstance(main_body, Tag), str(main_body)
        main_body_ctx.set(main_body)

        sink = TextCollector()
        root_streamer = TextStreamer(sink, 1, None, [])

        # singlehtml output wraps page sections in div.compound; individual html pages do not
        if main_body.find("div", class_="compound"):
            top_sections = main_body.select("div.compound > section")
        else:
            top_sections = main_body.find_all("section", recursive=False)

        for s in top_sections:
            assert isinstance(s, Tag)
            head = get_block_header(s)
            trunc_head = [h for h in head if h]
            section_streamer = root_streamer.child(trunc_head)
            for t in translate_block(section_streamer, s, head):
                sanity_checker(t)
                buffer.append(t)
                if len(buffer) == 50:
                    await db.add_chunks_batch(buffer)
                    buffer = []

        for i in sink.chunks():
            await db.add_manual_section(i)

    if buffer:
        await db.add_chunks_batch(buffer)

    logger.info(f"RAG database created at {output}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

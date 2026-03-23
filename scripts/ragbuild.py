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

from typing import Optional, Generator, cast, Iterable, Iterator

from dataclasses import dataclass
import json
import logging
import argparse
import contextvars
import pathlib
import sys

composer_dir = str(pathlib.Path(__file__).parent.parent.absolute())

if composer_dir not in sys.path:
    sys.path.append(composer_dir)


from bs4 import BeautifulSoup, NavigableString, Tag
import spacy #type: ignore
from composer.rag.db import PostgreSQLRAGDatabase, DEFAULT_CONNECTION
from composer.rag.types import BlockChunk
from composer.rag.text import get_code_refs, code_ref_tag
from composer.rag.models import get_model

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
section_label_ctx: contextvars.ContextVar[str] = contextvars.ContextVar('section_label', default="")

@dataclass
class InitContext:
    codes: list[str]
    context: str

class TextCollector:
    def __init__(self):
        self.bodies : list["TextStreamer"] = []

    def chunks(self) -> Iterator[BlockChunk]:
        for i in self.bodies:
            if not i._active:
                continue
            yield BlockChunk(
                i.header, 0, i._code_refs, i._buffer
            )

class TextStreamer():
    def __init__(self, sink: TextCollector, min_depth: int, parent: "TextStreamer | None", header: list[str]):
        self.parent = parent
        self.min_depth = min_depth
        self._code_refs : list[str] = []
        self._buffer = ""
        self.header = header
        self._active = len(header) > min_depth
        self._section_stack : list[list[str]] = []
        self.sink = sink
        sink.bodies.append(self)

    def stream_text(self, text: str):
        if self._active:
            self._buffer += " " + text
        if self.parent is not None:
            self.parent.stream_text(text)

    def stream_code(self, code: str):
        if self._active:
            self._buffer += "\n" + code_ref_tag(len(self._code_refs)) + "\n"
            self._code_refs.append(code)
        if self.parent is not None:
            self.parent.stream_code(code)
        
    def section_start(self, headers: list[str]):
        if self._active:
            self._buffer += f"\n\nSection: {' / '.join(headers)}\n"
            self._section_stack.append(headers)
        if self.parent is not None:
            self.parent.section_start(headers)
        
    def section_end(self):
        if self._active:
            assert len(self._section_stack) > 0
            curr_section = self._section_stack.pop()
            self._buffer += f"\n(End of Section {' / '.join(curr_section)})"
        if self.parent is not None:
            self.parent.section_end()

    def child(self, child_headers: list[str]) -> "TextStreamer":
        return TextStreamer(self.sink, self.min_depth, self, child_headers)

class BlockBuilder:
    def __init__(self, header: list[str]) -> None:
        self.siblings : list[BlockChunk] = []
        self.text = ""
        self.code_refs : list[str] = []
        self.part_counter = 0
        self.headers = header
        self.appended_child = False

    def _fixup_code_refs(self, text: str, refs: list[str]) -> str:
        replacement = {}
        id = 0
        replacer = text
        for (curr_name, ref) in get_code_refs(text):
            new_name = code_ref_tag(len(self.code_refs))
            assert ref in range(0, len(refs))
            new_key = f"repl{id}"
            new_id = f"%({new_key})s"
            replacer = replacer.replace(curr_name, new_id)
            replacement[new_key] = new_name
            self.code_refs.append(refs[ref])
            id += 1
        if len(replacement) == 0:
            return text
        return replacer % replacement

    def add_code(self, code: str) -> None:
        self.appended_child = False
        self.text += f"\n{code_ref_tag(len(self.code_refs))}"
        self.code_refs.append(code)

    def _push(self) -> None:
        if not self.text.strip():
            self.text = ""
            return
        self.siblings.append(BlockChunk(
            headers=self.headers,
            chunk=self.text.strip(),
            code_refs=self.code_refs,
            part=self.part_counter
        ))
        self.part_counter += 1
        self.code_refs = []
        self.text = ""

    def _init_new_chunk(self, new_text: str, unbreakable: bool, context: Optional[InitContext]) -> None:
        assert self.text == ""
        if context is not None:
            ctxt_string = self._fixup_code_refs(context.context, context.codes) + " "
        else:
            ctxt_string = ""
        if len(new_text) < max_length:
            self.text = ctxt_string + new_text
            return
        if unbreakable:
            self.text = ctxt_string + new_text
            self._push()
            return
        doc = nlp(ctxt_string + new_text)
        for s in doc.sents:
            l = s.text.strip()
            if not l:
                continue
            self.text += l + " "
            if len(self.text) > max_length:
                self._push()
        return

    def append_text(self, txt: str, is_structured_boundary: bool, unbreakable: bool) -> None:
        self.appended_child = False
        new_len = len(txt) + len(self.text)
        if new_len < max_length:
            self.text += txt
            return

        if is_structured_boundary and not unbreakable:
            new_nlp = nlp(txt).sents
            first_sent = None
            for next_s in new_nlp:
                if next_s.text.strip():
                    continue
                first_sent = next_s.text.strip()
                break

            last_sent_of_curr : Optional[str] = None
            curr_nlp = nlp(self.text)
            for curr_s in curr_nlp.sents:
                if curr_s.text.strip():
                    last_sent_of_curr = curr_s.text.strip()
            if first_sent is not None:
                self.text += " " + first_sent
            last_init : Optional[InitContext] = None
            if last_sent_of_curr is not None:
                last_init = InitContext(self.code_refs, last_sent_of_curr)
            self._push()
            self._init_new_chunk(txt, unbreakable=False, context=last_init)
            return
        elif is_structured_boundary and unbreakable:
            prev = self.text
            last_sentence: Optional[str] = None
            for s in nlp(prev).sents:
                if l := s.text.strip():
                    last_sentence = l + "\n"
            last_context = None
            if last_sentence is not None:
                last_context = InitContext(
                    codes=self.code_refs,
                    context=last_sentence
                )
            self._push()
            self._init_new_chunk(new_text=txt, unbreakable=True, context=last_context)
            return
        else:
            chunks = []
            curr_chunk = self.text
            for s in nlp(txt).sents:
                l = s.text.strip()
                if not l:
                    continue
                curr_chunk += " " + l
                if len(curr_chunk) > max_length:
                    chunks.append(curr_chunk)
                    curr_chunk = l
            if len(curr_chunk) > 0:
                chunks.append(curr_chunk)
            assert len(chunks) > 0
            for d in chunks[:-1]:
                self.text = d
                self._push()
            self.text = chunks[-1]

    def push_child(self, c: BlockChunk) -> None:
        if not self.text.strip():
            return
        first_sent = next(iter(nlp(c.chunk).sents))
        context = " / ".join([h for h in c.headers if len(h) > 0])
        self.text += context + "\n" + self._fixup_code_refs(first_sent.text, c.code_refs)
        self._push()

    def finish(self) -> Iterable[BlockChunk]:
        self._push()
        return self.siblings

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

def skip_class(s: Tag) -> bool:
    cl = class_or_empty(s)
    return "versionchanged" in cl or "versionadded" in cl or "math" in cl

def translate_block(streamer: TextStreamer, s: Tag, headers: list[str]) -> Generator[BlockChunk, None, None]:
    assert s.name == "section"
    builder = BlockBuilder(
        header=headers
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

def main() -> None:
    parser = argparse.ArgumentParser(description='Build RAG database from HTML documentation')
    parser.add_argument('files', nargs='+', metavar='HTML_FILE',
                        help='One or more HTML files to process directly')
    parser.add_argument('--connection', type=str, default=DEFAULT_CONNECTION,
                        help='Database connection string (default: rag_db)')
    args = parser.parse_args()

    file_entries = [{"file": (path:=pathlib.Path(f)), "section": path.stem} for f in args.files]

    db = PostgreSQLRAGDatabase(args.connection, get_model(), skip_test=False, create_schema=True)
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
                    db.add_chunks_batch(buffer)
                    buffer = []

        for i in sink.chunks():
            db.add_manual_section(i)

    if buffer:
        db.add_chunks_batch(buffer)

if __name__ == "__main__":
    main()

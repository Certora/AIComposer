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

from typing import Optional, List, Generator, cast, Iterable

from dataclasses import dataclass
import logging
import argparse
import contextvars
import pathlib
import sys

verisafe_dir = str(pathlib.Path(__file__).parent.parent.parent.absolute())

if verisafe_dir not in sys.path:
    sys.path.append(verisafe_dir)


from bs4 import BeautifulSoup, NavigableString, Tag
import spacy #type: ignore
from verisafe.rag.db import PostgreSQLRAGDatabase, DEFAULT_CONNECTION
from verisafe.rag.types import BlockChunk
from verisafe.rag.text import get_code_refs, code_ref_tag
from verisafe.rag.models import get_model

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
    assert head_tag is not None
    target = head_tag
    header = target.getText()
    level = int(target.name[1:])
    return Header(head=header, level=level)

max_length = 800
nlp = spacy.load("en_core_web_sm")
main_body_ctx: contextvars.ContextVar[Tag] = contextvars.ContextVar('main_body')

@dataclass
class InitContext:
    codes: List[str]
    context: str

class BlockBuilder:
    def __init__(self, header: List[str]) -> None:
        self.siblings : List[BlockChunk] = []
        self.text = ""
        self.code_refs : List[str] = []
        self.part_counter = 0
        self.headers = header
        self.appended_child = False

    def _fixup_code_refs(self, text: str, refs: List[str]) -> str:
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
        self.text += f"\n{code_ref_tag(len(self.code_refs))}>"
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

def class_or_empty(s: Tag) -> List[str]:
    return cast(List[str], s.attrs.get("class", []))

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

def translate_block(s: Tag, headers: List[str]) -> Generator[BlockChunk, None, None]:
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
                builder.append_text(block, True, False)
            case Tag(name="div") if "admonition" in ch.attrs.get("class", []):
                builder.append_text(ch.getText(""), is_structured_boundary=True, unbreakable=True)
            case Tag(name="div") if isinstance(ch.find("pre"), Tag):
                builder.add_code(extract_code(cast(Tag, ch.find("pre"))))
            case Tag(name="ul") | Tag(name="ol"):
                builder.append_text(convert_ul(ch), is_structured_boundary=True, unbreakable=True)
            case Tag(name="span") if ch.getText() == "":
                continue
            case NavigableString():
                builder.append_text(ch.text, False, False)
            case Tag(name="section"):
                head = get_block_header(ch)
                first = True
                for child_block in translate_block(ch, head):
                    if first:
                        builder.push_child(child_block)
                        first = False
                    yield child_block
            case Tag(name=nm) if nm.startswith("h"):
                continue
            case _:
                pp = ch.name if isinstance(ch, Tag) else str(type(ch))
                print(f"Have unhandled element {pp} in {' '.join(headers)}")
    for x in builder.finish():
        yield x

def get_block_header(s: Tag) -> List[str]:
    assert isinstance(s, Tag)
    main_body = main_body_ctx.get()
    h = get_section_header(s)
    headers = [""] * 6
    assert h is not None
    headers[h.level - 1] = h.head
    for p in s.parents:
        if p == main_body:
            break
        if p.name == "section":
            head = get_section_header(p)
            assert head is not None
            assert headers[head.level - 1] == ""
            headers[head.level - 1] = head.head
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
    parser.add_argument('html_file', help='Path to the HTML file to process')
    args = parser.parse_args()

    with open(args.html_file, "r") as f:
        manual = f.read()

    m = BeautifulSoup(manual, "html.parser")

    for s in m.find_all("a", {"class": "headerlink"}):
        s.decompose()

    # delete documentation of changes, not interesting to the LLM
    changes = m.find("section", {"id": "changes-since-cvl-1"})
    assert isinstance(changes, Tag)
    changes.decompose()

    main_body = m.find("div", {"itemprop": "articleBody"})
    assert isinstance(main_body, Tag), str(main_body)

    main_body_ctx.set(main_body)

    db = PostgreSQLRAGDatabase(DEFAULT_CONNECTION, get_model(), skip_test=False)
    buffer : List[BlockChunk] = []

    for s in main_body.select("div.compound > section"):
        assert isinstance(s, Tag)
        for t in translate_block(s, get_block_header(s)):
            sanity_checker(t)
            buffer.append(t)
            if len(buffer) == 50:
                db.add_chunks_batch(buffer)
                buffer = []

    if len(buffer) > 0:
        db.add_chunks_batch(buffer)

if __name__ == "__main__":
    main()

from typing import Iterator, Optional, Iterable
from dataclasses import dataclass

from composer.rag.types import BlockChunk
from composer.rag.text import code_ref_tag, get_code_refs

import spacy
from spacy.language import Language

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
    
@dataclass
class BuilderConfig:
    nlp: Language
    max_length: int

class BlockBuilder:
    def __init__(self, header: list[str], config: BuilderConfig) -> None:
        self.siblings : list[BlockChunk] = []
        self.text = ""
        self.code_refs : list[str] = []
        self.part_counter = 0
        self.headers = header
        self.appended_child = False
        self.config = config

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
        if len(new_text) < self.config.max_length:
            self.text = ctxt_string + new_text
            return
        if unbreakable:
            self.text = ctxt_string + new_text
            self._push()
            return
        doc = self.config.nlp(ctxt_string + new_text)
        for s in doc.sents:
            l = s.text.strip()
            if not l:
                continue
            self.text += l + " "
            if len(self.text) > self.config.max_length:
                self._push()
        return

    def append_text(self, txt: str, is_structured_boundary: bool, unbreakable: bool) -> None:
        self.appended_child = False
        new_len = len(txt) + len(self.text)
        if new_len < self.config.max_length:
            self.text += txt
            return

        if is_structured_boundary and not unbreakable:
            new_nlp = self.config.nlp(txt).sents
            first_sent = None
            for next_s in new_nlp:
                if next_s.text.strip():
                    continue
                first_sent = next_s.text.strip()
                break

            last_sent_of_curr : Optional[str] = None
            curr_nlp = self.config.nlp(self.text)
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
            for s in self.config.nlp(prev).sents:
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
            for s in self.config.nlp(txt).sents:
                l = s.text.strip()
                if not l:
                    continue
                curr_chunk += " " + l
                if len(curr_chunk) > self.config.max_length:
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
        first_sent = next(iter(self.config.nlp(c.chunk).sents))
        context = " / ".join([h for h in c.headers if len(h) > 0])
        self.text += context + "\n" + self._fixup_code_refs(first_sent.text, c.code_refs)
        self._push()

    def finish(self) -> Iterable[BlockChunk]:
        self._push()
        return self.siblings

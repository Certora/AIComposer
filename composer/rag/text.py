import re
from typing import Iterable, Tuple

def code_ref_tag(i: int) -> str:
    return f"<code-ref-{i}>"

_code_ref_re = r'<code-ref-(\d+)>'
_code_ref_matcher = re.compile(_code_ref_re)

def get_code_refs(s: str) -> Iterable[Tuple[str, int]]:
    for i in _code_ref_matcher.finditer(s):
        yield (i.group(0), int(i.group(1)))

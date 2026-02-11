import os
from typing import Iterator
from pathlib import Path

import uuid
import contextlib

@contextlib.contextmanager
def temp_certora_file(
    *,
    root: str,
    ext: str,
    content: str,
    prefix: str = "generated"
) -> Iterator[str]:
    tmp_name = f"{prefix}_{uuid.uuid1().hex[:16]}.{ext}"
    tgt = Path(root) / "certora" / tmp_name
    tgt.write_text(content)
    try:
        yield tmp_name
    finally:
        os.unlink(tgt)

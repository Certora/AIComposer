import contextlib
import hashlib
import os
import uuid
from pathlib import Path
from typing import Iterator


def string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@contextlib.contextmanager
def temp_certora_file(
    *,
    root: str,
    ext: str,
    content: str,
    prefix: str = "generated"
) -> Iterator[str]:
    """Write a temp file into the project's certora/ dir, yield its name, clean up."""
    tmp_name = f"{prefix}_{uuid.uuid1().hex[:16]}.{ext}"
    tgt = Path(root) / "certora" / tmp_name
    tgt.write_text(content)
    try:
        yield tmp_name
    finally:
        os.unlink(tgt)

FS_FORBIDDEN_READ = r"(^lib/.*)|(^\.certora_internal.*)|(^\.git.*)|(^test/.*)|(^emv-.*)|(.*\.json$)|(^node_modules/.*)"

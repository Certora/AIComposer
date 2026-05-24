import contextlib
import hashlib
import os
import re
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
    certora_dir = Path(root) / "certora"
    certora_dir.mkdir(exist_ok=True, parents=True)
    tgt = certora_dir / tmp_name
    tgt.write_text(content)
    try:
        yield tmp_name
    finally:
        os.unlink(tgt)

FS_FORBIDDEN_READ = r"(^lib/.*)|(^\.certora_internal.*)|(^\.git.*)|(^test/.*)|(^emv-.*)|(.*\.json$)|(^node_modules/.*(?<!\.sol)$)"


def find_files(project_root: Path, suffix: str) -> list[str]:
    """Recursively find files under ``project_root`` whose name ends with
    ``suffix``, skipping any path forbidden by ``FS_FORBIDDEN_READ``.

    Forbidden subtrees are pruned during traversal, not scanned then filtered.
    """
    forbidden = re.compile(FS_FORBIDDEN_READ)
    found: list[str] = []
    for root, dirs, files in project_root.walk():
        rel_root = root.relative_to(project_root)
        dirs[:] = [
            d for d in dirs
            if forbidden.fullmatch((rel_root / d / "x").as_posix()) is None
        ]
        for f in files:
            if not f.endswith(suffix):
                continue
            if forbidden.fullmatch((rel_root / f).as_posix()) is None:
                found.append(str(root / f))
    return sorted(found)


def uniq_thread_id(prefix: str) -> str:
    suff = uuid.uuid4().hex[:16]
    return f"{prefix}-{suff}"

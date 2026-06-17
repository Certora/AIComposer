import contextlib
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Iterator

from composer.spec.gen_types import CERTORA_DIR


def string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def slugify_filename(name: str) -> str:
    # Collapse any run of filesystem-unsafe characters into a single underscore so the
    # result is safe to use as a filename component; falls back to "unnamed" if empty.
    # Example: "transfer(address,uint256)" -> "transfer_address_uint256"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return slug or "unnamed"


def ensure_dir(path: Path) -> Path:
    """``mkdir -p`` *path* (no-op if it already exists) and return it, so it can be
    used inline, e.g. ``ensure_dir(certora_dir / "specs") / spec_name``."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def temp_certora_file(
    *,
    root: str,
    ext: str,
    content: str,
    prefix: str = "generated",
    name: str | None = None,
    dest_dir: Path = CERTORA_DIR,
) -> Iterator[str]:
    """Write a temp file under ``<root>/<dest_dir>``, yield its path **relative to
    the project root**, and clean it up.

    *dest_dir* is itself project-root-relative (default ``certora``). The yielded
    path uses the same project-root-relative convention as the persisted artifacts,
    so callers use it verbatim (no ``certora/`` prefixing). Materializing a spec in
    the same directory it will ultimately be dumped to (e.g. ``certora/specs``)
    makes the prover resolve the spec's CVL ``import`` statements identically at
    verify-time and after persistence.

    *name* (without extension) names the file ``<name>.<ext>`` verbatim instead of a
    unique ``<prefix>_<uid>.<ext>``. Since it is then not unique, callers passing
    *name* must serialize same-name use (the file is unlinked on exit).
    """
    tmp_name = f"{name}.{ext}" if name is not None else f"{prefix}_{uuid.uuid1().hex[:16]}.{ext}"
    target_dir = ensure_dir(Path(root) / dest_dir)
    tgt = target_dir / tmp_name
    tgt.write_text(content)
    try:
        yield (dest_dir / tmp_name).as_posix()
    finally:
        os.unlink(tgt)

FS_FORBIDDEN_READ = r"(^lib/.*)|(^\.certora_internal.*)|(^\.git.*)|(^test/.*)|(^emv-.*)|(.*\.json$)|(^node_modules/.*(?<!\.sol)$)"

def uniq_thread_id(prefix: str) -> str:
    suff = uuid.uuid4().hex[:16]
    return f"{prefix}-{suff}"

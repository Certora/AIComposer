import contextlib
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Iterator


def string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def slugify_filename(name: str) -> str:
    # Collapse any run of filesystem-unsafe characters into a single underscore so the
    # result is safe to use as a filename component; falls back to "unnamed" if empty.
    # Example: "transfer(address,uint256)" -> "transfer_address_uint256"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return slug or "unnamed"


@contextlib.contextmanager
def temp_certora_file(
    *,
    root: str,
    ext: str,
    content: str,
    prefix: str = "generated",
    dest_dir: str = "certora",
) -> Iterator[str]:
    """Write a temp file under ``<root>/<dest_dir>``, yield its path **relative to
    the project root**, and clean it up.

    *dest_dir* is itself project-root-relative (default ``certora``). The yielded
    path uses the same project-root-relative convention as the persisted artifacts,
    so callers use it verbatim (no ``certora/`` prefixing). Materializing a spec in
    the same directory it will ultimately be dumped to (e.g. ``certora/specs``)
    makes the prover resolve the spec's CVL ``import`` statements identically at
    verify-time and after persistence.
    """
    tmp_name = f"{prefix}_{uuid.uuid1().hex[:16]}.{ext}"
    target_dir = Path(root) / dest_dir
    target_dir.mkdir(exist_ok=True, parents=True)
    tgt = target_dir / tmp_name
    tgt.write_text(content)
    try:
        yield f"{dest_dir}/{tmp_name}"
    finally:
        os.unlink(tgt)

FS_FORBIDDEN_READ = r"(^lib/.*)|(^\.certora_internal.*)|(^\.git.*)|(^test/.*)|(^emv-.*)|(.*\.json$)|(^node_modules/.*(?<!\.sol)$)"

def uniq_thread_id(prefix: str) -> str:
    suff = uuid.uuid4().hex[:16]
    return f"{prefix}-{suff}"

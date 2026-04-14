"""
Cloud run source download and link resolution.

Resolves cloud run output URLs to rule names via the jobData API,
downloads source code from ZIP archives, extracts the .certora_sources
directory, filters out known library directories, and performs
content-based deduplication across multiple cloud runs.
"""

import asyncio
import io
import os
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse, urlencode, urlunparse

import aiohttp

from composer.prover.cloud import _NO_BROTLI_HEADERS, find_results_root
from composer.prover.results import (
    get_final_treeview, NoTreeViewResultError, MalformedTreeVew,
)


# ---------------------------------------------------------------------------
# Link resolution — output URL → rule metadata via jobData API
# ---------------------------------------------------------------------------

@dataclass
class ResolvedLink:
    """A cloud run output URL resolved to its rule metadata.

    ``explicit_rules`` come from the ``-rule`` parameter in jarSettings
    (high confidence — the run was specifically for these rules).
    ``included_rules`` come from scanning the treeView results archive
    (the run verified these rules, but may cover an entire spec file).
    """
    output_url: str
    explicit_rules: list[str] = field(default_factory=list)
    included_rules: list[str] = field(default_factory=list)
    contract: str = ""
    job_status: str = "UNKNOWN"


def _to_job_data_url(output_url: str) -> str | None:
    """Transform a cloud run output URL into its jobData API URL.

    Same transform as ``_to_zip_url`` but replaces ``/output/`` with
    ``/jobData/``.  Returns None if ``anonymousKey`` is missing.
    """
    parsed = urlparse(output_url)

    qs = parse_qs(parsed.query)
    keys = qs.get("anonymousKey")
    if not keys:
        return None

    new_path = parsed.path.replace("/output/", "/jobData/", 1)
    new_query = urlencode({"anonymousKey": keys[0]})
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        new_path,
        "",
        new_query,
        "",
    ))


def _to_zip_output_url(output_url: str) -> str | None:
    """Transform a cloud run output URL into its zipOutput download URL.

    Same transform as ``_to_job_data_url`` but replaces ``/output/`` with
    ``/zipOutput/``.  Returns None if ``anonymousKey`` is missing.
    """
    parsed = urlparse(output_url)

    qs = parse_qs(parsed.query)
    keys = qs.get("anonymousKey")
    if not keys:
        return None

    new_path = parsed.path.replace("/output/", "/zipOutput/", 1)
    new_query = urlencode({"anonymousKey": keys[0]})
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        new_path,
        "",
        new_query,
        "",
    ))


def _extract_rule_names(jar_settings: list[str]) -> list[str]:
    """Extract rule names from jarSettings by finding the ``-rule`` argument.

    The ``-rule`` value may be comma-separated (e.g. ``rule1,rule2``).
    """
    for i, arg in enumerate(jar_settings):
        if arg == "-rule" and i + 1 < len(jar_settings):
            return [r.strip() for r in jar_settings[i + 1].split(",") if r.strip()]
    return []


async def _scan_treeview_rules(
    output_url: str,
    sem: asyncio.Semaphore,
) -> list[str]:
    """Download zipOutput and scan treeView for rule names.

    Falls back gracefully — returns an empty list if anything goes wrong
    (download failure, malformed archive, missing treeView, etc.).
    """
    zip_output_url = _to_zip_output_url(output_url)
    if zip_output_url is None:
        return []

    try:
        async with sem:
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(
                headers=_NO_BROTLI_HEADERS,
                cookies={"certoraKey": os.environ["CERTORAKEY"]},
            ) as session:
                async with session.get(zip_output_url, timeout=timeout) as resp:
                    if resp.status != 200:
                        print(
                            f"Warning: zipOutput fetch returned {resp.status} "
                            f"for {output_url}",
                            file=sys.stderr,
                        )
                        return []
                    tar_bytes = await resp.read()

        with tempfile.TemporaryDirectory(prefix="certora_treeview_") as tmp_dir:
            tmp_path = Path(tmp_dir) / "output.tar.gz"
            tmp_path.write_bytes(tar_bytes)

            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=Path(tmp_dir))

            tmp_path.unlink()
            results_root = find_results_root(Path(tmp_dir))
            treeview, _ = get_final_treeview(results_root)
            return [r.name for r in treeview.rules]

    except (NoTreeViewResultError, MalformedTreeVew) as e:
        print(
            f"Warning: treeView scan failed for {output_url}: {e}",
            file=sys.stderr,
        )
        return []
    except Exception as e:
        print(
            f"Warning: zipOutput download/extraction failed for {output_url}: {e}",
            file=sys.stderr,
        )
        return []


async def resolve_cloud_links(
    output_urls: list[str],
    sem: asyncio.Semaphore,
) -> list[ResolvedLink]:
    """Resolve cloud run output URLs to rule metadata by fetching jobData.

    Fetches each URL's jobData JSON and extracts the rule name from
    the ``-rule`` argument in ``jarSettings``, plus the contract and
    job status.  URLs that fail to resolve are silently skipped.
    """
    unique_urls = list(dict.fromkeys(output_urls))

    async def _resolve_one(url: str) -> ResolvedLink | None:
        job_data_url = _to_job_data_url(url)
        if job_data_url is None:
            return None
        async with sem:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(headers=_NO_BROTLI_HEADERS) as session:
                async with session.get(job_data_url, timeout=timeout) as resp:
                    if resp.status != 200:
                        print(
                            f"Warning: jobData fetch returned {resp.status} for {url}",
                            file=sys.stderr,
                        )
                        return None
                    data = await resp.json()

        jar_settings = data.get("jarSettings", [])
        explicit_rules = _extract_rule_names(jar_settings)
        contract = data.get("contract", "")
        job_status = data.get("jobStatus", "UNKNOWN")

        # If no explicit -rule parameter, fall back to scanning the
        # treeView results archive for rule names.
        included_rules: list[str] = []
        if not explicit_rules:
            included_rules = await _scan_treeview_rules(url, sem)

        if not explicit_rules and not included_rules:
            return None

        return ResolvedLink(
            output_url=url,
            explicit_rules=explicit_rules,
            included_rules=included_rules,
            contract=contract,
            job_status=job_status,
        )

    results = await asyncio.gather(
        *(_resolve_one(u) for u in unique_urls)
    )
    return [r for r in results if r is not None]


def _to_zip_url(output_url: str) -> str | None:
    """Transform a cloud run output URL into its zipInput download URL.

    Replaces ``/output/`` with ``/zipInput/`` in the path and keeps only
    the ``anonymousKey`` query parameter.  Returns None (with a warning)
    if the ``anonymousKey`` parameter is missing.
    """
    parsed = urlparse(output_url)

    qs = parse_qs(parsed.query)
    keys = qs.get("anonymousKey")
    if not keys:
        print(
            f"Warning: no anonymousKey in cloud run URL, skipping: {output_url}",
            file=sys.stderr,
        )
        return None
    
    p = PurePosixPath(parsed.path).parts

    if len(p) != 4 or p[0] != "/" or p[1] != "output":
        return None
    
    job_id = p[3]
    new_query = urlencode({"anonymousKey": keys[0]})    
    final_url = f"{parsed.scheme}://{parsed.netloc}/v1/domain/jobs/{job_id}/f/inputs?{new_query}"

    return final_url

def _extract_sources(zip_bytes: bytes) -> dict[str, bytes]:
    """Extract files from the .certora_sources directory inside a ZIP.

    ``.certora_sources`` is always the top-level directory in the archive.
    Skips entries whose remaining path components start with '.' (e.g.
    ``.pre_autofinders.8/``).
    """
    result: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            parts = PurePosixPath(info.filename).parts
            if len(parts) < 2 or parts[0] != ".certora_sources":
                continue

            rel_parts = parts[1:]

            # Skip entries with dot-prefixed components
            if any(p.startswith(".") for p in rel_parts):
                continue

            rel_path = str(PurePosixPath(*rel_parts))
            result[rel_path] = zf.read(info.filename)

    return result


async def download_sources(
    url: str,
    dest: Path,
    sem: asyncio.Semaphore,
) -> str | None:
    """Download and extract sources for a single cloud run URL.

    Args:
        url: Cloud run output URL.
        dest: Directory to write extracted sources into.
        sem: Semaphore bounding concurrent downloads.

    Returns:
        Path to the source directory, or None if the URL lacks anonymousKey.
    """
    if dest.exists():
        return None

    zip_url = _to_zip_url(url)
    if zip_url is None:
        return "No zip url found"

    async with sem:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(headers=_NO_BROTLI_HEADERS) as session:
            async with session.get(zip_url, timeout=timeout) as resp:
                resp.raise_for_status()
                zip_bytes = await resp.read()
    
    full_source = _extract_sources(zip_bytes)

    dest.mkdir(parents=True, exist_ok=True)
    for rel_path, content in full_source.items():
        out = dest / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)

    return None

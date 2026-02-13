"""Cloud prover integration via anonymous key URLs.

Handles job submission polling, and result downloading for the Certora
cloud prover, using the anonymousKey parameter embedded in the job URL
to access job data without authentication.
"""

import asyncio
import logging
import tarfile
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable
from urllib.parse import urlparse, parse_qs

import aiohttp

logger = logging.getLogger("composer.spec")

# Terminal job statuses — anything not in this set means "still running"
_TERMINAL_STATUSES = frozenset({"DONE", "FAILED", "ERROR", "CANCELLED", "TIMEOUT"})


@dataclass
class CloudJob:
    """Parsed cloud prover job reference."""
    base_url: str       # e.g. "https://prover.certora.com"
    user_id: str
    job_id: str
    anonymous_key: str

    @property
    def job_data_url(self) -> str:
        return f"{self.base_url}/jobData/{self.user_id}/{self.job_id}?anonymousKey={self.anonymous_key}"


def parse_cloud_link(link: str) -> CloudJob:
    """Parse a CertoraRunResult.link URL into a CloudJob.

    Expected format:
        https://prover.certora.com/jobStatus/{user_id}/{job_id}?anonymousKey=...
    """
    parsed = urlparse(link)
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    # Expect: ["jobStatus", user_id, job_id]
    if len(parts) < 3 or parts[0] != "jobStatus":
        raise ValueError(f"Unexpected cloud link format: {link}")

    user_id = parts[1]
    job_id = parts[2]

    qs = parse_qs(parsed.query)
    keys = qs.get("anonymousKey", [])
    if not keys:
        raise ValueError(f"No anonymousKey in cloud link: {link}")

    return CloudJob(
        base_url=f"{parsed.scheme}://{parsed.netloc}",
        user_id=user_id,
        job_id=job_id,
        anonymous_key=keys[0],
    )


async def poll_job(
    job: CloudJob,
    *,
    interval: float = 10.0,
    timeout: float = 1800.0,
    on_status: Callable[[str], Awaitable[None]] | None = None,
) -> dict:
    """Poll /jobData until the job reaches a terminal status.

    Returns the full jobData JSON dict.
    Raises TimeoutError if the job doesn't finish within `timeout` seconds.
    """
    url = job.job_data_url
    elapsed = 0.0

    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.json()

            status = data.get("jobStatus", "UNKNOWN")

            if on_status is not None:
                await on_status(status)

            if status in _TERMINAL_STATUSES:
                return data

            if elapsed >= timeout:
                raise TimeoutError(
                    f"Cloud job {job.job_id} did not complete within {timeout}s (last status: {status})"
                )

            await asyncio.sleep(interval)
            elapsed += interval


def _find_results_root(dest: Path) -> Path:
    """Navigate past the extra TarName/ top-level directory in the extracted archive."""
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "Reports").is_dir():
        return children[0]

    if (dest / "Reports").is_dir():
        return dest

    raise RuntimeError(
        f"Could not find Reports/ in extracted results at {dest}. "
        f"Contents: {[p.name for p in dest.iterdir()]}"
    )


@asynccontextmanager
async def cloud_results(
    run_result_link: str,
    poll_callback: Callable[[str, str], Awaitable[None]] | None = None,
) -> AsyncIterator[Path]:
    """Async context manager: poll cloud job, download results, yield path, clean up.

    Parses the cloud link, polls until completion, downloads and extracts the
    results archive, then yields the path to the results root. The temporary
    directory is cleaned up on exit.
    """
    cloud_job = parse_cloud_link(run_result_link)

    logger.info("Cloud job submitted: %s/%s", cloud_job.user_id, cloud_job.job_id)

    async def on_status(status: str) -> None:
        logger.info("Cloud job %s status: %s", cloud_job.job_id[:8], status)
        if poll_callback:
            await poll_callback(status, f"Cloud job {cloud_job.job_id[:8]}: {status}")

    job_data = await poll_job(cloud_job, on_status=on_status)

    status = job_data.get("jobStatus", "UNKNOWN")
    if status != "DONE":
        raise RuntimeError(f"Cloud job finished with status {status} (expected DONE)")

    zip_url = job_data.get("zipOutputUrl")
    if not zip_url:
        raise RuntimeError("Cloud job completed but no zipOutputUrl in response")

    separator = "&" if "?" in zip_url else "?"
    full_url = f"{zip_url}{separator}anonymousKey={cloud_job.anonymous_key}"

    with tempfile.TemporaryDirectory(prefix="certora_cloud_") as tmp_dir:
        dest = Path(tmp_dir)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)

            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=dest)
        finally:
            tmp_path.unlink(missing_ok=True)

        yield _find_results_root(dest)

"""``forge test`` runner — the foundry analog of ``spec/source/prover.py``.

Exposes ``ForgeTestTool`` (and a convenience ``get_forge_test_tool`` factory)
that:

* Reads ``curr_test`` from injected state.
* Stages it into ``<project_root>/test/_composer_draft_<key>.t.sol`` for
  the duration of one ``forge test`` run, then deletes the staged file.
  ``<key>`` is either the agent-supplied ``seed`` arg (stable across
  invocations — replays foundry's ``cache/fuzz`` failure persistence) or
  a fresh UUID (unique per run — no replay).
* Runs ``forge test --json --match-path test/<staged>`` so only the
  draft's tests run and the result comes back as structured JSON.
* On a fully green run (excluding tests marked via
  ``expect_test_failure``), stamps ``validations[FORGE_TEST_VALIDATION_KEY]``
  ONLY when the run was unseeded. Seeded runs are a debugging aid for
  iterating on specific fuzz/invariant counterexamples; a clean unseeded
  run is what certifies the property for publish.
* On any unexpected failure or output that doesn't parse as JSON,
  returns the raw output and leaves ``validations`` untouched.

JSON shape (forge ≥ 0.2, stable across recent versions):

    {
      "<path>:<ContractName>": {
        "test_results": {
          "<signature(...)>": {
             "status": "Success" | "Failure" | "Skipped",
             "reason": null | str,
             ...other fields we don't use...
          },
          ...
        },
        ...
      },
      ...
    }
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from typing_extensions import TypedDict

from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import Field

from graphcore.graph import tool_state_update
from graphcore.tools.schemas import (
    WithAsyncDependencies, WithInjectedId, WithInjectedState,
)

from composer.ui.tool_display import tool_display

from composer.foundry.state import (
    FORGE_TEST_VALIDATION_KEY,
    FoundryGenerationState,
    make_foundry_validation_stamper,
)


# ---------------------------------------------------------------------------
# Deps + event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgeTestDeps:
    """Per-project bindings supplied at tool-construction time."""
    project_root: str
    sem: asyncio.Semaphore
    forge_binary: str = "forge"
    timeout_s: int = 600


class ForgeTestRunEvent(TypedDict):
    """Streamed summary of one forge-test invocation. Must be a TypedDict
    (not a pydantic BaseModel) so it round-trips through langgraph's
    custom-stream as a plain dict — ``run_graph`` asserts ``isinstance(
    payload, dict)`` on every stream item."""
    type: Literal["forge_test_run"]
    summary: str


@dataclass(frozen=True)
class _TestResult:
    name: str
    status: str
    reason: str | None


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


@tool_display("Running forge test", "Forge test result")
class ForgeTestTool(
    WithAsyncDependencies[Command | str, ForgeTestDeps],
    WithInjectedId,
    WithInjectedState[FoundryGenerationState],
):
    """
    Run the project's foundry test suite against your current draft.

    The current ``curr_test`` buffer is written into
    ``<project_root>/test/`` as a ``.t.sol`` file, then
    ``forge test --json --match-path test/<that>`` runs.

    ``seed`` controls the staged filename. If you provide a seed, the
    file is staged at ``test/_composer_draft_<seed>.t.sol`` — the SAME
    path across invocations — so foundry's ``cache/fuzz`` persistence
    finds prior failing inputs and replays them first. This is how you
    iterate on a specific fuzz / invariant counterexample.

    If you omit ``seed``, the file is staged under a fresh UUID — a path
    foundry has never seen, so the persistence cache misses and the
    fuzzer runs a brand-new campaign with no carryover.

    The publish gate is ONLY stamped by an unseeded (no-``seed``) run.
    A seeded run is a debugging aid; a clean unseeded run is the
    certification. The intended loop is:

    1. Author / revise tests, call ``forge_test`` without ``seed``.
    2. If something fails, call ``forge_test(seed="x")`` repeatedly with
       the SAME ``seed`` to zero in on the counterexample (each replays
       the cached failing inputs first, so progress on the specific bug
       is monotonic).
    3. Once the seeded run is clean, call ``forge_test`` once more
       without ``seed`` to stamp the publish gate.
    """
    seed: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_]{1,32}$",
        description=(
            "Optional debug seed identifying the staged test file. When "
            "set, the file is staged at a stable path so foundry's "
            "fuzz-failure cache replays prior counterexamples. When "
            "omitted, a fresh UUID is used and the run is eligible to "
            "stamp the publish gate."
        ),
    )

    @override
    async def run(self) -> Command | str:
        if self.state["curr_test"] is None:
            return "No test written yet. Call put_test_raw before forge_test."
        
        with self.tool_deps() as deps:
            async with deps.sem:
                root = Path(deps.project_root).resolve()
                if not (root / "foundry.toml").is_file():
                    return (
                        f"forge_test cannot run: {root}/foundry.toml does not "
                        "exist. The project does not look like a foundry project."
                    )
                test_dir = root / "test"
                test_dir.mkdir(exist_ok=True)

                # ``seed`` controls staged filename; an unseeded run picks a
                # fresh UUID. Only the unseeded variant is eligible to stamp.
                path_key = self.seed if self.seed is not None else uuid.uuid4().hex[:12]
                seeded = self.seed is not None
                staged_name = f"_composer_draft_{path_key}.t.sol"
                staged = test_dir / staged_name
                staged.write_text(self.state["curr_test"])

                try:
                    proc = await asyncio.create_subprocess_exec(
                        deps.forge_binary, "test", "--json",
                        "--match-path", f"test/{staged_name}",
                        cwd=str(root),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        stdout_b, stderr_b = await asyncio.wait_for(
                            proc.communicate(), timeout=deps.timeout_s,
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        return f"forge test timed out after {deps.timeout_s}s"
                    stdout = stdout_b.decode(errors="replace")
                    stderr = stderr_b.decode(errors="replace")
                    returncode = proc.returncode if proc.returncode is not None else -1
                except FileNotFoundError:
                    return f"`{deps.forge_binary}` not found on PATH"
                finally:
                    try:
                        staged.unlink()
                    except OSError:
                        pass

        # Every response surfaces ``path_key`` as the seed the agent
        # should pass back to iterate against this specific campaign's
        # cached failures. Without this the auto-UUID we picked for an
        # unseeded run is invisible, and the agent has no way to drive
        # the cache/fuzz replay loop on a failure.
        seed_footer = (
            f"\n\n(seed for this run: {path_key!r} — pass "
            f"`forge_test(seed={path_key!r})` to iterate against this "
            "campaign's cached failures.)"
        )

        results = _parse_forge_json(stdout)
        if results is None:
            # Most likely a compile failure: forge didn't reach the test
            # runner. Surface the raw output so the agent sees the solc /
            # linker error. No tests ran, and any previously recorded test
            # names no longer describe a runnable buffer — clear them.
            msg = (
                f"forge test did not produce parseable JSON "
                f"(exit {returncode}). This usually means the project "
                "failed to build.\n\n"
                f"stderr:\n{stderr}\n\nstdout:\n{stdout}"
                f"{seed_footer}"
            )
            get_stream_writer()(
                ForgeTestRunEvent(type="forge_test_run", summary=msg)
            )
            return tool_state_update(
                tool_call_id=self.tool_call_id,
                content=msg,
                last_test_names=[],
            )

        expected_failures = self.state["expected_failures"]
        unexpected_failures = [
            r for r in results
            if r.status == "Failure" and r.name not in expected_failures
        ]
        unexpected_passes = [
            r for r in results
            if r.status == "Success" and r.name in expected_failures
        ]

        summary = _format_summary(results, expected_failures)
        get_stream_writer()(
            ForgeTestRunEvent(type="forge_test_run", summary=summary)
        )

        clean = (
            returncode == 0
            and not unexpected_failures
            and not unexpected_passes
        )

        # Every run that produced results records the names of the tests
        # that ran. The publish gate validates the declared property→test
        # mapping against this ground truth instead of trusting the agent's
        # transcription of its own test names.
        test_names = [r.name for r in results]

        if clean and not seeded:
            stamper = make_foundry_validation_stamper(FORGE_TEST_VALIDATION_KEY)
            return tool_state_update(
                tool_call_id=self.tool_call_id,
                content=(
                    f"All tests passed (publish gate stamped).\n\n{summary}"
                ),
                validations=stamper(self.state),
                last_test_names=test_names,
            )

        if clean and seeded:
            return tool_state_update(
                tool_call_id=self.tool_call_id,
                content=(
                    f"All tests passed under seed={self.seed!r} — the "
                    "fuzz-failure cache replayed prior counterexamples and "
                    "they no longer trigger. Call forge_test WITHOUT a seed "
                    "to run a fresh campaign and stamp the publish gate.\n\n"
                    f"{summary}"
                ),
                last_test_names=test_names,
            )

        problems: list[str] = []
        if unexpected_failures:
            problems.append(
                "Unexpected failures: "
                + ", ".join(r.name for r in unexpected_failures)
            )
        if unexpected_passes:
            problems.append(
                "Tests marked expect_test_failure that actually passed (call "
                "expect_test_passage to clear the marker, or rework the test): "
                + ", ".join(r.name for r in unexpected_passes)
            )
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content=(
                "forge test did not produce a clean run.\n"
                + "\n".join(problems)
                + f"\n\n{summary}"
                + seed_footer
            ),
            last_test_names=test_names,
        )


def get_forge_test_tool(
    project_root: str,
    forge_sem: asyncio.Semaphore,
    *,
    forge_binary: str = "forge",
    timeout_s: int = 600,
) -> BaseTool:
    """Convenience factory: build the ``forge_test`` tool bound to one
    foundry project. Equivalent to constructing ``ForgeTestDeps`` directly
    and calling ``ForgeTestTool.bind(deps).as_tool("forge_test")``."""
    deps = ForgeTestDeps(
        project_root=project_root,
        forge_binary=forge_binary,
        timeout_s=timeout_s,
        sem=forge_sem
    )
    return ForgeTestTool.bind(deps).as_tool("forge_test")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_forge_json(stdout: str) -> list[_TestResult] | None:
    """Parse forge's ``--json`` output into a flat list of ``_TestResult``.

    Returns ``None`` if the output isn't JSON (compile error / runner
    crash). Otherwise returns one entry per ``(contract, signature)``
    pair regardless of how many contracts were exercised in the run.

    Test names are the function identifier portion of the JSON key
    (``"test_Foo(uint256)"`` → ``"test_Foo"``) — that's what the agent
    uses in ``expect_test_failure`` and in the property→test mapping,
    and it's stable across argument shapes (fuzz tests with parametric
    types still get the same name).
    """
    text = stdout.strip()
    if not text:
        return None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(doc, dict):
        return None

    out: list[_TestResult] = []
    for contract_block in doc.values():
        if not isinstance(contract_block, dict):
            continue
        tests = contract_block.get("test_results")
        if not isinstance(tests, dict):
            continue
        for signature, body in tests.items():
            if not isinstance(body, dict):
                continue
            name = signature.split("(", 1)[0].strip()
            status = str(body.get("status", "")).strip() or "Unknown"
            reason_raw = body.get("reason")
            reason = str(reason_raw) if reason_raw is not None else None
            out.append(_TestResult(name=name, status=status, reason=reason))
    return out


def _format_summary(results: list[_TestResult], expected_failures: dict[str, str]) -> str:
    """Render a compact human-readable summary of the JSON results."""
    if not results:
        return "(no tests reported)"
    passed: list[str] = []
    failed: list[_TestResult] = []
    skipped_by_forge: list[str] = []
    for r in results:
        match r.status:
            case "Success":
                passed.append(r.name)
            case "Failure":
                failed.append(r)
            case _:
                skipped_by_forge.append(r.name)
    lines = [f"forge test: {len(passed)} passed, {len(failed)} failed"]
    if skipped_by_forge:
        lines.append(f"  (forge-skipped: {', '.join(skipped_by_forge)})")
    for r in failed:
        marker = " [expected-fail]" if r.name in expected_failures else ""
        reason = f" — {r.reason}" if r.reason else ""
        lines.append(f"  FAIL {r.name}{marker}{reason}")
    return "\n".join(lines)

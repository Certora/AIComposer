"""Entry point for the foundry test-generation pipeline — console (no TUI) mode."""

import asyncio

import composer.bind as _

from composer.diagnostics.timing import RunSummary
from composer.foundry.entry import _entry_point
from composer.ui.foundry_console import FoundryConsoleHandler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> int:
    summary = RunSummary()
    async with _entry_point(summary) as run:
        result = await run(FoundryConsoleHandler().make_handler)
        print(f"\n{'=' * 60}")
        print(summary.format())
        print(f"\n  Components:    {result.n_components}")
        print(f"  Properties:    {result.n_properties}")
        print(f"  Tests written: {len(result.written)}")
        for p in result.written:
            print(f"    - {p}")
        if result.failures:
            print(f"  Failures:      {len(result.failures)}")
            for f in result.failures:
                print(f"    - {f}")
        print(f"{'=' * 60}")
        return 0


def main() -> int:
    return asyncio.run(_main())

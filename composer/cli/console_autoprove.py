"""Entry point for the auto-prove pipeline — console (no TUI) mode."""

import asyncio

import composer.bind as _

from composer.ui.autoprove_console import AutoProveConsoleHandler
from composer.spec.source.autoprove_common import _entry_point


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> int:
    async with _entry_point() as run:
        result = await run(AutoProveConsoleHandler().make_handler)
        print(f"\n{'=' * 60}")
        print("Auto-prove complete")
        print(f"  Components:  {result.n_components}")
        print(f"  Properties:  {result.n_properties}")
        if result.failures:
            print(f"  Failures:    {len(result.failures)}")
            for f in result.failures:
                print(f"    - {f}")
        print(f"{'=' * 60}")
        return 0


def main() -> int:
    return asyncio.run(_main())


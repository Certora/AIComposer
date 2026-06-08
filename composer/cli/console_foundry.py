"""Entry point for the foundry test-generation pipeline — console (no TUI) mode.

Parallel to ``composer/cli/console_autoprove.py`` but routes through
``composer.foundry.entry._entry_point``. For now this uses the noisy
debug handler from ``composer.foundry.noisy_handler`` — the autoprove
console handler is shaped around prover events / autoprove-specific
phases and would not render the foundry workflow's state transitions
usefully. The noisy handler dumps everything to stderr; replace once a
real foundry UI exists.
"""

import asyncio

import composer.bind as _

from composer.diagnostics.timing import RunSummary
from composer.foundry.entry import _entry_point
from composer.foundry.noisy_handler import noisy_handler_factory


async def _main() -> int:
    summary = RunSummary()
    async with _entry_point(summary) as run:
        result = await run(noisy_handler_factory())
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

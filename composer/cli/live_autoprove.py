"""Entry point for the auto-prove pipeline — live-display mode.

Renders the pipeline as a single inline ``rich.live.Live`` region
showing the tree of active agents (one root per phase task, sub-agents
nested below). Scrollback above the Live region captures phase
boundaries, prover lifecycle terminals, and per-rule analysis events.

Sister to ``tui-autoprove`` (Textual full-screen TUI) and
``console-autoprove`` (plain ``print``).
"""

import asyncio

import composer.bind as _

from composer.spec.source.autoprove_common import _entry_point
from composer.ui.autoprove_live import AutoProveLiveHandler


async def _main() -> int:
    async with _entry_point() as exec:
        async with AutoProveLiveHandler() as handler:
            result = await exec(handler.make_handler)
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

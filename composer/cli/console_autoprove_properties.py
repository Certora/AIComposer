"""Entry point for the known-properties auto-prove pipeline — console mode."""

import asyncio

import composer.bind as _

from composer.diagnostics.timing import RunSummary
from composer.ui.autoprove_console import AutoProveConsoleHandler
from composer.spec.source.autoprove_common import _properties_entry_point
from composer.spec.source.common_pipeline import Unmatched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> int:
    summary = RunSummary()
    async with _properties_entry_point(summary) as run:
        result = await run(AutoProveConsoleHandler().make_handler)
        print(f"\n{'=' * 60}")
        print(summary.format())
        print(f"\n  Components:  {result.n_components}")
        print(f"  Properties:  {result.n_properties}")
        if result.failures:
            print(f"  Failures:    {len(result.failures)}")
            for f in result.failures:
                print(f"    - {f}")
        # Surface — loudly — every input property that did NOT result in a
        # verified rule (formalize-phase unmatched + CVL-phase skips/gave-up).
        # See certora/properties/uncovered_properties.json for the full report.
        if result.uncovered:
            print(f"\n  {'!' * 56}")
            print(f"  UNCOVERED PROPERTIES: {len(result.uncovered)} input "
                  "propert(y/ies) did NOT result in a verified rule:")
            for u in result.uncovered:
                where = (
                    "unmatched (no component)"
                    if isinstance(u.reason, Unmatched)
                    else u.reason.feat.component.name
                )
                print(f"    - {u.property_id} [{where}]: {u.reason}")
            print("  See certora/properties/uncovered_properties.json")
            print(f"  {'!' * 56}")
        print(f"{'=' * 60}")
        return 0


def main() -> int:
    return asyncio.run(_main())

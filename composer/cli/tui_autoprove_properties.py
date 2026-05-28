"""Entry point for the properties-driven auto-prove pipeline TUI."""

import asyncio

import composer.bind as _

from composer.ui.autoprove_app import AutoProveApp
from composer.spec.source.autoprove_properties_common import _entry_point


async def _main() -> int:
    async with _entry_point() as pipeline:
        app = AutoProveApp()

        async def work():
            try:
                result = await pipeline(app.make_handler)
                summary = (
                    f"Auto-prove (properties) complete: {result.n_properties} properties"
                )
                if result.failures:
                    summary += f", {len(result.failures)} failures"
                app.notify(summary)
                app._pipeline_done = True
            except Exception as exc:
                app.notify(f"Pipeline failed: {exc}", severity="error")
                app._pipeline_done = True

        app.set_work(work)
        await app.run_async()
        return 0


def main() -> int:
    return asyncio.run(_main())

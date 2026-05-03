"""DEPRECATED. Use ``tui-codegen`` (registered in ``[project.scripts]``).

This shim continues to work — it forwards to ``composer.cli.tui_codegen``,
which carries the same parser surface and runs the same workflow against
``CodeGenRichApp``. New code, scripts, and harnesses should reach for
``tui-codegen`` (or import ``composer.cli.tui_codegen`` directly).
"""

import asyncio
import sys
import warnings

from composer.cli.tui_codegen import run as _run


warnings.warn(
    "tui_main.py is deprecated; use the `tui-codegen` console script "
    "or import `composer.cli.tui_codegen` instead.",
    DeprecationWarning,
    stacklevel=2,
)


async def main() -> int:
    """Backwards-compat async entry. Delegates to ``composer.cli.tui_codegen.run``."""
    return await _run()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

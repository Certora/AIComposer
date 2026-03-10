import composer.certora as _

import asyncio
from typing import cast

from composer.input.types import ModelOptions, RAGDBOptions, LangraphOptions
from composer.input.parsing import add_protocol_args
from composer.io.natspec_rich import NatSpecRichApp
from composer.io.ide_bridge import IDEBridge
from composer.spec.natspec import execute, NatSpecArgs

import argparse


async def main() -> int:
    """TUI entry point for the NatSpec generation tool."""
    parser = argparse.ArgumentParser(usage="Generate a CVL from a natural language design doc.")
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, LangraphOptions)
    parser.add_argument("input_file", help="The input file to use for the spec generation")
    parser.add_argument("--show-checkpoints", action="store_true",
                        help="Show checkpoint IDs inline in the event log")

    args = cast(NatSpecArgs, parser.parse_args())

    ide = await IDEBridge.connect()

    app = NatSpecRichApp(show_checkpoints=args.show_checkpoints, ide=ide)  # type: ignore

    exit_code = 1

    async def work():
        nonlocal exit_code
        exit_code = await execute(args, handler=app)

    app.set_work(work)
    await app.run_async()

    if ide is not None:
        await ide.close()

    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

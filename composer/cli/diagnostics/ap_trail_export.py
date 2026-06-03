"""``ap-trail export`` — dump a run's full forest + per-thread timelines to a
self-contained gzipped JSON file consumable by ``ap-trail view --from-export``."""

import argparse
import asyncio
import sys

from composer.io.run_index import build_export, write_export
from composer.workflow.services import checkpointer_context, store_context
from .uid_bind import bind_uid_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id", help="Run id to export (as shown by `ap-trail ls`).")
    bind_uid_args(parser)
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path. Defaults to <run_id>.json.gz in the current directory.",
    )


async def _main(args: argparse.Namespace) -> int:
    output_path = args.output or f"{args.run_id}.json.gz"

    async with store_context() as store, checkpointer_context() as checkpointer:
        try:
            exported = await build_export(store, checkpointer, args.run_id, uid=args.uid)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    size = write_export(exported, output_path)
    n_threads = len(exported.threads)
    n_items = sum(len(t.timeline) for t in exported.threads)
    print(
        f"Exported run {args.run_id}: {n_threads} thread(s), {n_items} timeline item(s), "
        f"{size:,} bytes compressed -> {output_path}"
    )
    return 0


def main(args: argparse.Namespace) -> int:
    return asyncio.run(_main(args))

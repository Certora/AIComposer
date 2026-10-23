"""``ap-trail data`` — show the ``run_data`` metadata dicts recorded for a run.

These are the arbitrary per-run records written via the ``RunDataLogger`` yielded
by ``thread_logger`` (e.g. ``token_usage``), distinct from the ``RunMeta.tags``
shown by ``ap-trail ls``.

With ``--json`` the command emits pure JSON on stdout (nothing else), so it can be
piped into ``jq`` and friends; diagnostics go to stderr. Pass an optional ``key``
to target a single metadata record instead of the whole mapping.
"""

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from composer.io.run_index import get_run, get_run_data, list_run_data
from composer.workflow.services import store_context
from .uid_bind import bind_uid_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id", help="Run id to show recorded data for (as shown by `ap-trail ls`).")
    parser.add_argument(
        "key",
        nargs="?",
        default=None,
        help="Optional metadata key to show on its own (e.g. `token_usage`). Omit for all.",
    )
    bind_uid_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit pure JSON on stdout (pipeable); diagnostics go to stderr.",
    )


async def _main(args: argparse.Namespace) -> int:
    async with store_context() as store:
        run = await get_run(store, args.run_id, uid=args.uid)
        if run is None:
            print(f"No such run: {args.run_id}", file=sys.stderr)
            return 1

        if args.key is not None:
            value = await get_run_data(store, args.run_id, args.key, uid=args.uid)
            if value is None:
                available = [k for k, _ in await list_run_data(store, args.run_id, uid=args.uid)]
                print(
                    f"No run data '{args.key}' for run {args.run_id}. "
                    f"Available: {', '.join(available) or '(none)'}",
                    file=sys.stderr,
                )
                return 1
            entries = [(args.key, value)]
        else:
            entries = await list_run_data(store, args.run_id, uid=args.uid)

    if args.json:
        # Single key → that dict; otherwise the full {key: dict} mapping.
        payload = entries[0][1] if args.key is not None else dict(entries)
        print(json.dumps(payload, indent=2))
        return 0

    console = Console()
    if not entries:
        console.print(f"[dim]No run data recorded for run {args.run_id}.[/dim]")
        return 0
    for key, value in entries:
        console.print(Panel(JSON.from_data(value), title=key, title_align="left", border_style="cyan"))
    return 0


def main(args: argparse.Namespace) -> int:
    return asyncio.run(_main(args))

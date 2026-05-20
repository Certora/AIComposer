"""Standalone exporter for the natspec property + spec report.

Re-derives the same markdown report ``run_natspec_pipeline`` produces
at the end of an interactive run, but does it from the cached pipeline
state — useful for inspecting a previously-run pipeline whose output
folder is gone, or for diffing reports between runs.

Wired as ``natspec-report`` in ``pyproject.toml``.

Usage::

    natspec-report <input_file> --cache-ns <ns> [--from-source]
                                [--output <path>]

The input document feeds the digest that scopes the cache namespace,
``--cache-ns`` is the user-chosen prefix the pipeline ran under, and
``--from-source`` selects the right ``Application`` subclass for cache
lookups (greenfield vs from-source). Memory namespace isn't relevant —
the walk is pure cache reads.

Without ``--output`` the report goes to stdout. With ``--output``, it's
written to the named file (parent dirs created as needed).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from composer.spec.context import (
    WorkflowContext,
    get_document_input,
)
from composer.spec.natspec.report import (
    REPORT_FILENAME,
    report_from_cache,
    report_to_markdown,
)
from composer.spec.util import string_hash
from composer.ui.cache_explorer import DummyServices
from composer.workflow.services import get_store


async def _amain() -> int:
    parser = argparse.ArgumentParser(
        description="Export a natspec property + spec report from cached pipeline state."
    )
    parser.add_argument("input_file", help="Path to the design document (text or PDF)")
    parser.add_argument(
        "--cache-ns", required=True, dest="cache_ns",
        help="Cache namespace (same value passed to tui-natspec).",
    )
    parser.add_argument(
        "--from-source", action="store_true",
        help="Set if the original pipeline run was invoked with --source-root "
             "(selects the FromSourceApplication model for cache lookups). "
             "Omit for greenfield runs.",
    )
    parser.add_argument(
        "--output", dest="output", default=None,
        help=f"Output file path. If a directory, writes ``{REPORT_FILENAME}`` "
             "inside it. Omitted: prints to stdout.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    content = get_document_input(input_path)
    if content is None:
        print(f"Error: cannot read {input_path}", file=sys.stderr)
        return 1

    store = get_store()
    doc_digest = string_hash(str(content))
    root_ns = (args.cache_ns, doc_digest)

    root_ctx: WorkflowContext = WorkflowContext.create(
        services=DummyServices(),  # type: ignore[arg-type]
        thread_id="report-exporter",
        store=store,
        memory_namespace=None,
        cache_namespace=root_ns,
    )

    report = await report_from_cache(root_ctx, from_source=args.from_source)
    if report is None:
        print(
            f"Error: no cached source-analysis under namespace {root_ns}. "
            "Was the pipeline run with this --cache-ns?",
            file=sys.stderr,
        )
        return 1

    md = report_to_markdown(report)

    if args.output is None:
        print(md)
        return 0

    out_path = Path(args.output).expanduser().resolve()
    if out_path.is_dir() or args.output.endswith("/"):
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / REPORT_FILENAME
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote {out_path}")
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Inspect messages from an AIAutoProver run.

The events.jsonl produced by autoprove logging records thread paths but
not message content. The real LLM messages, tool calls, and tool results
live in Postgres (langgraph_checkpoint_db). This script bridges the two.

Subcommands:
  summary <log-path>                 - list threads & basic stats for a run
  messages <log-path> [flags]        - print messages from a thread
  message <log-path> <idx> [flags]   - print one message in full

<log-path> can be a .events.jsonl file, a .log file, the autoProve folder,
or a project root (a .certora_internal/autoProve subfolder will be located
automatically).
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


# ------------------------- path resolution -------------------------

def resolve_events_file(log_path: str) -> Path:
    p = Path(log_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        if p.name.endswith(".events.jsonl"):
            return p
        if p.name.endswith(".log"):
            sibling = p.with_name(p.name[:-len(".log")] + ".events.jsonl")
            if sibling.exists():
                return sibling
            raise FileNotFoundError(f"No matching .events.jsonl next to {p}")
        raise ValueError(f"Don't know how to handle file: {p}")

    direct = sorted(p.glob("*.events.jsonl"))
    if direct:
        candidates = direct
    else:
        candidates = sorted((p / ".certora_internal" / "autoProve").glob("*.events.jsonl"))

    if not candidates:
        raise FileNotFoundError(f"No .events.jsonl files found under {p}")

    chosen = candidates[-1]
    if len(candidates) > 1:
        print(
            f"note: {len(candidates)} runs in folder; using newest: {chosen.name}",
            file=sys.stderr,
        )
    return chosen


# ------------------------- event walking -------------------------

def walk_threads(events_file: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    """
    Walk the events.jsonl and return one entry per unique thread path.
    Entry contains: leaf thread_id, optional description (from start events),
    and a stable order key.
    """
    by_path: dict[tuple[str, ...], dict[str, Any]] = {}
    order = 0
    with events_file.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_path = e.get("path") or []
            path: tuple[str, ...] = tuple(str(x) for x in raw_path)
            if not path:
                continue
            if path not in by_path:
                order += 1
                by_path[path] = {
                    "thread_id": path[-1],
                    "depth": len(path),
                    "parent": path[-2] if len(path) > 1 else None,
                    "description": None,
                    "order": order,
                }
            if e.get("kind") == "start" and not by_path[path]["description"]:
                by_path[path]["description"] = e.get("description")
    return by_path


# ------------------------- checkpoint access -------------------------

def _ensure_composer_import() -> None:
    """
    Make sure `composer.workflow.services` is importable. We expect the
    script to be invoked with the appropriate venv Python; if not, surface
    the import error clearly so the user can re-run with the right one.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "composer").is_dir() and (parent / "composer" / "workflow").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            break


async def fetch_messages(thread_id: str) -> list[Any]:
    _ensure_composer_import()
    from composer.workflow.services import get_async_checkpointer
    from langchain_core.runnables import RunnableConfig
    saver = await get_async_checkpointer()
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    tup = await saver.aget_tuple(cfg)
    if tup is None:
        return []
    return list(tup.checkpoint.get("channel_values", {}).get("messages", []))


# ------------------------- run-index access (run_id) -------------------------
#
# Newer runs are registered in a Postgres-backed run index (composer.io.run_index
# / thread_logging) keyed by RunSummary.run_id. This lets us resolve a whole
# run's threads from a run_id alone — no events.jsonl needed. The `ap-trail`
# CLI (`ap-trail ls` / `view <run_id>` / `export <run_id>`) is the richer,
# interactive view of the same data; this is the message-level companion.

_RUN_ID_RE = re.compile(r"[0-9a-fA-F]{32}\Z")


async def _get_store() -> Any:
    _ensure_composer_import()
    from composer.workflow.services import get_async_store
    return await get_async_store()


async def list_recent_runs(limit: int, uid: str | None) -> list[tuple[str, Any]]:
    from composer.io.run_index import list_runs
    store = await _get_store()
    return await list_runs(store, limit=limit, uid=uid)


async def fetch_run_catalog(run_id: str, uid: str | None) -> dict[tuple[str, ...], dict[str, Any]]:
    """Build the same thread-catalog shape as ``walk_threads`` from a run_id,
    using the run index. A thread with no ``from_tool_id`` is a top-level phase
    execution; otherwise it's a subagent spawned by that tool call."""
    from composer.io.run_index import list_threads_for_run
    store = await _get_store()
    threads = await list_threads_for_run(store, run_id, uid=uid)
    if not threads:
        raise RuntimeError(
            f"No threads found for run_id {run_id!r}. List runs with "
            f"`inspect_run.py runs` (or check the uid with --uid)."
        )
    by_path: dict[tuple[str, ...], dict[str, Any]] = {}
    for order, (thread_run_id, meta) in enumerate(threads, start=1):
        tool = meta.get("from_tool_id")
        by_path[(thread_run_id,)] = {
            "thread_id": meta["thread_id"],
            "depth": 1 if tool is None else 2,
            "parent": (f"tool:{tool}" if tool else None),
            "description": meta.get("description"),
            "order": order,
        }
    return by_path


async def resolve_catalog(args: argparse.Namespace) -> tuple[dict[tuple[str, ...], dict[str, Any]], str]:
    """Return ``(thread_catalog, source_label)`` from either ``--run-id`` /
    a run_id passed positionally, or an events.jsonl path."""
    run_id = getattr(args, "run_id", None)
    log_path = getattr(args, "log_path", None)
    if run_id is None and log_path and _RUN_ID_RE.match(log_path) and not Path(log_path).expanduser().exists():
        run_id = log_path  # positional looks like a run_id, not a path
    if run_id:
        return await fetch_run_catalog(run_id, getattr(args, "uid", None)), f"run_id={run_id}"
    if not log_path:
        raise SystemExit(
            "Provide an events.jsonl/.log/autoProve path, or a run_id "
            "(positionally or via --run-id). Use `inspect_run.py runs` to list run_ids."
        )
    events = resolve_events_file(log_path)
    return walk_threads(events), f"events={events}"


# ------------------------- message helpers -------------------------

def _tool_names(m: Any) -> list[str]:
    tcs = getattr(m, "tool_calls", None) or []
    out: list[str] = []
    for tc in tcs:
        if isinstance(tc, dict):
            n = tc.get("name")
        else:
            n = getattr(tc, "name", None)
        if n:
            out.append(n)
    return out


def _content_text(m: Any) -> str:
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for blk in c:
            if isinstance(blk, dict):
                parts.append(blk.get("text") or json.dumps(blk)[:500])
            else:
                parts.append(str(blk))
        return "\n".join(parts)
    return str(c)


def _is_tool_error(m: Any) -> bool:
    if type(m).__name__ != "ToolMessage":
        return False
    t = _content_text(m)
    return (
        t.startswith("Error:")
        or "Error invoking tool" in t
        or "ToolInvocationError" in t
        or "ValidationError" in t
    )


def _short_preview(text: str, n: int = 240) -> str:
    one = text.replace("\n", " ⏎ ").strip()
    if len(one) <= n:
        return one
    return one[:n] + " …"


def _msg_oneline(i: int, m: Any) -> str:
    t = type(m).__name__
    meta = getattr(m, "response_metadata", {}) or {}
    usage = getattr(m, "usage_metadata", {}) or {}
    stop = meta.get("stop_reason") or meta.get("finish_reason")
    tools = _tool_names(m)
    err = _is_tool_error(m)

    bits = [f"[{i:3d}] {t}"]
    if stop:
        bits.append(f"stop={stop}")
    if usage:
        ti, to = usage.get("input_tokens"), usage.get("output_tokens")
        if ti is not None or to is not None:
            bits.append(f"tok in={ti} out={to}")
    if tools:
        bits.append(f"tools={tools}")
    if err:
        bits.append("ERROR")

    head = " ".join(bits)
    preview = _short_preview(_content_text(m))
    return f"{head}\n     {preview}" if preview else head


def _msg_full(i: int, m: Any) -> str:
    t = type(m).__name__
    meta = getattr(m, "response_metadata", {}) or {}
    usage = getattr(m, "usage_metadata", {}) or {}
    tcs = getattr(m, "tool_calls", None) or []
    lines = [f"========== [{i}] {t} ==========",
             f"response_metadata: {json.dumps(meta, default=str)[:1000]}",
             f"usage_metadata:    {json.dumps(usage, default=str)}",
             "",
             "content:",
             _content_text(m),
             ""]
    if tcs:
        lines.append("tool_calls:")
        for j, tc in enumerate(tcs):
            d = tc if isinstance(tc, dict) else getattr(tc, "__dict__", {"repr": str(tc)})
            lines.append(f"  [{j}] {json.dumps(d, default=str, indent=2)}")
    return "\n".join(lines)


# ------------------------- filtering -------------------------

def _parse_range(spec: str, n: int) -> tuple[int, int]:
    a, _, b = spec.partition(":")
    lo = int(a) if a else 0
    hi = int(b) if b else n
    return max(0, lo), min(n, hi)


def _select_messages(msgs: list[Any], args: argparse.Namespace) -> Iterable[tuple[int, Any]]:
    lo, hi = _parse_range(args.range, len(msgs)) if args.range else (0, len(msgs))
    for i, m in enumerate(msgs):
        if i < lo or i >= hi:
            continue
        if args.type and type(m).__name__ != args.type:
            continue
        if args.errors_only and not _is_tool_error(m):
            continue
        if args.tool:
            invokes = args.tool in _tool_names(m)
            is_result = type(m).__name__ == "ToolMessage" and getattr(m, "name", None) == args.tool
            if not (invokes or is_result):
                continue
        yield i, m


# ------------------------- subcommands -------------------------

async def cmd_summary(args: argparse.Namespace) -> int:
    threads, source = await resolve_catalog(args)

    top_executions = sorted(
        [(p, info) for p, info in threads.items() if info["depth"] == 1],
        key=lambda x: x[1]["order"],
    )
    subs = sorted(
        [(p, info) for p, info in threads.items() if info["depth"] > 1],
        key=lambda x: x[1]["order"],
    )

    print(f"# Run summary")
    print(f"source: {source}")
    print(f"threads: {len(threads)} ({len(top_executions)} top-level execution(s), {len(subs)} subagent)")

    print(f"\n## Top-level execution(s)")
    if not top_executions:
        print("(none)")
    for _, info in top_executions:
        tid = info["thread_id"]
        try:
            msgs = await fetch_messages(tid)
        except Exception as e:
            print(f"- {tid}: ERROR fetching messages: {e}")
            continue
        types = Counter(type(m).__name__ for m in msgs)
        errs = sum(1 for m in msgs if _is_tool_error(m))
        max_tok = sum(
            1
            for m in msgs
            if (getattr(m, "response_metadata", {}) or {}).get("stop_reason") == "max_tokens"
        )
        last_stop = None
        for m in reversed(msgs):
            s = (getattr(m, "response_metadata", {}) or {}).get("stop_reason")
            if s:
                last_stop = s
                break
        print(f"- {tid}")
        if info["description"]:
            print(f"    description: {info['description']}")
        print(f"    messages: {len(msgs)}  types: {dict(types)}")
        print(f"    tool-error messages: {errs}  max_tokens stops: {max_tok}  last stop: {last_stop}")

    print(f"\n## Subagent threads ({len(subs)})")
    if not subs:
        print("(none)")
    for _, info in subs:
        desc = info["description"] or ""
        desc = (desc[:120] + " …") if len(desc) > 120 else desc
        print(f"- {info['thread_id']}   (under: {info['parent']})")
        if desc:
            print(f"    {desc}")

    print(
        "\nNext steps:"
        "\n  messages <log-path> [--thread T] [--errors-only] [--tool NAME] [--range A:B] [--full]"
        "\n  message  <log-path> <idx> [--thread T]"
    )
    return 0


def _default_thread_id(threads: dict[tuple[str, ...], dict[str, Any]]) -> str:
    top_executions = [info for _, info in threads.items() if info["depth"] == 1]
    if len(top_executions) == 0:
        raise RuntimeError("No top-level executions found in events file.")
    if len(top_executions) > 1:
        ids = [t["thread_id"] for t in top_executions]
        raise RuntimeError(
            f"Multiple top-level executions found; pass --thread explicitly. Choices: {ids}"
        )
    return top_executions[0]["thread_id"]


async def cmd_messages(args: argparse.Namespace) -> int:
    threads, _ = await resolve_catalog(args)
    thread_id = args.thread or _default_thread_id(threads)

    msgs = await fetch_messages(thread_id)
    if not msgs:
        print(f"(no messages found for thread_id={thread_id})")
        return 1

    print(f"# Messages for thread: {thread_id}")
    print(f"  total messages: {len(msgs)}")
    if args.range or args.errors_only or args.type or args.tool:
        print(
            f"  filters: range={args.range or '-'}  type={args.type or '-'}  "
            f"tool={args.tool or '-'}  errors_only={args.errors_only}"
        )
    print()

    shown = 0
    for i, m in _select_messages(msgs, args):
        print(_msg_full(i, m) if args.full else _msg_oneline(i, m))
        print()
        shown += 1

    if shown == 0:
        print("(no messages matched filters)")
    else:
        print(f"# shown: {shown}/{len(msgs)}")
    return 0


async def cmd_message(args: argparse.Namespace) -> int:
    threads, _ = await resolve_catalog(args)
    thread_id = args.thread or _default_thread_id(threads)

    msgs = await fetch_messages(thread_id)
    if not msgs:
        print(f"(no messages found for thread_id={thread_id})")
        return 1
    if not (0 <= args.index < len(msgs)):
        print(f"index {args.index} out of range (0..{len(msgs)-1}) for thread {thread_id}")
        return 1

    print(f"# thread: {thread_id}   message {args.index}/{len(msgs)-1}")
    print(_msg_full(args.index, msgs[args.index]))
    return 0


async def cmd_runs(args: argparse.Namespace) -> int:
    runs = await list_recent_runs(args.limit, args.uid)
    print(f"# {len(runs)} most-recent run(s)" + (f" for uid={args.uid}" if args.uid else ""))
    if not runs:
        print("(none — wrong uid, or this DB has no run-index records)")
        return 0
    for run_id, meta in runs:
        start = meta.get("start_time")
        end = meta.get("end_time") or "(unfinished)"
        print(f"- {run_id}   {start} -> {end}")
        tags = meta.get("tags") or {}
        if tags:
            shown = ", ".join(f"{k}={tags[k]}" for k in list(tags)[:5])
            print(f"    tags: {shown}")
    print("\nNext: inspect_run.py summary <run_id>   (or --run-id <run_id>)")
    return 0


# ------------------------- CLI -------------------------

def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by summary/messages/message: a positional log path (which may
    instead be a 32-hex run_id), plus explicit --run-id / --uid."""
    parser.add_argument(
        "log_path",
        nargs="?",
        help=".events.jsonl, .log, autoProve folder, project root, OR a run_id",
    )
    parser.add_argument("--run-id", dest="run_id", help="resolve threads from the run index by run_id")
    parser.add_argument("--uid", help="user namespace for the run index (default: current user)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect messages of an AIAutoProver run.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="List threads and basic stats for a run.")
    _add_source_args(s)

    m = sub.add_parser("messages", help="Print messages from a thread.")
    _add_source_args(m)
    m.add_argument("--thread", help="thread_id (defaults to the thread_id of the run's top-level execution)")
    m.add_argument("--range", help="A:B slice over message indices")
    m.add_argument("--errors-only", action="store_true", help="only tool-error messages")
    m.add_argument("--type", help="filter by class name (e.g. AIMessage, ToolMessage)")
    m.add_argument("--tool", help="only messages invoking/returning from this tool")
    m.add_argument("--full", action="store_true", help="dump full content for each match")

    one = sub.add_parser("message", help="Print one message in full.")
    _add_source_args(one)
    one.add_argument("index", type=int)
    one.add_argument("--thread")

    r = sub.add_parser("runs", help="List recent runs from the run index (by run_id).")
    r.add_argument("--limit", type=int, default=20, help="how many runs to list (default 20)")
    r.add_argument("--uid", help="user namespace for the run index (default: current user)")

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "summary":
        return asyncio.run(cmd_summary(args))
    if args.cmd == "messages":
        return asyncio.run(cmd_messages(args))
    if args.cmd == "message":
        return asyncio.run(cmd_message(args))
    if args.cmd == "runs":
        return asyncio.run(cmd_runs(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())

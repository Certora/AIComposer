"""CLI tool for reviewing knowledge base articles."""
from typing import Any
import argparse
import pathlib
import sys

composer_dir = str(pathlib.Path(__file__).parent.parent.absolute())

if composer_dir not in sys.path:
    sys.path.append(composer_dir)

from composer.workflow.services import get_indexed_store
from composer.kb.knowledge_base import DefaultEmbedder, ReviewStatus, get_review_status

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

KB_NS = ("cvl", "agent", "knowledge")

_STATUS_STYLES: dict[ReviewStatus, str] = {
    ReviewStatus.APPROVED: "green",
    ReviewStatus.PENDING_REVIEW: "yellow",
    ReviewStatus.REJECTED: "red",
}


def get_store():
    return get_indexed_store(DefaultEmbedder())


def fetch_all_articles(store, status_filter: ReviewStatus | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Fetch all articles, optionally filtered by review status. Returns (key, value) pairs."""
    articles: list[tuple[str, dict[str, Any]]] = []
    offset = 0
    while True:
        batch = store.search(KB_NS, limit=100, offset=offset)
        if not batch:
            break
        for item in batch:
            item_status = get_review_status(item.value)
            if status_filter is None or item_status == status_filter:
                articles.append((item.key, item.value))
        offset += 100
    return articles


def cmd_list(args):
    store = get_store()
    status_filter: ReviewStatus | None = ReviewStatus(args.status) if args.status else None
    articles = fetch_all_articles(store, status_filter)

    if not articles:
        console.print(f"No articles found{f' with status {status_filter}' if status_filter else ''}.")
        return

    table = Table(title=f"Knowledge Base Articles{f' ({status_filter})' if status_filter else ''}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold", max_width=50)
    table.add_column("Status", width=16)
    table.add_column("Source", width=8)
    table.add_column("Symptom", max_width=60)

    for i, (key, value) in enumerate(articles, 1):
        status = get_review_status(value)
        source = value.get("source", "unknown")
        style = _STATUS_STYLES.get(status, "")
        table.add_row(
            str(i),
            value.get("title", key),
            f"[{style}]{status}[/{style}]",
            source,
            (value.get("symptom", "")[:57] + "...") if len(value.get("symptom", "")) > 60 else value.get("symptom", "")
        )

    console.print(table)
    console.print(f"\nTotal: {len(articles)} articles")


def render_article(value: dict[str, Any]):
    """Render an article in a Rich Panel."""
    status = get_review_status(value)
    source = value.get("source", "unknown")
    title = value.get("title", "Untitled")
    symptom = value.get("symptom", "")
    body = value.get("body", "")

    md = f"**Symptom:** {symptom}\n\n---\n\n{body}"
    style = _STATUS_STYLES.get(status, "")

    console.print(Panel(
        Markdown(md),
        title=f"[bold]{title}[/bold]",
        subtitle=f"[{style}]{status}[/{style}] | source: {source}",
        border_style=style,
    ))


def cmd_review(_args):
    store = get_store()

    while True:
        articles = fetch_all_articles(store, ReviewStatus.PENDING_REVIEW)
        if not articles:
            console.print("[green]No articles pending review.[/green]")
            return

        table = Table(title="Articles Pending Review")
        table.add_column("#", style="dim", width=4)
        table.add_column("Title", style="bold", max_width=50)
        table.add_column("Source", width=8)
        table.add_column("Symptom", max_width=60)

        for i, (key, value) in enumerate(articles, 1):
            source = value.get("source", "unknown")
            table.add_row(
                str(i),
                value.get("title", key),
                source,
                (value.get("symptom", "")[:57] + "...") if len(value.get("symptom", "")) > 60 else value.get("symptom", "")
            )

        console.print(table)
        console.print(f"\n{len(articles)} article(s) pending review.")
        choice = console.input("\nEnter article number to review (or [bold]q[/bold] to quit): ").strip()

        if choice.lower() == "q":
            return

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(articles):
                console.print("[red]Invalid number.[/red]")
                continue
        except ValueError:
            console.print("[red]Invalid input.[/red]")
            continue

        key, value = articles[idx]
        console.print()
        render_article(value)

        action = console.input("\n[bold][a]pprove[/bold] / [bold][r]eject[/bold] / [bold][b]ack[/bold]: ").strip().lower()

        if action in ("a", "approve"):
            value["review_status"] = ReviewStatus.APPROVED
            store.put(KB_NS, key, value, index=["symptom"])
            console.print(f"[green]Approved:[/green] {value.get('title', key)}\n")
        elif action in ("r", "reject"):
            value["review_status"] = ReviewStatus.REJECTED
            store.put(KB_NS, key, value, index=["symptom"])
            console.print(f"[red]Rejected:[/red] {value.get('title', key)}\n")
        else:
            console.print("Skipped.\n")


def cmd_approve(args):
    store = get_store()
    item = store.get(KB_NS, args.title)
    if item is None:
        console.print(f"[red]No article found with title '{args.title}'[/red]")
        sys.exit(1)
    item.value["review_status"] = ReviewStatus.APPROVED
    store.put(KB_NS, args.title, item.value, index=["symptom"])
    console.print(f"[green]Approved:[/green] {args.title}")


def cmd_reject(args):
    store = get_store()
    item = store.get(KB_NS, args.title)
    if item is None:
        console.print(f"[red]No article found with title '{args.title}'[/red]")
        sys.exit(1)
    item.value["review_status"] = ReviewStatus.REJECTED
    store.put(KB_NS, args.title, item.value, index=["symptom"])
    console.print(f"[red]Rejected:[/red] {args.title}")


def main():
    parser = argparse.ArgumentParser(description="Review knowledge base articles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List articles")
    list_parser.add_argument("--status", choices=[s.value for s in ReviewStatus], default=None,
                             help="Filter by review status")
    list_parser.set_defaults(func=cmd_list)

    # review
    review_parser = subparsers.add_parser("review", help="Interactively review pending articles")
    review_parser.set_defaults(func=cmd_review)

    # approve
    approve_parser = subparsers.add_parser("approve", help="Approve an article by title")
    approve_parser.add_argument("--title", required=True, help="Article title")
    approve_parser.set_defaults(func=cmd_approve)

    # reject
    reject_parser = subparsers.add_parser("reject", help="Reject an article by title")
    reject_parser.add_argument("--title", required=True, help="Article title")
    reject_parser.set_defaults(func=cmd_reject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

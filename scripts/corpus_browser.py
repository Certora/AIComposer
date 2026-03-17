"""
Corpus Browser — TUI for navigating security report corpus results.

Reads ProcessedReport JSON files from the pipeline output directory
and presents them in a navigable tree with syntax-highlighted CVL code.

Usage:
    python -m scripts.corpus_browser [output_dir]
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Tree, Static

from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text

from composer.corpus.models import (
    CorpusEntry, ProcessedReport, UnmatchedRule,
)


# ---------------------------------------------------------------------------
# Tree node data types
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    report: ProcessedReport


@dataclass
class GroupData:
    group_title: str
    group_description: str
    assumptions: str | None
    entries: list[CorpusEntry]


@dataclass
class EntryData:
    entry: CorpusEntry


@dataclass
class UnmatchedGroupData:
    unmatched: list[UnmatchedRule]


@dataclass
class UnmatchedData:
    rule: UnmatchedRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Order matters: "partially verified" must match "partially" before "verified".
_STATUS_MAP: list[tuple[str, str, str]] = [
    ("violated", "\u2717", "red"),
    ("timeout", "\u23f1", "yellow"),
    ("partially", "\u25d0", "yellow"),
    ("verified", "\u2713", "green"),
]


def _status_icon_style(status: str) -> tuple[str, str]:
    s = status.lower()
    for substr, icon, style in _STATUS_MAP:
        if substr in s:
            return icon, style
    return "\u25cb", "dim"


def load_corpus(output_dir: Path) -> list[ProcessedReport]:
    """Load all ProcessedReport JSONs from the output directory."""
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        paths = [output_dir / e["output_file"] for e in manifest]
    else:
        paths = sorted(
            p for p in output_dir.glob("*.json")
            if p.name != "manifest.json"
        )

    reports: list[ProcessedReport] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            reports.append(ProcessedReport.model_validate_json(p.read_text()))
        except Exception as exc:
            print(f"Warning: failed to load {p.name}: {exc}", file=sys.stderr)
    return reports


# ---------------------------------------------------------------------------
# Detail renderers
# ---------------------------------------------------------------------------

def _render_overview(output_dir: Path, reports: list[ProcessedReport]) -> Group:
    t = Text()
    t.append("Corpus Overview\n\n", style="bold underline")
    t.append("Source: ", style="bold")
    t.append(f"{output_dir}\n")
    t.append("Reports: ", style="bold")
    t.append(f"{len(reports)}\n\n")

    total_entries = sum(len(r.entries) for r in reports)
    total_unmatched = sum(len(r.unmatched) for r in reports)
    skipped = sum(1 for r in reports if r.skipped_reason)
    t.append("Rules analyzed: ", style="bold")
    t.append(f"{total_entries}\n")
    t.append("Unmatched: ", style="bold")
    t.append(f"{total_unmatched}\n")
    t.append("Skipped reports: ", style="bold")
    t.append(f"{skipped}\n\n")

    t.append("Navigate the tree to inspect reports, properties, and rules.", style="dim")
    return Group(t)


def _render_report(r: ProcessedReport) -> Group:
    parts: list[Text] = []

    header = Text()
    header.append(f"{r.metadata.protocol_name}\n", style="bold underline")
    header.append(f"\n{r.metadata.protocol_description}\n")
    parts.append(header)

    meta = Text()
    meta.append("\nReport Type: ", style="bold")
    meta.append(f"{r.metadata.report_type}\n")
    meta.append("Source PDF: ", style="bold")
    meta.append(f"{r.source_pdf}\n")
    if r.metadata.repo:
        meta.append("Repository: ", style="bold")
        meta.append(f"{r.metadata.repo.url}\n")
        if r.metadata.repo.commit:
            meta.append("Commit: ", style="bold")
            meta.append(f"{r.metadata.repo.commit}\n")
    parts.append(meta)

    if r.skipped_reason:
        skip = Text()
        skip.append("\nSkipped: ", style="bold yellow")
        skip.append(r.skipped_reason, style="yellow")
        parts.append(skip)
    else:
        stats = Text()
        stats.append("\nStatistics\n", style="bold")
        stats.append(f"  Rules analyzed: {len(r.entries)}\n")
        stats.append(f"  Unmatched: {len(r.unmatched)}\n")

        statuses: dict[str, int] = {}
        for e in r.entries:
            statuses[e.status] = statuses.get(e.status, 0) + 1
        if statuses:
            stats.append("\n  By status:\n")
            for status, count in sorted(statuses.items()):
                icon, style = _status_icon_style(status)
                stats.append(f"    {icon} {status}: {count}\n", style=style)
        parts.append(stats)

    return Group(*parts)


def _render_group(g: GroupData) -> Group:
    parts: list[Text] = []

    header = Text()
    header.append(f"{g.group_title}\n", style="bold underline")
    header.append(f"\n{g.group_description}\n")
    parts.append(header)

    if g.assumptions:
        assumptions = Text()
        assumptions.append("\nAssumptions\n", style="bold")
        assumptions.append(g.assumptions)
        parts.append(assumptions)

    rules = Text()
    rules.append(f"\nRules ({len(g.entries)})\n", style="bold")
    for entry in g.entries:
        icon, style = _status_icon_style(entry.status)
        rules.append(f"\n  {icon} ", style=style)
        rules.append(entry.rule_name, style="bold")
        rules.append(f"  {entry.status}", style=f"dim {style}")
        if entry.rule_description:
            rules.append(f"\n    {entry.rule_description}", style="dim")
    parts.append(rules)

    return Group(*parts)


def _render_entry(e: CorpusEntry) -> Group:
    parts: list[object] = []

    header = Text()
    header.append(f"{e.rule_name}\n", style="bold underline")
    icon, style = _status_icon_style(e.status)
    header.append(f"\n{icon} Status: ", style="bold")
    header.append(f"{e.status}\n", style=style)
    header.append("Spec file: ", style="bold")
    header.append(f"{e.spec_file}\n")
    header.append("Property: ", style="bold")
    header.append(f"{e.property_title}\n")
    if e.rule_description:
        header.append(f"\n{e.rule_description}\n")
    parts.append(header)

    if e.cvl_code:
        parts.append(Text("\nCVL Code\n", style="bold"))
        parts.append(Syntax(
            e.cvl_code.strip(), "solidity",
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        ))

    if e.mechanism:
        mech = Text()
        mech.append("\nMechanism\n", style="bold")
        mech.append(e.mechanism)
        parts.append(mech)

    if e.implementation_notes:
        notes = Text()
        notes.append("\nImplementation Notes\n", style="bold")
        notes.append(e.implementation_notes)
        parts.append(notes)

    if e.assumptions:
        assumptions = Text()
        assumptions.append("\nAssumptions\n", style="bold")
        assumptions.append(e.assumptions)
        parts.append(assumptions)

    return Group(*parts)


def _render_unmatched_group(unmatched: list[UnmatchedRule]) -> Group:
    t = Text()
    t.append(f"Unmatched Rules ({len(unmatched)})\n\n", style="bold underline yellow")
    t.append(
        "These rules were mentioned in the report but could not be "
        "located in the repository.\n\n",
        style="dim",
    )
    for um in unmatched:
        t.append(f"\u2717 {um.rule.name}", style="red bold")
        t.append(f"  {um.rule.status}\n", style="dim")
        t.append(f"  {um.rule.description}\n", style="dim")
        t.append(f"  Reason: {um.reason}\n\n")
    return Group(t)


def _render_unmatched_single(um: UnmatchedRule) -> Group:
    t = Text()
    t.append(f"{um.rule.name}\n", style="bold underline red")
    t.append(f"\nStatus: {um.rule.status}\n")
    t.append(f"Property Group: {um.property_id}\n\n")
    if um.rule.description:
        t.append("Description\n", style="bold")
        t.append(f"{um.rule.description}\n\n")
    t.append("Reason Not Found\n", style="bold")
    t.append(um.reason)
    return Group(t)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class CorpusBrowserApp(App):
    TITLE = "Corpus Browser"

    CSS = """
    #tree-pane {
        width: 1fr;
        min-width: 35;
        max-width: 65;
        border: solid $primary;
    }
    #detail-pane {
        width: 3fr;
        border: solid $primary;
        padding: 1 2;
    }
    #detail-content {
        width: 1fr;
    }
    #stats-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("e", "expand_all", "Expand All", show=True),
        Binding("c", "collapse_all", "Collapse", show=True),
    ]

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self._output_dir = output_dir
        self._reports: list[ProcessedReport] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="tree-pane"):
                yield Tree("Corpus", id="corpus-tree")
            with VerticalScroll(id="detail-pane"):
                yield Static("", id="detail-content")
        yield Static("", id="stats-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_and_build()

    def _load_and_build(self) -> None:
        self._reports = load_corpus(self._output_dir)
        self._build_tree()
        self._update_stats()
        self.query_one("#detail-content", Static).update(
            _render_overview(self._output_dir, self._reports)
        )

    # -- Stats --

    def _update_stats(self) -> None:
        total = len(self._reports)
        entries = sum(len(r.entries) for r in self._reports)
        unmatched = sum(len(r.unmatched) for r in self._reports)
        skipped = sum(1 for r in self._reports if r.skipped_reason)

        self.query_one("#stats-bar", Static).update(
            f" {total} reports \u2502 {total - skipped} active \u2502 "
            f"{entries} rules \u2502 {unmatched} unmatched"
        )

    # -- Tree --

    def _build_tree(self) -> None:
        tree = self.query_one("#corpus-tree", Tree)
        tree.clear()
        tree.root.data = None

        for report in self._reports:
            self._add_report_node(tree, report)

        tree.root.expand()

    def _add_report_node(self, tree: Tree, report: ProcessedReport) -> None:
        if report.skipped_reason:
            label = Text.assemble(
                ("\u2298 ", "dim"),
                (report.metadata.protocol_name, "dim strikethrough"),
            )
            tree.root.add_leaf(label, data=ReportData(report))
            return

        n = len(report.entries)
        um = len(report.unmatched)
        parts = [f"{n} rules"]
        if um:
            parts.append(f"{um} unmatched")
        suffix = f" ({', '.join(parts)})"

        label = Text.assemble(
            ("\u25cf ", "green"),
            report.metadata.protocol_name,
            (suffix, "dim"),
        )
        rnode = tree.root.add(label, data=ReportData(report))

        # Group entries by property_id, preserving insertion order
        groups: dict[str, list[CorpusEntry]] = {}
        for entry in report.entries:
            groups.setdefault(entry.property_id, []).append(entry)

        for gid, entries in groups.items():
            first = entries[0]
            glabel = Text.assemble(
                (f"[{gid}] ", "bold"),
                first.property_title,
                (f" ({len(entries)})", "dim"),
            )
            gnode = rnode.add(
                glabel,
                data=GroupData(
                    first.property_title,
                    first.property_description,
                    first.assumptions,
                    entries,
                ),
            )

            for entry in entries:
                icon, style = _status_icon_style(entry.status)
                elabel = Text.assemble(
                    (f"{icon} ", style),
                    entry.rule_name,
                    (f"  {entry.status}", f"dim {style}"),
                )
                gnode.add_leaf(elabel, data=EntryData(entry))

        if report.unmatched:
            ulabel = Text.assemble(
                ("\u26a0 ", "yellow"),
                ("Unmatched", "yellow"),
                (f" ({len(report.unmatched)})", "dim"),
            )
            unode = rnode.add(ulabel, data=UnmatchedGroupData(report.unmatched))
            for um_rule in report.unmatched:
                umlabel = Text.assemble(
                    ("\u2717 ", "red"),
                    um_rule.rule.name,
                )
                unode.add_leaf(umlabel, data=UnmatchedData(um_rule))

    # -- Detail dispatch --

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        data = event.node.data
        if data is None:
            return

        content = self.query_one("#detail-content", Static)

        match data:
            case ReportData(report):
                content.update(_render_report(report))
            case GroupData() as g:
                content.update(_render_group(g))
            case EntryData(entry):
                content.update(_render_entry(entry))
            case UnmatchedGroupData(unmatched):
                content.update(_render_unmatched_group(unmatched))
            case UnmatchedData(rule):
                content.update(_render_unmatched_single(rule))

    # -- Actions --

    def action_refresh(self) -> None:
        self._load_and_build()
        self.notify("Corpus reloaded")

    def action_expand_all(self) -> None:
        tree = self.query_one("#corpus-tree", Tree)
        for node in tree.root.children:
            node.expand_all()

    def action_collapse_all(self) -> None:
        tree = self.query_one("#corpus-tree", Tree)
        for node in tree.root.children:
            node.collapse_all()

    def action_quit(self) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse corpus ingestion results",
    )
    parser.add_argument(
        "output_dir", type=Path, nargs="?",
        default=Path("./corpus_output"),
        help="Directory containing ProcessedReport JSON files (default: ./corpus_output)",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Error: {args.output_dir} is not a directory")
        sys.exit(1)

    app = CorpusBrowserApp(args.output_dir)
    app.run()


if __name__ == "__main__":
    main()

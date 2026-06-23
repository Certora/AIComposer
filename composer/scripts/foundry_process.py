"""MDX → HTML converter for foundry cheatcode docs.

A Python port of the original Node script (``scratch/cheatcode-rag/process.js``),
which ran the unified / remark / rehype stack. This reproduces the same output
shape that :mod:`composer.scripts.foundry_ragbuild` parses:

* YAML frontmatter (``--- ... ---``) is stripped.
* Headings, fenced code (``<pre><code class="language-X">``), GFM tables
  (``<table><thead>/<tbody>``), lists, links, and inline code render as usual.
* ``:::warning`` / ``:::note`` (and the other vocs admonition kinds) container
  directives become ``<div class="admonition {kind}">`` — the marker
  ``foundry_ragbuild`` keys its Gotchas / code-group handling off of.

The original used ``remark-directive``, which turned *any* ``:::name`` into an
admonition div; markdown-it's container plugin registers one name at a time, so
we enumerate the kinds the book actually uses. Linkify and task lists (also part
of ``remark-gfm``) are intentionally omitted — cheatcode pages use neither (only
explicit ``[text](url)`` links) — which also avoids the ``linkify-it-py`` dep.

Usage mirrors process.js for a single file (``input.mdx output.html``); passing a
directory as ``input`` converts every ``*.mdx`` under it (recursively), mirroring
the directory structure into the output directory.
"""

import argparse
import pathlib
import sys

from markdown_it import MarkdownIt
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.front_matter import front_matter_plugin


# vocs / foundry-book admonition kinds.
_ADMONITIONS = (
    "note", "tip", "info", "warning", "danger", "caution", "important", "details",
)


# Cheatcode pages deliberately kept out of the RAG (the .mdx files that had no
# generated .html sibling in the original prototype): the env-var readers, host
# IO (ffi / fs / write-*), and the non-cheatcode ``overview`` index page. Applied
# in directory mode only — naming a single file explicitly still converts it.
_EXCLUDED_STEMS = frozenset({
    "env-address", "env-bool", "env-bytes", "env-bytes32",
    "env-int", "env-or", "env-string", "env-uint",
    "ffi", "fs", "overview", "write-json", "write-toml",
})


def _admonition_render(kind: str):
    """Container render for one admonition kind: ``<div class="admonition
    {kind}">`` ... ``</div>``. Inner markdown still renders as block content, so
    paragraphs / code inside an admonition are preserved."""
    def render(self, tokens, idx, _options, _env) -> str:
        if tokens[idx].nesting == 1:
            return f'<div class="admonition {kind}">'
        return "</div>\n"
    return render


def make_converter() -> MarkdownIt:
    """Build the markdown-it parser configured to match process.js's output."""
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    md.use(front_matter_plugin)
    for kind in _ADMONITIONS:
        md.use(container_plugin, name=kind, render=_admonition_render(kind))
    return md


def convert_text(md: MarkdownIt, source: str) -> str:
    return md.render(source)


def _convert_file(md: MarkdownIt, src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(convert_text(md, src.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert foundry cheatcode .mdx file(s) to .html."
    )
    parser.add_argument(
        "input", type=pathlib.Path,
        help="An .mdx file, or a directory of .mdx files (searched recursively).",
    )
    parser.add_argument(
        "output", type=pathlib.Path,
        help="Output .html file (single-file input), or output directory "
             "(directory input — structure is mirrored).",
    )
    args = parser.parse_args()

    md = make_converter()

    if args.input.is_dir():
        all_mdx = sorted(args.input.rglob("*.mdx"))
        if not all_mdx:
            print(f"No .mdx files found under {args.input}", file=sys.stderr)
            return 1
        sources = [s for s in all_mdx if s.stem not in _EXCLUDED_STEMS]
        for src in sources:
            dst = args.output / src.relative_to(args.input).with_suffix(".html")
            _convert_file(md, src, dst)
        skipped = len(all_mdx) - len(sources)
        print(
            f"Converted {len(sources)} file(s) into {args.output} "
            f"({skipped} excluded)",
            file=sys.stderr,
        )
        return 0

    if not args.input.is_file():
        print(f"No such file or directory: {args.input}", file=sys.stderr)
        return 1
    _convert_file(md, args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

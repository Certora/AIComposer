"""Jinja environment + markdown rendering for web-frontend HTML.

Two helpers:

- ``render_fragment(name, **ctx)`` — Jinja, autoescape on, rooted at
  ``composer/web/templates/``. Used everywhere wire-bound HTML
  is built.
- ``render_markdown(text)`` — markdown-it-py with raw-HTML disabled,
  CommonMark, ``linkify`` for bare URLs. Used for AI / sub-agent text
  content that arrives as markdown and needs to render as safe HTML
  inside the panel (sub-agents in particular emit prose with code
  blocks, lists, headings — rendering them as plain text loses
  readability, and injecting raw markdown leaves the user staring at
  literal asterisks).
"""

import pathlib
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt


_TEMPLATES = pathlib.Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES),
    autoescape=select_autoescape(default=True, default_for_string=True),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render_fragment(name: str, /, **ctx: Any) -> str:
    """Render the named fragment template with *ctx* and return the HTML.

    *name* is template-relative — e.g. ``"fragments/nested_workflow.j2"``
    resolves to ``composer/web/templates/fragments/nested_workflow.j2``.
    Pass plain strings; autoescape handles HTML special characters."""
    return _env.get_template(name).render(**ctx)


# CommonMark + linkify, raw HTML disabled. The ``html=False`` flag is
# the security knob: any literal ``<script>`` (or any tag) in the
# markdown source gets escaped rather than passed through. Combined
# with linkify for bare URLs and ``breaks`` for soft-newline-as-<br>
# (matches typical chat-style formatting), this is the configuration
# the Python community defaults to for "trusted-but-paranoid" markdown.
_md = MarkdownIt(
    "commonmark",
    {
        "breaks": True,
        "html": False,
        "linkify": True,
    },
).enable("table").enable("strikethrough")


def render_markdown(text: str) -> str:
    """Render *text* as markdown to safe HTML.

    No raw HTML is honoured (``<script>`` and friends get escaped).
    Output is suitable to drop into a Jinja template via ``|safe``."""
    return _md.render(text)

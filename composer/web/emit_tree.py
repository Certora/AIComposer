"""Nesting-aware emit tree for the autoprove web frontend (Phase 1).

The browser dispatcher knows three operations — ``append``, ``replace``,
``inner`` — each carrying a CSS selector and an HTML payload. The
server's job is to produce the right ``(op, sel, html)`` triples in the
right order so an IOHandler-style nested-workflow protocol
(``log_start(path) ... log_end(path)``) renders as a tree of
``<details>`` skeletons in the DOM with appends landing in the correct
nest level.

``EmitTree`` owns the ``path → selector`` mapping for a single top-level
task and the rules for opening, closing, and appending to nested
contexts. It is deliberately a pure synchronous data structure — no
asyncio, no FastAPI, no browser. Phase 2 wires it into
``AutoProveWebHandler``; Phase 3 drives it from the real pipeline.

All HTML structure is rendered via Jinja templates under
``composer/web/templates/fragments/`` (see :mod:`composer.web.render`)
so escaping is consistent and there are no f-string-formatted markup
strings in this module.

Run the bottom-of-file scenarios for a hand-eyeballable smoke test::

    python -m composer.web.emit_tree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Callable

from composer.web.render import render_fragment


Path = tuple[str, ...]


class WireOp(NamedTuple):
    """A single DOM operation in SSE wire format.

    Browser-side dispatch maps each ``op`` to a DOM call::

        append  → target.insertAdjacentHTML("beforeend", html)
        replace → target.outerHTML = html
        inner   → target.innerHTML = html
    """
    op: Literal["append", "replace", "inner"]
    sel: str
    html: str


@dataclass(frozen=True)
class NestedContext:
    """Selectors for an open nested workflow.

    ``body_selector`` is where appends at this path land.
    ``summary_selector`` targets the ``<summary>`` of the same
    ``<details>``; useful when the caller wants to ``replace`` the
    summary text on close (e.g. to stamp "done in 1.4s" onto a
    finished nest).

    ``summary_selector`` is ``None`` for *aliased* paths registered via
    :meth:`EmitTree.register_alias` — those paths emit no structural
    HTML, so there is no ``<summary>`` to update on leave. ``label`` is
    the empty string in that case.
    """
    body_selector: str
    summary_selector: str | None
    label: str


@dataclass
class EmitTree:
    """Per-task selector tree.

    One ``EmitTree`` per top-level task. ``root_selector`` is the task's
    main log container (e.g. ``"#log-abc123"``); empty paths resolve to it.
    Nested workflows register sub-selectors via ``enter(path, label)``;
    subsequent ``append_at(path, ...)`` calls target the registered body.

    ``id_prefix`` namespaces the generated DOM ids so two ``EmitTree``s
    on the same page don't collide (e.g. ``"task-abc123"`` →
    ``#task-abc123-nest-body-0``). Caller picks something unique to the
    task / run context.

    Error policy is strict on purpose: re-entering an already-entered
    path raises, leaving an unentered path raises, appending at an
    unentered non-root path raises. These are protocol-level mistakes;
    surfacing them at the EmitTree boundary keeps the renderer honest.
    """

    root_selector: str
    id_prefix: str
    _entries: dict[Path, NestedContext] = field(default_factory=dict)
    _counter: int = 0

    # ---- read-only lookups --------------------------------------------------

    def selector_for(self, path: Path) -> str:
        """Resolve the selector at *path*. Empty path → root.

        Raises ``KeyError`` if a non-empty path was never ``enter()``-ed."""
        if not path:
            return self.root_selector
        try:
            return self._entries[path].body_selector
        except KeyError:
            raise KeyError(
                f"no nested context at path {path!r}; "
                f"caller must enter() before append_at()"
            ) from None

    def context_for(self, path: Path) -> NestedContext:
        """Return the body/summary selectors registered for *path*.

        Raises ``KeyError`` for unentered or root paths (root has no
        ``<summary>``)."""
        if not path:
            raise KeyError("root path has no NestedContext")
        try:
            return self._entries[path]
        except KeyError:
            raise KeyError(
                f"no nested context at path {path!r}"
            ) from None

    # ---- mutation -----------------------------------------------------------

    def enter(self, path: Path, label: str) -> WireOp:
        """Open a nested workflow at *path*.

        Generates a ``<details>`` skeleton (via
        ``fragments/nested_workflow.j2``) appended under *path*'s parent
        body (or root if path is depth-1), registers the new body and
        summary selectors, and returns the ``append`` op the caller
        forwards to the browser.

        Raises ``ValueError`` for empty paths (root has no parent) or
        when *path* is already registered. Raises ``KeyError`` if the
        parent path isn't registered (so depth-2 paths require the
        depth-1 parent to be entered first)."""
        if not path:
            raise ValueError("cannot enter() at the empty (root) path")
        if path in self._entries:
            raise ValueError(
                f"path {path!r} is already entered; leave() first"
            )

        parent_sel = self.selector_for(path[:-1])
        body_id = f"{self.id_prefix}-nest-body-{self._counter}"
        sum_id = f"{self.id_prefix}-nest-sum-{self._counter}"
        self._counter += 1
        self._entries[path] = NestedContext(
            body_selector=f"#{body_id}",
            summary_selector=f"#{sum_id}",
            label=label,
        )

        html = render_fragment(
            "fragments/nested_workflow.j2",
            summary_id=sum_id,
            body_id=body_id,
            label=label,
        )
        return WireOp("append", parent_sel, html)

    def register_alias(self, path: Path) -> None:
        """Register *path* as a transparent alias for its parent's body
        selector — append/leave at this path emit no structural HTML.

        Used when a path needs to be in the registry so depth-N+1
        children can find their parent at lookup time, but emitting a
        ``<details>`` would be unwanted chrome. The autoprove web
        handler uses this for top-level workflows within a task scope:
        the task panel already serves as the visual container, so
        depth-1 paths shouldn't add a wrapper of their own — but
        depth-2+ children need the depth-1 selector to resolve to the
        task root.

        Raises ``ValueError`` for empty / already-registered paths;
        raises ``KeyError`` if the parent path isn't registered."""
        if not path:
            raise ValueError("cannot register_alias() at the empty (root) path")
        if path in self._entries:
            raise ValueError(
                f"path {path!r} is already registered; leave() first"
            )
        parent_sel = self.selector_for(path[:-1])
        self._entries[path] = NestedContext(
            body_selector=parent_sel,
            summary_selector=None,
            label="",
        )

    def leave(
        self, path: Path, *, status: str | None = None, status_class: str = "done",
    ) -> tuple[WireOp, ...]:
        """Deregister *path*. Returns a (possibly empty) tuple of ops.

        If *status* is provided AND the path was registered via
        :meth:`enter` (not :meth:`register_alias`), also emits a
        ``replace`` against the original ``<summary>`` so the closed
        nest carries final state text. Aliases ignore ``status``
        silently — they have no ``<summary>`` to update — so callers
        can pass ``status="done"`` uniformly across depth levels
        without depth-aware branching.

        Raises ``KeyError`` if *path* was never entered."""
        if path not in self._entries:
            raise KeyError(
                f"leave() called for path {path!r} but it was never entered"
            )
        ctx = self._entries.pop(path)

        if status is None or ctx.summary_selector is None:
            return ()

        sum_id = ctx.summary_selector.lstrip("#")
        html = render_fragment(
            "fragments/summary_status.j2",
            summary_id=sum_id,
            label=ctx.label,
            status=status,
            status_class=status_class,
        )
        return (WireOp("replace", ctx.summary_selector, html),)

    def append_at(self, path: Path, html: str) -> WireOp:
        """Append the already-rendered *html* at *path*'s registered body
        selector (or the root selector when *path* is empty).

        EmitTree does not template *html* — the caller is expected to
        have rendered it via ``render_fragment`` (or equivalent) so that
        autoescaping is applied at the right boundary."""
        return WireOp("append", self.selector_for(path), html)


# ---------------------------------------------------------------------------
# Hand-eyeballable scenarios
# ---------------------------------------------------------------------------

def _log(kind: str, text: str) -> str:
    """Render a demo log entry through the same template Phase 2 will use."""
    return render_fragment("fragments/log_entry.j2", kind=kind, text=text)


def _print_op(op: WireOp) -> None:
    """Pretty-print one triple. HTML is single-lined and length-capped so
    the structure (op + sel) is what stands out, not the markup."""
    flat = " ".join(op.html.split())
    
    print(f"  {op.op:7} {op.sel:30} {flat}")


def _scenario(label: str) -> None:
    print(f"\n── {label} ──")


def _run_scenarios() -> None:
    # 1. Flat — no nesting. Three appends to root.
    _scenario("scenario 1: flat task, no nesting")
    t = EmitTree(root_selector="#log-task1", id_prefix="task1")
    _print_op(t.append_at((), _log("tool", "read_file(Vault.sol)")))
    _print_op(t.append_at((), _log("thinking", "considering invariants")))
    _print_op(t.append_at((), _log("ai", "draft written")))

    # 2. One nested workflow inside a task.
    _scenario("scenario 2: one level of nesting (CVL research sub-agent)")
    t = EmitTree(root_selector="#log-task2", id_prefix="task2")
    _print_op(t.append_at((), _log("ai", "starting analysis")))
    _print_op(t.enter(("research",), "CVL Research"))
    _print_op(t.append_at(("research",), _log("tool", "manual search: invariants")))
    _print_op(t.append_at(("research",), _log("ai", "found relevant section")))
    for op in t.leave(("research",), status="done (1.4s)"):
        _print_op(op)
    _print_op(t.append_at((), _log("ai", "continuing after research")))

    # 3. Two levels deep.
    _scenario("scenario 3: two levels deep (sub-agent spawns sub-sub-agent)")
    t = EmitTree(root_selector="#log-task3", id_prefix="task3")
    _print_op(t.enter(("outer",), "Outer agent"))
    _print_op(t.append_at(("outer",), _log("ai", "outer working")))
    _print_op(t.enter(("outer", "inner"), "Inner agent"))
    _print_op(t.append_at(("outer", "inner"), _log("tool", "deep query")))
    for op in t.leave(("outer", "inner"), status="done"):
        _print_op(op)
    _print_op(t.append_at(("outer",), _log("ai", "outer done with inner")))
    for op in t.leave(("outer",), status="done"):
        _print_op(op)

    # 4. Sibling nests, both open at once.
    _scenario("scenario 4: sibling nests open simultaneously")
    t = EmitTree(root_selector="#log-task4", id_prefix="task4")
    _print_op(t.enter(("a",), "Branch A"))
    _print_op(t.enter(("b",), "Branch B"))
    _print_op(t.append_at(("a",), _log("ai", "a: log line")))
    _print_op(t.append_at(("b",), _log("ai", "b: log line")))
    for op in t.leave(("a",), status="done"):
        _print_op(op)
    for op in t.leave(("b",), status="failed", status_class="error"):
        _print_op(op)

    # 5. Protocol violations — each should raise.
    _scenario("scenario 5: protocol violations (each should raise)")
    t = EmitTree(root_selector="#log-task5", id_prefix="task5")

    def reenter() -> None:
        t.enter(("x",), "first")
        t.enter(("x",), "second")

    cases: list[tuple[str, "Callable"]] = [
        ("re-enter without leave",        reenter),
        ("emit at unentered path",        lambda: t.append_at(("never-entered",), "<div/>")),
        ("leave unentered path",          lambda: t.leave(("nope",))),
        ("enter at empty (root) path",    lambda: t.enter((), "root cannot be a nest")),
        ("enter depth-2 without parent",  lambda: t.enter(("missing-parent", "child"), "orphan")),
    ]
    for label, fn in cases:
        try:
            fn()
            print(f"  FAIL    [{label}] did not raise")
        except (ValueError, KeyError) as exc:
            print(f"  raise   {type(exc).__name__:10} {label}: {exc}")


if __name__ == "__main__":
    _run_scenarios()

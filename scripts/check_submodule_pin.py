#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging"]
# ///

#!/usr/bin/env python3
"""Verify that git submodule pointers match the @<ref> pins in pyproject.toml.

Configure PINNED_SUBMODULES below to declare which submodule paths should
track which dependency entries. Run from the repo root.
"""
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlparse

# (submodule path relative to the repo root, dependency name in pyproject.toml)
PINNED_SUBMODULES = [
    ("graphcore", "graphcore"),
]

PYPROJECT = Path("pyproject.toml")

from packaging.requirements import Requirement

def parse_dep(raw: str) -> tuple[str, str | None]:
    """For a PEP 508 dep `name[extras] @ <url>@<ref>`, return (name, ref).

    Returns (None, None) for non-URL deps; (name, None) for URLs with no @ref.
    """
    req = Requirement(raw)
    if not req.url:
        return req.name, None
    parsed = urlparse(req.url)
    if "@" in parsed.path:
        return req.name, parsed.path.rsplit("@", 1)[1]
    return req.name, None


def submodule_sha(path: str) -> str:
    """Return the SHA the submodule pointer is set to in the *index*.

    Reads from the index rather than HEAD so this works correctly inside
    a pre-commit hook, where a submodule bump may have been staged but
    not yet committed. On a clean CI checkout the index equals HEAD, so
    this is equivalent there.
    """
    out = subprocess.check_output(
        ["git", "ls-files", "--stage", "--", path], text=True
    ).strip()
    if not out:
        sys.exit(f"submodule {path!r} is not tracked")
    # format: "<mode> <sha> <stage>\t<path>"
    mode, sha, _stage, _name = out.split(maxsplit=3)
    if mode != "160000":
        sys.exit(f"{path!r} is not a submodule (mode={mode})")
    return sha

def read_indexed(path: str) -> str:
    """Return the contents of `path` as it exists in the git index."""
    return subprocess.check_output(
        ["git", "show", f":{path}"], text=True
    )

def pinned_sha(dep_name: str) -> str:
    data = tomllib.loads(read_indexed(str(PYPROJECT)))
    for raw in data["project"]["dependencies"]:
        name, ref = parse_dep(raw)
        if name and name.lower() == dep_name.lower():
            if ref is None:
                sys.exit(f"dependency {dep_name!r} has no @<ref>: {raw}")
            return ref
    sys.exit(f"dependency {dep_name!r} not found in pyproject.toml")


def main() -> int:
    failures = []
    for path, dep in PINNED_SUBMODULES:
        sm = submodule_sha(path)
        pin = pinned_sha(dep)
        # Accept abbreviated SHAs in either direction.
        if not (sm.startswith(pin) or pin.startswith(sm)):
            failures.append((path, dep, sm, pin))
    if not failures:
        return 0
    print("Submodule pointer / pyproject pin mismatch:\n", file=sys.stderr)
    for path, dep, sm, pin in failures:
        print(f"  {dep}  (submodule: {path})", file=sys.stderr)
        print(f"    submodule  -> {sm}", file=sys.stderr)
        print(f"    pyproject  -> {pin}\n", file=sys.stderr)
    print("Resolve by running one of:", file=sys.stderr)
    print("  # adopt the pyproject pin (move the submodule):", file=sys.stderr)
    print("  git -C <submodule-path> fetch && git -C <submodule-path> checkout <pin>", file=sys.stderr)
    print("  git add <submodule-path>", file=sys.stderr)
    print("  # adopt the submodule pointer (edit the pin):", file=sys.stderr)
    print("  # change the @<sha> on the dep in pyproject.toml to match the submodule's sha", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
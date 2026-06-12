"""Tests for the harness import-resolution guard (`_unresolved_imports`).

This guards against harnesses that import a sibling by bare filename
(`import "Base.sol";`), which Solidity does not treat as a relative import and
which therefore fails to compile with "File not found".
"""

from composer.spec.source.harness import _unresolved_imports


# Files that "exist" — project-root-relative paths, as the resolver produces.
_EXISTING = {
    "certora/harnesses/BaseHarness.sol",
    "src/contracts/token/MyToken.sol",
}


def _exists(rel: str) -> bool:
    return rel in _EXISTING


def test_bare_sibling_import_is_flagged():
    src = 'pragma solidity ^0.8.0;\nimport "BaseHarness.sol";\ncontract Inst1 is BaseHarness {}'
    bad = _unresolved_imports("certora/harnesses/Inst1.sol", src, _exists)
    assert bad == ["BaseHarness.sol"]


def test_relative_sibling_import_resolves():
    src = 'import "./BaseHarness.sol";\ncontract Inst1 is BaseHarness {}'
    assert _unresolved_imports("certora/harnesses/Inst1.sol", src, _exists) == []


def test_project_relative_target_import_resolves():
    src = 'import "src/contracts/token/MyToken.sol";\ncontract TokenInstance1 is MyToken {}'
    assert _unresolved_imports("certora/harnesses/TokenInstance1.sol", src, _exists) == []


def test_named_and_dotdot_imports_resolve():
    src = (
        'import {MyToken} from "../../src/contracts/token/MyToken.sol";\n'
        'import * as B from "./BaseHarness.sol";\n'
        "contract Inst is BaseHarness {}"
    )
    # ../../ from certora/harnesses/ lands at src/contracts/token/MyToken.sol
    assert _unresolved_imports("certora/harnesses/Inst.sol", src, _exists) == []


def test_multiple_imports_reports_only_unresolved():
    src = (
        'import "./BaseHarness.sol";\n'        # ok (relative sibling)
        'import "MissingThing.sol";\n'         # bad (bare, missing)
        'import "src/contracts/token/MyToken.sol";\n'  # ok (project-relative)
    )
    assert _unresolved_imports("certora/harnesses/Inst.sol", src, _exists) == ["MissingThing.sol"]

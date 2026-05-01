"""Typed validation keys for the codegen completion gate.

Each delivery requires a set of validation stamps to land in
``AIComposerState.validation`` keyed by domain — the prover stamps once
per registered spec when its run verifies, and the requirements judge
stamps once when every non-skipped requirement is satisfied. The
completion check (in ``composer/tools/result.py``) iterates the required
keys and refuses to deliver until every required stamp matches the
current state digest.

Domains and their payloads:

* ``"prover"`` → ``data: str`` — the VFS path of the spec file the prover
  just verified.
* ``"reqs"``   → ``data: None`` — the requirements judge has no per-key
  axis; it stamps once for the whole requirement set.

Construct via the convenience helpers — :func:`prover_validation` and
:data:`REQS_VALIDATION` — rather than calling :class:`Validation`
directly, so the right ``D`` is picked up for each domain.

State storage: the validation map is ``dict[str, str]`` keyed by
``str(Validation)`` (e.g. ``"prover:foo.spec"``, ``"reqs"``). The
structured ``Validation`` type is the API at construction and lookup
time; the canonical-string encoding is what actually lands in state.
We store strings (rather than ``Validation`` instances directly) because
LangGraph's ``InjectedState`` round-trips state through pydantic when
splicing it into tool args, and pydantic serializes a vanilla
``@dataclass`` to ``dict`` — which kills hashability and equality with
the original instance, breaking dict-key lookup. Strings round-trip
cleanly through every serializer in the stack.
"""

from dataclasses import dataclass
from typing import Literal


type ValidationDomain = Literal["prover", "reqs"]


@dataclass(frozen=True)
class Validation[D]:
    """A typed validation key.

    ``domain`` selects the validation kind; ``data`` carries the
    domain-specific payload (per the docstring at the module level).

    The canonical state-dict key is ``str(self)`` — use that on both
    write and lookup against ``state["validation"]``.
    """
    domain: ValidationDomain
    data: D

    def __str__(self) -> str:
        if self.data is None:
            return self.domain
        return f"{self.domain}:{self.data}"


# ---------------------------------------------------------------------------
# Convenience constructors — prefer these to direct ``Validation(...)`` calls.
# ---------------------------------------------------------------------------


def prover_validation(spec_vfs_path: str) -> Validation[str]:
    """Stamp key for a prover run against a single committed spec."""
    return Validation(domain="prover", data=spec_vfs_path)


REQS_VALIDATION: Validation[None] = Validation(domain="reqs", data=None)
"""Singleton stamp key for the requirements-judge gate."""

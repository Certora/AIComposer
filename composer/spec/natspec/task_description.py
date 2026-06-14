import contextlib
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Awaitable, ContextManager, Iterator, Mapping, Self, TypedDict

from composer.spec.gen_types import InjectedTemplate
from composer.spec.natspec.models import (
    InterfaceDeclModel,
    InterfaceResult,
    StubDeclarationModel,
)
from composer.spec.system_model import ExplicitContract, NatspecApplication, SolidityIdentifier
from composer.spec.util import temp_certora_file


# ---------------------------------------------------------------------------
# Per-call prompt param types
#
# Each generator has a fixed shape of per-call params it injects into the
# description's prompt via ``InjectedTemplate.inject(...)``. Workflow-constant
# params (e.g. ``sort``) are pre-bound in the description itself via
# ``TypedTemplate.bind(...).depends(<CallParams>)``.
# ---------------------------------------------------------------------------


class InterfaceGenCallParams(TypedDict):
    summary: NatspecApplication
    target_contracts: list[ExplicitContract]
    existing_contracts: list[ExplicitContract]
    solc_version: str


class StubGenCallParams(TypedDict):
    solidity_identifier: str
    interface_name: str
    interface_path: str
    the_interface: str
    solc_version: str


ExtraInputProducer = Callable[[], Awaitable[dict | str] | dict | str]


async def resolve_extra_input(
    items: list[str | dict | ExtraInputProducer],
) -> list[str | dict]:
    """Resolve an ``extra_input`` list — awaiting any async producers — into
    a flat list of literal messages suitable for ``FlowInput.input``.
    """
    import inspect
    out: list[str | dict] = []
    for item in items:
        if callable(item):
            produced = item()
            if inspect.isawaitable(produced):
                produced = await produced
            out.append(produced)
        else:
            out.append(item)
    return out


@dataclass
class AgentDescription[T, X: Mapping[str, Any]]:
    """Describes an agent call: its output type + a partially-bound prompt.

    ``output_ty`` is the concrete pydantic model the agent produces (drives
    structured output). ``prompt`` is a template bound with workflow-generic
    params (e.g. ``sort``) and still expecting per-call injection of
    shape ``X`` (e.g. ``summary``, ``contract_name``). ``extra_input`` is
    a list of literal items and/or (possibly-async) producers — each producer
    yields a single ``dict | str`` that gets appended to the agent's initial
    ``FlowInput`` messages. Producers let descriptions pull lazy, async state
    (e.g. current stubs) at agent-dispatch time.
    """
    output_ty: type[T]
    prompt: InjectedTemplate[X]
    extra_input: list[str | dict | ExtraInputProducer] = field(default_factory=list)


class ConfigurationBuilder:
    """Fluent, **functional** builder for a Certora conf dict.

    Seed with the user-supplied ``config_init`` (e.g. ``prover_conf`` overrides).
    Each ``with_*`` call returns a NEW builder with the corresponding key
    overwritten — the receiver is never mutated. This makes a single shared
    "base" builder safe to fan out across parallel workers, each of which can
    chain its own task-specific overrides without racing on a shared dict.

    Pipeline-authoritative ``with_*`` calls still always win over seeded values
    (their later application overwrites). ``build_to`` materializes the merged
    conf as a temp file under ``<path>/certora/`` and yields its absolute path.
    """

    def __init__(self, config_init: dict | None = None):
        self.config: dict = dict(config_init or {})

    def _replace(self, **updates: Any) -> Self:
        """Return a new builder of the same type with ``updates`` merged in.

        Shallow-merges into a fresh dict; existing list/dict values aren't
        deep-copied because every ``with_*`` either supplies a fresh list at
        the call site or sets a primitive, so aliasing isn't possible in
        normal use.
        """
        new = type(self).__new__(type(self))
        new.config = {**self.config, **updates}
        return new

    def with_files(self, files: list[str]) -> Self:
        return self._replace(files=list(files))

    def with_verify(self, *, main_contract: SolidityIdentifier, spec_file: str) -> Self:
        return self._replace(verify=f"{main_contract}:certora/{spec_file}")

    def with_solc(self, version: str) -> Self:
        return self._replace(
            solc=version if version.startswith("solc") else f"solc{version}"
        )

    def with_compilation_steps_only(self) -> Self:
        return self._replace(compilation_steps_only=True)

    def with_loop_iter(self, n: int) -> Self:
        return self._replace(loop_iter=str(n))

    def with_optimistic_loop(self) -> Self:
        return self._replace(optimistic_loop=True)

    def with_optimistic_hashing(self) -> Self:
        return self._replace(optimistic_hashing=True)

    def with_solc_via_ir(self) -> Self:
        return self._replace(solc_via_ir=True)

    def with_strict_solc_optimizer(self) -> Self:
        return self._replace(strict_solc_optimizer=True)

    def with_prover_args(self, args: list[str]) -> Self:
        return self._replace(prover_args=list(args))

    def with_rule(self, rule: str) -> Self:
        return self._replace(rule=[rule])

    def build_to(self, path: pathlib.Path) -> ContextManager[pathlib.Path]:
        """Write the merged conf to ``<path>/certora/run_<uniq>.conf``; yield its absolute path; clean up on exit."""
        return self._build_to(path)

    @contextlib.contextmanager
    def _build_to(self, path: pathlib.Path) -> Iterator[pathlib.Path]:
        import json
        with temp_certora_file(
            content=json.dumps(self.config, indent=2),
            root=str(path),
            ext="conf",
            prefix="run",
        ) as basename:
            yield path / "certora" / basename


class Assembler(ABC):
    @abstractmethod
    def project_directory(self) -> AsyncContextManager[pathlib.Path]:
        ...

@dataclass
class MentalModel[A: NatspecApplication, I: InterfaceDeclModel, S: StubDeclarationModel]:
    """Static, setup-time description of a verification task.

    Holds only configuration seeded once at pipeline entry: the application
    subtype, output-type + prompt bindings for each agent (``interface_desc``,
    ``stub_desc``), the source tree, and user ``prover_conf`` overrides.
    Generated artifacts (interfaces, stubs) are NOT stored here — each
    ``assembler_for_*`` method takes the accumulated results so far and
    returns a fresh ``Assembler`` seeded with them.
    """
    model_ty: type[A]
    interface_desc: AgentDescription[InterfaceResult[I], InterfaceGenCallParams]
    stub_desc: AgentDescription[S, StubGenCallParams]
    source_root: pathlib.Path | None = None
    config_init: dict | None = None

    @property
    def from_existing(self) -> bool:
        return self.source_root is not None

    def config_builder(self) -> ConfigurationBuilder:
        """Fresh ``ConfigurationBuilder`` seeded with the user's ``prover_conf``."""
        return ConfigurationBuilder(self.config_init)

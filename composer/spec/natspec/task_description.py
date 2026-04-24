import contextlib
import pathlib
import shutil
import tempfile
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, AsyncIterator, Awaitable, ContextManager, Iterator, Mapping, Self, TypedDict

from composer.spec.gen_types import InjectedTemplate
from composer.spec.natspec.models import (
    InterfaceDeclModel,
    InterfaceResult,
    StubDeclarationModel,
)
from composer.spec.system_model import ExplicitContract, NatspecApplication
from composer.spec.util import temp_certora_file


# ---------------------------------------------------------------------------
# Per-call prompt param types
#
# Each generator has a fixed shape of per-call params it injects into the
# description's prompt via ``InjectedTemplate.inject(...)``. Workflow-constant
# params (e.g. ``has_source``) are pre-bound in the description itself via
# ``TypedTemplate.bind(...).depends(<CallParams>)``.
# ---------------------------------------------------------------------------


class InterfaceGenCallParams(TypedDict):
    summary: NatspecApplication
    target_contracts: list[ExplicitContract]
    existing_contracts: list[ExplicitContract]
    solc_version: str


class StubGenCallParams(TypedDict):
    contract_name: str
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
    params (e.g. ``has_source``) and still expecting per-call injection of
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
    """Fluent builder for a Certora conf dict.

    Seed with the user-supplied ``config_init`` (e.g. ``prover_conf`` overrides).
    Each ``with_*`` call overwrites the corresponding key, so pipeline-authoritative
    writes always win over seeded values. ``build_to`` materializes the merged conf
    as a temp file under ``<path>/certora/`` and yields its absolute path.
    """

    def __init__(self, config_init: dict | None = None):
        self.config: dict = dict(config_init or {})

    def with_files(self, files: list[str]) -> Self:
        self.config["files"] = list(files)
        return self

    def with_verify(self, *, main_contract: str, spec_file: str) -> Self:
        self.config["verify"] = f"{main_contract}:certora/{spec_file}"
        return self

    def with_solc(self, version: str) -> Self:
        self.config["solc"] = version if version.startswith("solc") else f"solc{version}"
        return self

    def with_compilation_steps_only(self) -> Self:
        self.config["compilation_steps_only"] = True
        return self

    def with_loop_iter(self, n: int) -> Self:
        self.config["loop_iter"] = str(n)
        return self

    def with_optimistic_loop(self) -> Self:
        self.config["optimistic_loop"] = True
        return self

    def with_optimistic_hashing(self) -> Self:
        self.config["optimistic_hashing"] = True
        return self

    def with_solc_via_ir(self) -> Self:
        self.config["solc_via_ir"] = True
        return self

    def with_strict_solc_optimizer(self) -> Self:
        self.config["strict_solc_optimizer"] = True
        return self

    def with_prover_args(self, args: list[str]) -> Self:
        self.config["prover_args"] = list(args)
        return self

    def with_rule(self, rule: str) -> Self:
        self.config["rule"] = [rule]
        return self

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


@contextlib.asynccontextmanager
async def _project_directory(
    source_root: pathlib.Path | None,
    populate: Callable[[pathlib.Path], None],
) -> AsyncIterator[pathlib.Path]:
    """Create a tmpdir, mirror ``source_root`` into it if set, run ``populate``, yield the tmpdir."""
    with tempfile.TemporaryDirectory() as td:
        tmpdir = pathlib.Path(td)
        if source_root is not None:
            for entry in source_root.iterdir():
                target = tmpdir / entry.name
                if entry.is_dir():
                    await asyncio.to_thread(shutil.copytree, entry, target)
                else:
                    shutil.copy(entry, target)
        populate(tmpdir)
        yield tmpdir


class Assembler(ABC):
    @abstractmethod
    def project_directory(self) -> AsyncContextManager[pathlib.Path]:
        ...


@dataclass
class InterfaceGenAssembler(Assembler):
    """Lays out interfaces for solc compilation. No stubs.

    Used by the interface-gen agent's ``validate_interface``.
    """
    interface: InterfaceResult
    source_root: pathlib.Path | None = None

    def project_directory(self) -> AsyncContextManager[pathlib.Path]:
        def populate(tmpdir: pathlib.Path) -> None:
            self.interface.dump_to_path(tmpdir)
        return _project_directory(self.source_root, populate)


def _write_stub(tmpdir: pathlib.Path, stub: StubDeclarationModel) -> None:
    """Write a stub to its agent-chosen ``stub.path`` under ``tmpdir``."""
    target = tmpdir / stub.path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(stub.content)


@dataclass
class StubGenAssembler(Assembler):
    """Lays out interfaces + a single candidate stub.

    Used by the stub-gen agent's ``validate_stub``. The stub is placed at
    its agent-chosen ``stub.path`` (project-relative).
    """
    interface: InterfaceResult
    stub: StubDeclarationModel
    source_root: pathlib.Path | None = None

    def project_directory(self) -> AsyncContextManager[pathlib.Path]:
        def populate(tmpdir: pathlib.Path) -> None:
            self.interface.dump_to_path(tmpdir)
            _write_stub(tmpdir, self.stub)
        return _project_directory(self.source_root, populate)


@dataclass
class SpecCheckAssembler(Assembler):
    """Lays out interfaces + all current stubs. The spec is layered in by the
    caller via ``temp_certora_file`` after ``project_directory()`` yields.

    Used by ``typecheck_spec`` (merge typecheck + advisory typecheck). Each
    stub is written to its own agent-chosen ``stub.path``.
    """
    interface: InterfaceResult
    stubs: dict[str, StubDeclarationModel] = field(default_factory=dict)
    source_root: pathlib.Path | None = None

    def project_directory(self) -> AsyncContextManager[pathlib.Path]:
        def populate(tmpdir: pathlib.Path) -> None:
            self.interface.dump_to_path(tmpdir)
            for stub in self.stubs.values():
                _write_stub(tmpdir, stub)
        return _project_directory(self.source_root, populate)


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

    def assembler_for_interface_gen(self, interface: InterfaceResult[I]) -> Assembler:
        """Assembler for validating a candidate interface (no stubs)."""
        return InterfaceGenAssembler(interface=interface, source_root=self.source_root)

    def assembler_for_stub_gen(
        self, interface: InterfaceResult[I], stub: S
    ) -> Assembler:
        """Assembler for validating a candidate stub against the interfaces."""
        return StubGenAssembler(interface=interface, stub=stub, source_root=self.source_root)

    def assembler_for_spec_check(
        self, interface: InterfaceResult[I], stubs: dict[str, S]
    ) -> Assembler:
        """Assembler for spec typecheck (merge + advisory) — interfaces + all current stubs."""
        return SpecCheckAssembler(interface=interface, stubs={ **stubs }, source_root=self.source_root)

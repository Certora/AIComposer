"""
Semantic registry agent for shared stub field management.

Serializes stub edits via an asyncio.Lock. Each field request spawns a fresh
registry agent that receives accumulated field metadata, the current stub,
and the interface, then decides whether to reuse an existing field or add a
new one. When adding a new field, the agent produces the updated stub source
which is validated against the Solidity compiler before acceptance.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, NotRequired, override, Iterable
import pathlib

_log = logging.getLogger(__name__)

from pydantic import BaseModel, Field as PydanticField, ValidationError

from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from langgraph.types import Command

from graphcore.graph import FlowInput, tool_state_update
from graphcore.tools.schemas import WithAsyncImplementation, WithInjectedId
from graphcore.tools.vfs import Materializer

from composer.spec.context import PlainBuilder, CVLOnlyBuilder
from composer.spec.system_model import SolidityIdentifier
from composer.spec.graph_builder import run_to_completion
from composer.spec.natspec.pipeline_events import StubUpdate
from composer.spec.natspec.models import (
    InterfaceResult,
    StubDeclarationModel,
)
from composer.spec.util import uniq_thread_id
from composer.ui.tool_display import tool_display
from composer.spec.natspec.task_description import Assembler


# ---------------------------------------------------------------------------
# Field metadata schema
# ---------------------------------------------------------------------------

class FieldSpec(BaseModel):
    name: str = PydanticField(description="The Solidity field name")
    type: str = PydanticField(description="The Solidity type (e.g., 'mapping(address => uint256)')")
    description: str = PydanticField(description="What this field tracks")


class FieldMetadata(BaseModel):
    stub_fields: dict[SolidityIdentifier, list[FieldSpec]] = PydanticField(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry agent result
# ---------------------------------------------------------------------------

class RegistryResult(BaseModel):
    """Decision on a stub field request."""
    field_name: str = PydanticField(
        description="The name of the field to use (existing or newly created)"
    )
    is_new: bool = PydanticField(
        description="Whether this is a newly added field"
    )
    field_type: str = PydanticField(
        default="",
        description="The Solidity type for the new field (e.g., 'mapping(address => uint256)'). "
        "Required when is_new is true.",
    )
    rejected: bool = PydanticField(
        default=False,
        description="Set to true if the field request was rejected as 'unsuitable'."
    )
    description: str = PydanticField(
        default="",
        description="A short description of what this field tracks OR why the request was rejected. "
        "Required when is_new is true or when rejected is true.",
    )
    updated_stub: str = PydanticField(
        default="",
        description="The complete updated stub source code with the new field declaration added. "
        "This must be the FULL source, not a diff. Required when is_new is true.",
    )


# ---------------------------------------------------------------------------
# Stub compilation check
# ---------------------------------------------------------------------------

async def _compile_stub(
    stub: str,
    assembler: Assembler,
    solc_version: str,
    stub_path: str,
) -> str | None:
    """Compile the stub against the interfaces with solc.

    The stub is written at its real ``stub_path`` (project-relative) inside
    the tmpdir so relative ``import`` statements in the stub resolve the
    same way they will in the real project tree. Returns ``None`` on
    success, an error string on failure.
    """
    solc_name = f"solc{solc_version}"
    async with assembler.project_directory() as tmpdir:
        stub_abs = tmpdir / stub_path
        stub_abs.parent.mkdir(parents=True, exist_ok=True)
        stub_abs.write_text(stub)
        try:
            proc = await asyncio.create_subprocess_exec(
                solc_name, stub_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
            )

            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return f"Solidity compiler {solc_name} not found"
        if proc.returncode != 0:
            return f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
        return None


# ---------------------------------------------------------------------------
# Registry agent
# ---------------------------------------------------------------------------

async def run_registry_agent(
    request: str,
    stub_content: str,
    stub_path: str,
    field_metadata: list[FieldSpec],
    interface: str,
    solc_version: str,
    builder: PlainBuilder,
    assembler: Assembler,
    *,
    within_tool: str,
    recursion_limit: int,
) -> RegistryResult:
    """Spawn a fresh registry agent to handle a single field request.

    The agent decides whether to reuse an existing field or add a new one.
    When adding, the result validator compiles the updated stub with solc,
    rejecting malformed output so the agent can retry.
    """

    class ST(MessagesState):
        result: NotRequired[RegistryResult]

    async def validate_result(res: RegistryResult) -> str | None:
        if res.is_new:
            if not res.field_type:
                return "When proposing a new field, you must provide field_type (the Solidity type)."
            if not res.description:
                return "When proposing a new field, you must provide field_description."
            if not res.updated_stub:
                return "When proposing a new field, you must provide updated_stub (the complete source code)."
            compile_err = await _compile_stub(res.updated_stub, assembler, solc_version, stub_path)
            if compile_err is not None:
                return (
                    f"The updated stub does not compile. Fix the issue and try again.\n"
                    f"{compile_err}"
                )
        return None
    
    class RegistryResultTool(WithAsyncImplementation[Command | str], WithInjectedId, RegistryResult):
        """
        Call this tool with the result of your work
        """

        async def run(self) -> Command | str:
            if (err_msg := await validate_result(self)) is not None:
                return err_msg
            r = self.model_dump()
            del r["tool_call_id"]
            reg = RegistryResult.model_construct(**r)
            return tool_state_update(self.tool_call_id, "Accepted", result=reg)

    workflow = (
        builder
        .with_default_summarizer()
        .with_state(ST)
        .with_output_key("result")
        .with_input(FlowInput)
        .with_tools([RegistryResultTool.as_tool("result")])
        .with_sys_prompt(
            "You are a Solidity stub field manager. You decide whether a requested "
            "storage variable already exists in the stub (semantically equivalent) "
            "or whether a new field needs to be added. When adding a new field, you "
            "produce the complete updated stub source code."
        ).with_initial_prompt_template(
            "registry_prompt.j2",
        )
    ).compile_async()

    input_parts: list[str | dict] = [
        "The field request is:",
        request,
        "The current stub source code is:",
        stub_content,
        "The interface for this stub is",
        interface
    ]

    if field_metadata:
        field_lines = "\n".join(
            f"  - `{f.type} {f.name}`: {f.description}"
            for f in field_metadata
        )
        input_parts.extend([
            "The currently registered fields are:",
            field_lines,
        ])
    else:
        input_parts.append("No fields have been registered yet.")

    res = await run_to_completion(
        workflow,
        FlowInput(input=input_parts),
        thread_id=uniq_thread_id("stub-registrar"),
        recursion_limit=recursion_limit,
        description="Stub update",
        within_tool=within_tool,
    )
    assert "result" in res
    return res["result"]


# ---------------------------------------------------------------------------
# StubRegistry — serializes stub edits
# ---------------------------------------------------------------------------

STUB_STATE_KEY = "stub_state"

@dataclass
class _StubMemoryState:
    _path: str
    _content: str
    _interface: str
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class _StubDurableState(BaseModel):
    fields: list[FieldSpec]
    content: str

def stub_state_namespace(ns: tuple[str, ...]) -> tuple[str, ...]:
    """The store namespace the StubRegistry persists its per-identifier
    ``_StubDurableState`` records under. Exposed so the cache explorer reads
    the same layout the registry writes, rather than re-deriving it."""
    return ns + (STUB_STATE_KEY,)

async def _state_write(store: BaseStore, ns: tuple[str, ...], id: SolidityIdentifier, state: _StubDurableState):
    await store.aput(
        ns, id, state.model_dump()
    )

async def _state_read(store: BaseStore, ns: tuple[str, ...], id: SolidityIdentifier) -> _StubDurableState | None:
    res = await store.aget(ns, id)
    if res is None:
        return None
    try:
        return _StubDurableState.model_validate(res.value)
    except ValidationError:
        return None

@dataclass
class StubRegistry:
    """Manages the shared stub and its field registry.

    All field mutations are serialized via an asyncio.Lock. Reads are lock-free.
    Field metadata is stored in BaseStore alongside the stub content.
    """
    _store: BaseStore
    _builder: PlainBuilder | CVLOnlyBuilder
    _solc_version: str
    _assembler: Assembler
    _mirror_by_path: dict[str, str]
    _state_by_id: dict[SolidityIdentifier, _StubMemoryState]
    _recursion_limit: int
    _namespace: tuple[str, ...] = ()

    @staticmethod
    async def acreate(
        store: BaseStore,
        namespace: tuple[str, ...],
        builder: PlainBuilder | CVLOnlyBuilder,
        interface: InterfaceResult,
        interface_only_mat: Assembler,
        initial_stubs: dict[SolidityIdentifier, StubDeclarationModel],
        solc_version: str,
        *,
        recursion_limit: int,
    ) -> "StubRegistry":
        ns = stub_state_namespace(namespace)

        """Create or resume a StubRegistry.

        If the store already contains stub content and field metadata for this
        namespace, they are preserved (resume after crash/restart). Otherwise
        the store is initialized with the provided ``initial_stubs`` — each
        serialized as ``{"path", "content", "solidity_identifier"}``. Paths
        are fixed at initialization; only content changes during field updates.
        """
        async def _init_stub(id: SolidityIdentifier, init_stub: StubDeclarationModel) -> tuple[SolidityIdentifier, _StubMemoryState]:
            curr = await _state_read(store, ns, id)
            intf_source = interface.name_to_interface[id].content
            if curr is None:
                await _state_write(store, ns, id, _StubDurableState(
                    content=init_stub.content, fields=[]
                ))
                return (id, _StubMemoryState(_path=init_stub.path, _content=init_stub.content, _interface=intf_source))
            else:
                return (id, _StubMemoryState(
                    _path=init_stub.path,
                    _content=curr.content, _interface=intf_source
                ))
            
        state = {
            k: v for (k,v) in (await asyncio.gather(
                *(_init_stub(id, init_stub) for (id, init_stub) in initial_stubs.items())
            ))
        }

        by_path = {
            s._path: s._content for s in state.values()
        }

        return StubRegistry(
            _store=store,
            _builder=builder,
            _solc_version=solc_version,
            _assembler=interface_only_mat,
            _state_by_id=state,
            _mirror_by_path=by_path,
            _recursion_limit=recursion_limit,
            _namespace=ns
        )
    
    # FS backend stuff

    def get(self, path: str) -> str | None:
        return self._mirror_by_path.get(path, None)
    
    def list(self) -> Iterable[str]:
        return self._mirror_by_path.keys()
    
    async def dump_to(
        self,
        target: pathlib.Path,
        include_path: Callable[[str], bool] | None = None,
    ) -> None:
        for (k, v) in self._mirror_by_path.items():
            if include_path is not None and not include_path(k):
                continue
            full_path = target / k
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(v)

    def read_stub(self, nm: SolidityIdentifier) -> str:
        """Read current stub content (no lock needed)."""
        return self._state_by_id[nm]._content

    async def _read_metadata(
        self, id: SolidityIdentifier
    ) -> _StubDurableState:
        to_ret = await _state_read(
            self._store, self._namespace, id
        )
        assert to_ret is not None
        return to_ret

    async def _write_metadata(
        self, id: SolidityIdentifier, st: _StubDurableState
    ):
        await _state_write(
            self._store, self._namespace, id, st
        )

    async def request_field(
        self, nm: SolidityIdentifier, purpose: str, *, within_tool: str ,
    ) -> str:
        """Request a stub field for a given purpose. Serialized via lock.

        Spawns a fresh registry agent. If a new field is added, the agent
        produces the updated stub (validated by solc) and we write it to the store.
        Returns a description of the field to use, or a rejection message.
        """
        if nm not in self._state_by_id:
            return f"Error: unknown contract identifier: {nm}"
        state = self._state_by_id[nm]
        async with state._lock:
            stub_content = state._content
            stub_path = state._path
            durable_state = await self._read_metadata(nm)
            assert durable_state.content == stub_content

            result = await run_registry_agent(
                request=purpose,
                stub_content=stub_content,
                stub_path=stub_path,
                field_metadata=durable_state.fields,
                interface=state._interface,
                solc_version=self._solc_version,
                builder=self._builder,
                assembler=self._assembler,
                within_tool=within_tool,
                recursion_limit=self._recursion_limit,
            )

            if result.rejected:
                return f"Field request was rejected: {result.description}"

            if result.is_new:
                durable_state.fields.append(FieldSpec(
                    name=result.field_name,
                    type=result.field_type,
                    description=result.description,
                ))
                durable_state.content = result.updated_stub
                state._content = result.updated_stub
                self._mirror_by_path[state._path] = result.updated_stub
                await self._write_metadata(nm, durable_state)
                evt: StubUpdate = {
                    "type": "stub_update",
                    "contract_id": nm,
                    "stub": result.updated_stub,
                }
                get_stream_writer()(evt)

            return f"Use field {result.field_name}"

    def get_tools(self, contract_identifier: SolidityIdentifier) -> "list[BaseTool]":
        """Return tools for injection into the property agent authoring the spec
        for ``contract_identifier``. The agent is primarily responsible for its
        own contract, but may read and request fields in any other contract's
        stub to support cross-contract specs — so the tools take the target
        contract identifier as an explicit LLM-visible parameter rather than
        closing over the author's own.
        """
        registry = self
        home_contract = contract_identifier

        @tool_display(
            lambda d: f"Requesting stub field in {d['contract_identifier']}: {d['purpose']}",
            "Stub field result",
        )
        class RequestStubField(WithAsyncImplementation[str], WithInjectedId):
            """Request a storage variable in a contract's verification stub.

            Describe what you need the field for (e.g., "a mapping to track
            per-user deposit amounts"). The registry will either return an
            existing field that serves the same purpose, or create a new one.
            Returns the field name to use in your CVL specification.

            Pass your own contract name to add a field to your own stub; pass
            another contract's identifier to request a field there (needed for
            cross-contract specs that depend on state in a dependency).

            You may *NOT* use this tool to request any change to the stub besides a new storage field.
            """
            contract_identifier: SolidityIdentifier = PydanticField(
                description=(
                    f"The contract whose stub should gain the field. You are "
                    f"authoring the spec for '{home_contract}' — pass that identifier "
                    f"for your own stub, or another registered contract's solidity identifier "
                    f"when the field belongs to a dependency."
                )
            )
            purpose: str = PydanticField(
                description="Natural language description of what the field should track, along with its " \
                "expected shape/type (address, signed int, unsigned int, etc.)"
            )

            @override
            async def run(self) -> str:
                return await registry.request_field(
                    self.contract_identifier, self.purpose,
                    within_tool=self.tool_call_id,
                )

        return [
            RequestStubField.as_tool("request_stub_field"),
        ]


# ---------------------------------------------------------------------------
# FileRegistry — declarative registry of source files to pull into the conf
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileEntry:
    """A registered source file, optionally tagged with the Solidity identifier
    to compile against. ``solidity_identifier=None`` means "use the file's stem
    as the identifier" (Certora's default behavior)."""
    path: str
    solidity_identifier: SolidityIdentifier | None = None

    def as_prover_arg(self) -> str:
        if self.solidity_identifier is None:
            return self.path
        return f"{self.path}:{self.solidity_identifier}"


@dataclass
class FileRegistry:
    """Per-contract registry of source files to pull into the Certora conf
    ``files`` list.

    Unlike ``StubRegistry``, this is a plain registration — no agent decisions,
    but ``register`` rejects paths that don't exist in the layered FS the
    registry closes over (``_materializer``). Each contract gets its own KV
    entry under ``_namespace`` keyed by contract name; ``read_all_contracts``
    enumerates via ``asearch``. The lock serializes the read-modify-write that
    backs ``register``'s per-path dedupe within a single contract.
    """
    _store: BaseStore
    _materializer: Materializer
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _namespace: tuple[str, ...] = ()

    @staticmethod
    async def acreate(
        store: BaseStore,
        namespace: tuple[str, ...],
        materializer: Materializer,
    ) -> "FileRegistry":
        return FileRegistry(
            _store=store, _materializer=materializer, _namespace=namespace,
        )

    async def _read_contract(self, contract_identifier: SolidityIdentifier) -> list[FileEntry]:
        item = await self._store.aget(self._namespace, contract_identifier)
        if item is None:
            return []
        return [
            FileEntry(path=e["path"], solidity_identifier=e["solidity_identifier"])
            for e in item.value["files"]
        ]

    async def _write_contract(self, contract_identifier: SolidityIdentifier, files: list[FileEntry]) -> None:
        _log.debug(
            "FileRegistry._write_contract: ns=%r contract=%s entries=%s",
            self._namespace, contract_identifier,
            [{"path": e.path, "ident": e.solidity_identifier} for e in files],
        )
        await self._store.aput(self._namespace, contract_identifier, {
            "files": [
                {"path": e.path, "solidity_identifier": e.solidity_identifier}
                for e in files
            ],
        })

    async def read_all(self, contract_identifier: SolidityIdentifier) -> list[str]:
        """Read the prover-ready file arguments registered for ``contract_identifier``.

        Each entry is either ``path`` or ``path:Identifier`` depending on
        whether a Solidity identifier was supplied at registration.
        """
        return [e.as_prover_arg() for e in await self._read_contract(contract_identifier)]

    async def register(
        self,
        contract_identifier: SolidityIdentifier,
        path: str,
        solidity_identifier: SolidityIdentifier | None = None,
    ) -> str:
        """Register ``path`` as a compilation-unit file for ``contract_identifier``.

        Rejects paths that don't exist in the layered FS this registry closes
        over. If ``path`` is already registered for this contract, the
        existing entry's ``solidity_identifier`` is overwritten (latest call
        wins). Each path appears at most once per contract.
        """
        _log.debug(
            "FileRegistry.register: ns=%r contract=%s path=%s ident=%s",
            self._namespace, contract_identifier, path, solidity_identifier,
        )
        if self._materializer.get(path) is None:
            _log.debug(
                "FileRegistry.register: REJECTED ns=%r contract=%s "
                "path=%s (not in materializer)",
                self._namespace, contract_identifier, path,
            )
            return (
                f"Cannot register {path}: that file does not exist in the "
                f"project tree. Use your source tools (`list_files`, "
                f"`grep_files`) to find the correct path before registering."
            )
        async with self._lock:
            files = await self._read_contract(contract_identifier)
            new_entry = FileEntry(path=path, solidity_identifier=solidity_identifier)
            for i, existing in enumerate(files):
                if existing.path == path:
                    if existing == new_entry:
                        return f"{new_entry.as_prover_arg()} is already registered for {contract_identifier}."
                    files[i] = new_entry
                    await self._write_contract(contract_identifier, files)
                    return (
                        f"Updated {path} for {contract_identifier}: "
                        f"{existing.as_prover_arg()} -> {new_entry.as_prover_arg()}."
                    )
            files.append(new_entry)
            await self._write_contract(contract_identifier, files)
        return f"Registered {new_entry.as_prover_arg()} for {contract_identifier}."

    def get_tools(self, contract_identifier: SolidityIdentifier) -> list[BaseTool]:
        """Return tools scoped to ``contract_identifier`` for injection into that
        contract's property agents.
        """
        registry = self

        @tool_display(
            lambda d: f"Registering spec file: {d['path']}",
            "Spec file registration result",
        )
        class RegisterSpecFile(WithAsyncImplementation[str]):
            """Register a Solidity source file that must be pulled into the
            verification task for the spec you're authoring. Use this for any
            contract source the spec references, e.g.,
            other stubs, extant code the stubs don't cover (if applicable)

            The path must be project-relative and point to a ``.sol`` file
            already present in the source tree (inspect the tree with the
            source tools if unsure). Registration of a path that does not
            exist in the project tree is rejected — verify with `list_files`
            or `get_file` before calling this tool. Registering the same path twice is a no-op.
            """
            path: str = PydanticField(
                description="Project-relative path to a .sol file"
            )

            @override
            async def run(self) -> str:
                return await registry.register(contract_identifier, self.path)

        @tool_display("Listing registered source files", None)
        class ListSpecFiles(WithAsyncImplementation[str]):
            """List every Solidity source file currently registered for this
            contract's verification unit unit.
            """

            @override
            async def run(self) -> str:
                files = await registry.read_all(contract_identifier)
                if not files:
                    return "No files registered yet."
                return "\n".join(f"- {p}" for p in files)

        return [
            RegisterSpecFile.as_tool("register_verification_file"),
            ListSpecFiles.as_tool("list_verification_files"),
        ]

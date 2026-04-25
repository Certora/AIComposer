"""
Semantic registry agent for shared stub field management.

Serializes stub edits via an asyncio.Lock. Each field request spawns a fresh
registry agent that receives accumulated field metadata, the current stub,
and the interface, then decides whether to reuse an existing field or add a
new one. When adding a new field, the agent produces the updated stub source
which is validated against the Solidity compiler before acceptance.
"""

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import NotRequired, override, Iterable
import pathlib

from pydantic import BaseModel, Field as PydanticField

from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from graphcore.graph import FlowInput
from graphcore.tools.schemas import WithAsyncImplementation

from composer.spec.context import WorkflowContext, PlainBuilder, CVLOnlyBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.natspec.pipeline_events import StubUpdate
from composer.spec.natspec.models import (
    InterfaceResult,
    LocatedStubDeclaration,
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
    stub_fields: dict[str, list[FieldSpec]] = PydanticField(default_factory=dict)


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

def _compile_stub(
    stub: str,
    interfaces: InterfaceResult,
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
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        interfaces.dump_to_path(root)
        stub_abs = root / stub_path
        stub_abs.parent.mkdir(parents=True, exist_ok=True)
        stub_abs.write_text(stub)
        try:
            proc = subprocess.run(
                [solc_name, stub_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdir,
            )
        except FileNotFoundError:
            return f"Solidity compiler {solc_name} not found"
        if proc.returncode != 0:
            return f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        return None


# ---------------------------------------------------------------------------
# Registry agent
# ---------------------------------------------------------------------------

async def run_registry_agent(
    contract_name: str,
    request: str,
    stub_content: str,
    stub_path: str,
    field_metadata: FieldMetadata,
    interface: InterfaceResult,
    solc_version: str,
    builder: PlainBuilder,
    assembler: Assembler,
) -> RegistryResult:
    """Spawn a fresh registry agent to handle a single field request.

    The agent decides whether to reuse an existing field or add a new one.
    When adding, the result validator compiles the updated stub with solc,
    rejecting malformed output so the agent can retry.
    """

    class ST(MessagesState):
        result: NotRequired[RegistryResult]

    def validate_result(_s: ST, res: RegistryResult) -> str | None:
        if res.is_new:
            if not res.field_type:
                return "When proposing a new field, you must provide field_type (the Solidity type)."
            if not res.description:
                return "When proposing a new field, you must provide field_description."
            if not res.updated_stub:
                return "When proposing a new field, you must provide updated_stub (the complete source code)."
            compile_err = _compile_stub(res.updated_stub, interface, solc_version, stub_path)
            if compile_err is not None:
                return (
                    f"The updated stub does not compile. Fix the issue and try again.\n"
                    f"{compile_err}"
                )
        return None

    workflow = bind_standard(
        builder, ST, validator=validate_result,
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        "You are a Solidity stub field manager. You decide whether a requested "
        "storage variable already exists in the stub (semantically equivalent) "
        "or whether a new field needs to be added. When adding a new field, you "
        "produce the complete updated stub source code."
    ).with_initial_prompt_template(
        "registry_prompt.j2",
    ).compile_async()

    input_parts: list[str | dict] = [
        "The field request is:",
        request,
        "The current stub source code is:",
        stub_content,
        "The interface for this stub is",
        interface.name_to_interface[contract_name].content
    ]

    if (flds := field_metadata.stub_fields.get(contract_name, [])):
        field_lines = "\n".join(
            f"  - `{f.type} {f.name}`: {f.description}"
            for f in flds
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
        recursion_limit=20,
        description="Stub update",
    )
    assert "result" in res
    return res["result"]


# ---------------------------------------------------------------------------
# StubRegistry — serializes stub edits
# ---------------------------------------------------------------------------

STUB_STORE_KEY = "stub_content"
FIELDS_STORE_KEY = "stub_fields"
SPEC_FILES_STORE_KEY = "spec_files"


@dataclass
class StubRegistry:
    """Manages the shared stub and its field registry.

    All field mutations are serialized via an asyncio.Lock. Reads are lock-free.
    Field metadata is stored in BaseStore alongside the stub content.
    """
    _store: BaseStore
    _builder: PlainBuilder | CVLOnlyBuilder
    _interface: InterfaceResult
    _solc_version: str
    _assembler: Assembler
    _mirror_by_path: dict[str, str]
    _mirror_by_name: dict[str, str]
    _path_by_name: dict[str, str]
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _namespace: tuple[str, ...] = ()

    @staticmethod
    async def acreate(
        store: BaseStore,
        namespace: tuple[str, ...],
        builder: PlainBuilder | CVLOnlyBuilder,
        interface: InterfaceResult,
        interface_only_mat: Assembler,
        initial_stubs: dict[str, StubDeclarationModel],
        solc_version: str,
    ) -> "StubRegistry":
        """Create or resume a StubRegistry.

        If the store already contains stub content and field metadata for this
        namespace, they are preserved (resume after crash/restart). Otherwise
        the store is initialized with the provided ``initial_stubs`` — each
        serialized as ``{"path", "content", "solidity_identifier"}``. Paths
        are fixed at initialization; only content changes during field updates.
        """
        curr_res = await store.aget(namespace, STUB_STORE_KEY)
        if curr_res is None:
            await store.aput(namespace, STUB_STORE_KEY, {
                nm: {
                    "path": decl.path,
                    "content": decl.content,
                    "solidity_identifier": decl.solidity_identifier,
                }
                for nm, decl in initial_stubs.items()
            })
            curr_content_path = {
                decl.path: decl.content for decl in initial_stubs.values()
            }
            curr_content_name = {
                nm: decl.content for (nm, decl) in initial_stubs.items()
            }
            curr_path_by_name = {
                nm: decl.path for (nm, decl) in initial_stubs.items()
            }
        else:
            curr_content_path = {
                t["path"]: t["content"] for t in curr_res.value.values()
            }
            curr_content_name = {
                nm: t["content"] for (nm, t) in curr_res.value.items()
            }
            curr_path_by_name = {
                nm: t["path"] for (nm, t) in curr_res.value.items()
            }
        if await store.aget(namespace, FIELDS_STORE_KEY) is None:
            await store.aput(namespace, FIELDS_STORE_KEY, {k: [] for k in initial_stubs.keys()})
        return StubRegistry(
            _store=store,
            _builder=builder,
            _interface=interface,
            _solc_version=solc_version,
            _assembler=interface_only_mat,
            _namespace=namespace,
            _mirror_by_path=curr_content_path,
            _mirror_by_name=curr_content_name,
            _path_by_name=curr_path_by_name,
        )
    
    # FS backend stuff

    def get(self, path: str) -> str | None:
        return self._mirror_by_path.get(path, None)
    
    def list(self) -> Iterable[str]:
        return self._mirror_by_path.keys()
    
    async def dump_to(self, target: pathlib.Path):
        for (k, v) in self._mirror_by_path.items():
            full_path = target / k
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(v)

    def read_stub(self, nm: str) -> str:
        """Read current stub content (no lock needed)."""
        return self._mirror_by_name[nm]

    async def read_all_stubs(self) -> dict[str, LocatedStubDeclaration]:
        """Read every stub as full declarations. Empty dict if registry isn't initialized."""
        item = await self._store.aget(self._namespace, STUB_STORE_KEY)
        if item is None:
            return {}
        return {
            nm: LocatedStubDeclaration(
                path=entry["path"],
                content=entry["content"],
                solidity_identifier=entry["solidity_identifier"],
            )
            for nm, entry in item.value.items()
        }

    async def _read_field_metadata(self) -> FieldMetadata:
        item = await self._store.aget(self._namespace, FIELDS_STORE_KEY)
        if item is None:
            return FieldMetadata()
        return FieldMetadata.model_validate(item.value)

    async def read_fields(self) -> dict[str, "list[FieldSpec]"]:
        """Snapshot the per-contract stub fields requested during this pipeline run.

        Returns ``{contract_name: [FieldSpec, ...]}``. Empty for contracts that never
        received a field request. Safe to call at any point after ``acreate``; intended
        for end-of-pipeline export so the codegen driver knows what storage layout
        each contract's implementation must carry.
        """
        return (await self._read_field_metadata()).stub_fields

    async def _write_field_metadata(self, metadata: FieldMetadata) -> None:
        await self._store.aput(self._namespace, FIELDS_STORE_KEY, metadata.model_dump())

    async def _write_stub(self, nm: str, content: str) -> None:
        """Update the stub content for ``nm`` in place. Path stays fixed."""
        it = await self._store.aget(self._namespace, STUB_STORE_KEY)
        assert it is not None
        to_put = {k: dict(v) for k, v in it.value.items()}
        to_put[nm]["content"] = content
        self._mirror_by_name[nm] = content
        self._mirror_by_path[to_put[nm]["path"]] = content
        await self._store.aput(self._namespace, STUB_STORE_KEY, to_put)

    async def request_field(self, nm: str, purpose: str) -> str:
        """Request a stub field for a given purpose. Serialized via lock.

        Spawns a fresh registry agent. If a new field is added, the agent
        produces the updated stub (validated by solc) and we write it to the store.
        Returns a description of the field to use, or a rejection message.
        """
        async with self._lock:
            stub_content = self.read_stub(nm)
            stub_path = self._path_by_name[nm]
            field_metadata = await self._read_field_metadata()

            result = await run_registry_agent(
                contract_name=nm,
                request=purpose,
                stub_content=stub_content,
                stub_path=stub_path,
                field_metadata=field_metadata,
                interface=self._interface,
                solc_version=self._solc_version,
                builder=self._builder,
                assembler=self._assembler,
            )

            if result.rejected:
                return f"Field request was rejected: {result.description}"

            if result.is_new:
                if nm not in field_metadata.stub_fields:
                    field_metadata.stub_fields[nm] = []
                field_metadata.stub_fields[nm].append(FieldSpec(
                    name=result.field_name,
                    type=result.field_type,
                    description=result.description,
                ))
                await self._write_field_metadata(field_metadata)
                await self._write_stub(nm, result.updated_stub)
                evt: StubUpdate = {
                    "type": "stub_update",
                    "contract_id": nm,
                    "stub": result.updated_stub,
                }
                get_stream_writer()(evt)

            return f"Use field {result.field_name}"

    def get_tools(self, contract_name: str) -> "list[BaseTool]":
        """Return tools for injection into the property agent authoring the spec
        for ``contract_name``. The agent is primarily responsible for its own
        contract, but may read and request fields in any other contract's stub
        to support cross-contract specs — so the tools take the target contract
        name as an explicit LLM-visible parameter rather than closing over the
        author's own name.
        """
        registry = self
        home_contract = contract_name

        @tool_display(
            lambda d: f"Requesting stub field in {d['contract_name']}: {d['purpose']}",
            "Stub field result",
        )
        class RequestStubField(WithAsyncImplementation[str]):
            """Request a storage variable in a contract's verification stub.

            Describe what you need the field for (e.g., "a mapping to track
            per-user deposit amounts"). The registry will either return an
            existing field that serves the same purpose, or create a new one.
            Returns the field name to use in your CVL specification.

            Pass your own contract name to add a field to your own stub; pass
            another contract's name to request a field there (needed for
            cross-contract specs that depend on state in a dependency).

            You may *NOT* use this tool to request any change to the stub besides a new storage field.
            """
            contract_name: str = PydanticField(
                description=(
                    f"The contract whose stub should gain the field. You are "
                    f"authoring the spec for '{home_contract}' — pass that name "
                    f"for your own stub, or another registered contract's name "
                    f"when the field belongs to a dependency."
                )
            )
            purpose: str = PydanticField(
                description="Natural language description of what the field should track"
            )

            @override
            async def run(self) -> str:
                return await registry.request_field(self.contract_name, self.purpose)

        return [
            RequestStubField.as_tool("request_stub_field"),
        ]


# ---------------------------------------------------------------------------
# FileRegistry — declarative registry of source files to pull into the conf
# ---------------------------------------------------------------------------


@dataclass
class FileRegistry:
    """Per-contract registry of source files to pull into the Certora conf
    ``files`` list.

    Unlike ``StubRegistry``, this is a plain registration — no agent decisions,
    no validation. Each contract's spec may need a different compilation unit,
    so registrations are scoped by contract name (matching how ``StubRegistry``
    scopes stubs). Writes are serialized via a lock; reads are lock-free.
    """
    _store: BaseStore
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _namespace: tuple[str, ...] = ()

    @staticmethod
    async def acreate(store: BaseStore, namespace: tuple[str, ...]) -> "FileRegistry":
        if await store.aget(namespace, SPEC_FILES_STORE_KEY) is None:
            await store.aput(namespace, SPEC_FILES_STORE_KEY, {})
        return FileRegistry(_store=store, _namespace=namespace)

    async def _read_map(self) -> dict[str, list[str]]:
        item = await self._store.aget(self._namespace, SPEC_FILES_STORE_KEY)
        if item is None:
            return {}
        return {k: list(v) for k, v in item.value.items()}

    async def read_all(self, contract_name: str) -> list[str]:
        """Read the files registered for ``contract_name``."""
        return (await self._read_map()).get(contract_name, [])

    async def read_all_contracts(self) -> dict[str, list[str]]:
        """Read the full registration map (contract → files)."""
        return await self._read_map()

    async def register(self, contract_name: str, path: str) -> str:
        """Register ``path`` as a compilation-unit file for ``contract_name``. Idempotent."""
        async with self._lock:
            current = await self._read_map()
            files = current.setdefault(contract_name, [])
            if path in files:
                return f"{path} is already registered for {contract_name}."
            files.append(path)
            await self._store.aput(self._namespace, SPEC_FILES_STORE_KEY, current)
        return f"Registered {path} for {contract_name}."

    def get_tools(self, contract_name: str) -> list[BaseTool]:
        """Return tools scoped to ``contract_name`` for injection into that
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
            source tools if unsure). Registering the same path twice is a no-op.
            """
            path: str = PydanticField(
                description="Project-relative path to a .sol file"
            )

            @override
            async def run(self) -> str:
                return await registry.register(contract_name, self.path)

        @tool_display("Listing registered spec files", None)
        class ListSpecFiles(WithAsyncImplementation[str]):
            """List every Solidity source file currently registered for this
            contract's spec compilation unit.
            """

            @override
            async def run(self) -> str:
                files = await registry.read_all(contract_name)
                if not files:
                    return "No files registered yet."
                return "\n".join(f"- {p}" for p in files)

        return [
            RegisterSpecFile.as_tool("register_verification_file"),
            ListSpecFiles.as_tool("list_verification_files"),
        ]

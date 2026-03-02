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
from typing import NotRequired, override

from pydantic import BaseModel, Field as PydanticField

from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore

from graphcore.graph import FlowInput
from graphcore.tools.schemas import WithAsyncImplementation

from composer.spec.context import WorkflowContext, PlainBuilder, CVLOnlyBuilder
from composer.spec.graph_builder import bind_standard, run_to_completion


# ---------------------------------------------------------------------------
# Field metadata schema
# ---------------------------------------------------------------------------

class FieldSpec(BaseModel):
    name: str = PydanticField(description="The Solidity field name")
    type: str = PydanticField(description="The Solidity type (e.g., 'mapping(address => uint256)')")
    description: str = PydanticField(description="What this field tracks")


class FieldMetadata(BaseModel):
    fields: list[FieldSpec] = PydanticField(default_factory=list)


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
    field_description: str = PydanticField(
        default="",
        description="A short description of what this field tracks. "
        "Required when is_new is true.",
    )
    updated_stub: str = PydanticField(
        default="",
        description="The complete updated stub source code with the new field declaration added. "
        "This must be the FULL source, not a diff. Required when is_new is true.",
    )


# ---------------------------------------------------------------------------
# Stub compilation check
# ---------------------------------------------------------------------------

def _compile_stub(stub: str, interface: str, solc_version: str) -> str | None:
    """Compile stub against interface with solc. Returns None on success, error string on failure."""
    solc_name = f"solc{solc_version}"
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir)
        (root / "Intf.sol").write_text(interface)
        (root / "Impl.sol").write_text(stub)
        try:
            proc = subprocess.run(
                [solc_name, str(root / "Impl.sol")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
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
    request: str,
    stub_content: str,
    field_metadata: FieldMetadata,
    interface: str,
    solc_version: str,
    builder: PlainBuilder | CVLOnlyBuilder,
    ctx: WorkflowContext,
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
            if not res.field_description:
                return "When proposing a new field, you must provide field_description."
            if not res.updated_stub:
                return "When proposing a new field, you must provide updated_stub (the complete source code)."
            compile_err = _compile_stub(res.updated_stub, interface, solc_version)
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
    ).compile_async(
        checkpointer=ctx.checkpointer
    )

    input_parts: list[str | dict] = [
        "The field request is:",
        request,
        "The current stub source code is:",
        stub_content,
        "The current interface source code is:",
        interface,
    ]
    if field_metadata.fields:
        field_lines = "\n".join(
            f"  - `{f.type} {f.name}`: {f.description}"
            for f in field_metadata.fields
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
        thread_id=ctx.uniq_thread_id(),
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


@dataclass
class StubRegistry:
    """Manages the shared stub and its field registry.

    All field mutations are serialized via an asyncio.Lock. Reads are lock-free.
    Field metadata is stored in BaseStore alongside the stub content.
    """
    _store: BaseStore
    _builder: PlainBuilder | CVLOnlyBuilder
    _ctx: WorkflowContext
    _interface: str
    _solc_version: str
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _namespace: tuple[str, ...] = ()

    @staticmethod
    def create(
        store: BaseStore,
        namespace: tuple[str, ...],
        builder: PlainBuilder | CVLOnlyBuilder,
        ctx: WorkflowContext,
        interface: str,
        initial_stub: str,
        solc_version: str,
    ) -> "StubRegistry":
        """Create a new StubRegistry with initial stub content."""
        store.put(namespace, STUB_STORE_KEY, {"content": initial_stub})
        store.put(namespace, FIELDS_STORE_KEY, {"fields": []})
        return StubRegistry(
            _store=store,
            _builder=builder,
            _ctx=ctx,
            _interface=interface,
            _solc_version=solc_version,
            _namespace=namespace,
        )

    def read_stub(self) -> str:
        """Read current stub content (no lock needed)."""
        item = self._store.get(self._namespace, STUB_STORE_KEY)
        if item is None:
            return ""
        return item.value["content"]

    def _read_field_metadata(self) -> FieldMetadata:
        item = self._store.get(self._namespace, FIELDS_STORE_KEY)
        if item is None:
            return FieldMetadata()
        return FieldMetadata.model_validate(item.value)

    def _write_field_metadata(self, metadata: FieldMetadata) -> None:
        self._store.put(self._namespace, FIELDS_STORE_KEY, metadata.model_dump())

    def _write_stub(self, content: str) -> None:
        self._store.put(self._namespace, STUB_STORE_KEY, {"content": content})

    async def request_field(self, purpose: str) -> str:
        """Request a stub field for a given purpose. Serialized via lock.

        Spawns a fresh registry agent. If a new field is added, the agent
        produces the updated stub (validated by solc) and we write it to the store.
        Returns the field name to use.
        """
        async with self._lock:
            stub_content = self.read_stub()
            field_metadata = self._read_field_metadata()

            result = await run_registry_agent(
                request=purpose,
                stub_content=stub_content,
                field_metadata=field_metadata,
                interface=self._interface,
                solc_version=self._solc_version,
                builder=self._builder,
                ctx=self._ctx,
            )

            if result.is_new:
                field_metadata.fields.append(FieldSpec(
                    name=result.field_name,
                    type=result.field_type,
                    description=result.field_description,
                ))
                self._write_field_metadata(field_metadata)
                self._write_stub(result.updated_stub)

            return result.field_name

    def get_tools(self) -> list[BaseTool]:
        """Return tools for injection into property agents."""
        registry = self

        class ReadStubTool(WithAsyncImplementation[str]):
            """Read the current shared verification stub source code."""
            @override
            async def run(self) -> str:
                return registry.read_stub()

        class RequestStubField(WithAsyncImplementation[str]):
            """Request a storage variable in the shared verification stub.
            Describe what you need the field for (e.g., "a mapping to track per-user
            deposit amounts"). The registry will either return an existing field that
            serves the same purpose, or create a new one.
            Returns the field name to use in your CVL specification.
            """
            purpose: str = PydanticField(
                description="Natural language description of what the field should track"
            )

            @override
            async def run(self) -> str:
                name = await registry.request_field(self.purpose)
                return f"Use field: {name}"

        return [
            ReadStubTool.as_tool("read_stub"),
            RequestStubField.as_tool("request_stub_field"),
        ]

from pydantic import BaseModel, Field


class FileConfiguration(BaseModel):
    spec_files: list[str] = Field(description="Relative path to CVL spec files (.spec)")
    interface_file: str = Field(description="Relative path to Solidity interface file (.sol)")
    system_doc: str = Field(description="Relative path to system/design document")

class CodegenConfiguration(FileConfiguration):
    """
    A rich codegen input.

    ``prover_conf`` is intentionally *not* on this configuration —
    prover overrides are an orthogonal runtime concern (independent of
    which spec/interface/system-doc set is being verified) and live on
    ``CommonCodeGen`` (assistant) / ``--prover-conf`` (CLI) instead.
    """
    kickstart_context: str | None = Field(
        default=None,
        description=(
            "Free-form briefing fed verbatim into the codegen agent's initial prompt. "
            "Use this to pass forward whatever context the agent needs to start work that "
            "isn't already covered by spec / interface / system_doc. Typical contents when "
            "this codegen run follows a natspec invocation: the agent-chosen implementation "
            "path (so the codegen agent writes its file at the same location the stub "
            "occupied — STRONGLY recommended), the natspec stub source as scaffold to "
            "evolve, the table of required storage fields with their types and purposes, "
            "and any dependency / tag notes from the implementation plan. The codegen "
            "agent treats anything in this field as authoritative orchestrator briefing."
        ),
    )
    implementation_path : str | None = Field(description="The recommended (relative) path to the implemented component", default=None)
    contract_name : str | None = Field(description="The recommended Solidity identifier to use for the generated contract", default=None)

class CmdlineCodegenConfiguration(CodegenConfiguration):
    source_root: str = Field(description="REQUIRED The absolute path to the root where code generation must take place")

from pydantic import BaseModel, Field

_PROVER_CONF_DESC = (
    "Certora config object (the JSON shape of certora.conf) whose keys — packages, link, solc_args, "
    "solc_via_ir, optimistic_loop, rule_sanity, etc. — are merged into every prover / typecheck "
    "invocation. Dynamic keys (`files`, `verify`, `solc`) are always set by the pipeline and will "
    "override whatever is in this object. Pass inline as a dict; the assistant will usually "
    "construct it from the codebase but users may override manually."
)


class FileConfiguration(BaseModel):
    spec_files: list[str] = Field(description="Relative path to CVL spec files (.spec)")
    interface_file: str = Field(description="Relative path to Solidity interface file (.sol)")
    system_doc: str = Field(description="Relative path to system/design document")

class CodegenConfiguration(FileConfiguration):
    """
    A rich codegen input.
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
    prover_conf: dict | None = Field(description=_PROVER_CONF_DESC, default=None)

class CmdlineCodegenConfiguration(CodegenConfiguration):
    source_root: str = Field(description="REQUIRED The absolute path to the root where code generation must take place")

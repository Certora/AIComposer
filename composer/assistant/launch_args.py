from pydantic import BaseModel, Field

_PROVER_CONF_DESC = (
    "Certora config object (the JSON shape of certora.conf) whose keys — packages, link, solc_args, "
    "solc_via_ir, optimistic_loop, rule_sanity, etc. — are merged into every prover / typecheck "
    "invocation. Dynamic keys (`files`, `verify`, `solc`) are always set by the pipeline and will "
    "override whatever is in this object. Pass inline as a dict; the assistant will usually "
    "construct it from the codebase but users may override manually."
)


class CommonCodeGen(BaseModel):
    prompt_addition: str | None = Field(description="Extra instructions for the codegen agent.", default=None)

class LaunchCodegenArgs(CommonCodeGen):
    spec_file: str = Field(description="Relative path to CVL spec file (.spec)")
    interface_file: str = Field(description="Relative path to Solidity interface file (.sol)")
    system_doc: str = Field(description="Relative path to system/design document")
    memory_namespace: str | None = Field(description="Namespace for persistent agent memory. When set, memory persists across thread changes (including crashes and relaunches).", default=None)
    resume_work_key: str | None = Field(description="Key to recover in-progress work from a crashed run. Provided in the crash result of a previous launch.", default=None)
    source_root: str | None = Field(
        default=None,
        description=(
            "Path to an existing codebase to use as the VFS underlay. When set, agents see "
            "existing files read-only and can layer new files on top."
        ),
    )
    prover_conf: dict | None = Field(default=None, description=_PROVER_CONF_DESC)
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


class LaunchResumeArgs(CommonCodeGen):
    thread_id: str = Field(description="Thread ID of the previous workflow (from /memories/)")
    working_dir: str = Field(description="Path to directory with current/updated files")
    commentary: str = Field(description="Description of changes since last run", default="")
    memory_namespace: str | None = Field(description="Namespace for persistent agent memory. Should match the memory_namespace used in the original codegen run.", default=None)
    resume_work_key: str | None = Field(description="Key to recover in-progress work from a crashed run. Provided in the crash result of a previous launch.", default=None)
    prover_conf: dict | None = Field(default=None, description=_PROVER_CONF_DESC)


class LaunchNatSpecArgs(BaseModel):
    input_file: str = Field(description="Relative path to design/system document")
    solc_version: str = Field(description="Solidity compiler version (e.g. '8.21')")
    cache_namespace: str = Field(description="Namespace for cross-run caching (e.g. 'mytoken'). Enables reuse of prior agent work.")
    memory_namespace: str = Field(description="Namespace for persistent agent memory (defaults to thread ID if empty)")
    source_root: str | None = Field(
        default=None,
        description="Path to an existing codebase root. When set, natspec runs in source-aware mode.",
    )
    forbidden_read: str | None = Field(
        default=None,
        description=(
            "Regex of paths source tools may not read. Defaults to the standard FS_FORBIDDEN_READ "
            "pattern when source_root is set."
        ),
    )
    prover_conf: dict | None = Field(default=None, description=_PROVER_CONF_DESC)
    output_root: str | None = Field(
        default=None,
        description=(
            "Workspace-relative directory under which the natspec run's `implementation_plan.json` "
            "is written, plus any generated files in no-IDE fallback. When unset, defaults to "
            "`natspec_output/<cache_namespace>` (or `natspec_output/` if no cache_namespace was "
            "supplied). The plan path is reported back in the launch result so subsequent "
            "codegen invocations can locate it."
        ),
    )
    interactive: bool = Field(
        default=False,
        description=(
            "When True, opens a per-component conversation channel during bug analysis so "
            "the user can refine the extracted property list interactively before CVL "
            "generation begins. Each component's channel is its own focusable panel in the "
            "TUI. Set when the user has expressed intent to review properties before "
            "formalization; leave false for batch / hands-off runs."
        ),
    )

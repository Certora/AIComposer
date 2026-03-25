from pydantic import BaseModel, Field

class CommonCodeGen(BaseModel):
    prompt_addition: str | None = Field(description="Extra instructions for the codegen agent.", default=None)

class LaunchCodegenArgs(CommonCodeGen):
    spec_file: str = Field(description="Relative path to CVL spec file (.spec)")
    interface_file: str = Field(description="Relative path to Solidity interface file (.sol)")
    system_doc: str = Field(description="Relative path to system/design document")
    memory_namespace: str | None = Field(description="Namespace for persistent agent memory. When set, memory persists across thread changes (including crashes and relaunches).", default=None)
    resume_work_key: str | None = Field(description="Key to recover in-progress work from a crashed run. Provided in the crash result of a previous launch.", default=None)


class LaunchResumeArgs(CommonCodeGen):
    thread_id: str = Field(description="Thread ID of the previous workflow (from /memories/)")
    working_dir: str = Field(description="Path to directory with current/updated files")
    commentary: str = Field(description="Description of changes since last run", default="")
    memory_namespace: str | None = Field(description="Namespace for persistent agent memory. Should match the memory_namespace used in the original codegen run.", default=None)
    resume_work_key: str | None = Field(description="Key to recover in-progress work from a crashed run. Provided in the crash result of a previous launch.", default=None)


class LaunchNatSpecArgs(BaseModel):
    input_file: str = Field(description="Relative path to design/system document")
    solc_version: str = Field(description="Solidity compiler version (e.g. '8.21')")
    cache_namespace: str = Field(description="Namespace for cross-run caching (e.g. 'mytoken'). Enables reuse of prior agent work.")
    memory_namespace: str = Field(description="Namespace for persistent agent memory (defaults to thread ID if empty)")

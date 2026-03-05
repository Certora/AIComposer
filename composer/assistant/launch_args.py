from pydantic import BaseModel, Field


class LaunchCodegenArgs(BaseModel):
    spec_file: str = Field(description="Relative path to CVL spec file (.spec)")
    interface_file: str = Field(description="Relative path to Solidity interface file (.sol)")
    system_doc: str = Field(description="Relative path to system/design document")


class LaunchResumeArgs(BaseModel):
    thread_id: str = Field(description="Thread ID of the previous workflow (from /memories/)")
    working_dir: str = Field(description="Path to directory with current/updated files")
    commentary: str = Field(description="Description of changes since last run", default="")


class LaunchNatSpecArgs(BaseModel):
    input_file: str = Field(description="Relative path to design/system document")
    contract_name: str = Field(description="Contract name for the generated spec")
    solc_version: str = Field(description="Solidity compiler version (e.g. '8.21')")
    cache_namespace: str = Field(description="Namespace for cross-run caching (e.g. 'mytoken'). Enables reuse of prior agent work.")
    memory_namespace: str = Field(description="Namespace for persistent agent memory (defaults to thread ID if empty)")

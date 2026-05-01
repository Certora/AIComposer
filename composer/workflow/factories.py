
from langchain_core.tools import BaseTool
import pathlib

from graphcore.tools.vfs import vfs_tools, VFSAccessor, VFSToolConfig, VFSState

from composer.core.state import AIComposerState




def get_memory_ns(thread_id: str, ns: str) -> str:
    return f"ai-composer-{thread_id}-{ns}"

def exclude_ts_and_natspec(p: pathlib.PurePath) -> bool:
    suff = p.suffix
    lower_last = p.parts[-1].lower()
    # o7 for our hardhat friends
    return (suff == ".js" or 
            suff == ".ts" or
            (p.parts[0] == "natspec_output" and suff == ".sol") or
            p.parts[-1] == "package.json" or
            suff == ".map" or 
            suff == ".json" or
            suff == ".mjs" or
            lower_last == "readme.md" or
            lower_last == "license"
            )

def get_vfs_tools(
    fs_layer: str | None,
    immutable: bool
) -> tuple[list[BaseTool], VFSAccessor[VFSState]]:
    if immutable:
        return vfs_tools(VFSToolConfig(
            fs_layer=fs_layer,
            immutable=True,
            global_exclude=exclude_ts_and_natspec
        ), VFSState)
    else:
        return vfs_tools(VFSToolConfig(
            fs_layer=fs_layer,
            immutable=False,
            # Block writes to ANY spec file. Spec mutations must go through
            # propose_spec_change (for committed edits) or write_working_spec
            # + commit_working_spec (for iterative drafts).
            forbidden_write=r"^.+\.spec$",
            global_exclude=exclude_ts_and_natspec,
            put_doc_extra= \
    """
    By convention, every Solidity file placed into the virtual filesystem should contain exactly one contract/interface/library definitions.
    Further, the name of the contract/interface/library defined in that file should match the name of the solidity source file sans extension.
    For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`.

    IMPORTANT: You may not use this tool to update, create, or delete any spec file (any path ending in `.spec`).
    All spec mutations must go through the propose_spec_change tool (for committed edits) or the
    write_working_spec / commit_working_spec flow (for iterative drafts).
    """
        ), AIComposerState)

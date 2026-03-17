from typing import List
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from composer.input.types import Platform
from composer.core.validation import ValidationType, prover

@dataclass
class PlatformConfig:
    spec_fn: str
    system_prompt: str
    initial_prompt: str
    forbidden_write: str
    put_doc_extra: str
    required_validations: List[ValidationType]


EVM_PUT_DOC = """\
By convention, every Solidity file placed into the virtual filesystem should contain exactly one contract/interface/library definition.
Further, the name of the contract/interface/library defined in that file should match the name of the Solidity source file sans extension.
For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`.

IMPORTANT: You may not use this tool to update the specification, nor should you attempt to
add new specification files.
"""

SVM_PUT_DOC = """\
By convention, Rust source files should follow standard Rust module conventions. The main library entry point should be in src/lib.rs.
You MUST create a valid Cargo.toml for the project to compile and run tests. The Cargo.toml should include necessary dependencies
like `cvlr` for CVLR macros.

IMPORTANT: You may not use this tool to update the specification files. If changes to spec files are necessary, use the
propose_spec_change tool or consult the user.
"""

PLATFORM_CONFIGS: dict[Platform, PlatformConfig] = {
    "evm": PlatformConfig(
        spec_fn="rules.spec",
        system_prompt="system_prompt.j2",
        initial_prompt="synthesis_prompt.j2",
        forbidden_write=r"^rules\.spec$",
        put_doc_extra=EVM_PUT_DOC,
        required_validations=[prover],
    ),
    "svm": PlatformConfig(
        spec_fn="rules.rs",
        system_prompt="svm_system_prompt.j2",
        initial_prompt="svm_synthesis_prompt.j2",
        forbidden_write=r"^rules\.rs$",
        put_doc_extra=SVM_PUT_DOC,
        # solana prover can't be run to verify all rules but must be given individual rule names
        # however, the prover validation only happens if all rules pass in a single run,
        # hence we relax the validation to pass if individual rules pass
        required_validations=[],
    ),
}


class Config:
    def __init__(self, platform: Platform | None = None):
        if platform is not None:
            self.platform = platform
            self.__dict__.update(vars(PLATFORM_CONFIGS[platform]))

    def __call__(self, platform: Platform):
        self.__init__(platform)

    def __getattr__(self, name: str):
        raise AttributeError(f"Config not initialized: access to '{name}' requires calling config(platform) first")

config = Config()

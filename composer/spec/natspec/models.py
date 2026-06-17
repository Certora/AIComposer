"""
Pure data models shared by the natspec pipeline (interface + stub declarations).

These live in a dedicated module so that ``task_description.py`` (which defines
``Assembler`` over these types) and ``interface_gen.py`` / ``stub_gen.py``
(which produce instances via agents and need ``Assembler`` for validation) can
both import without a cycle.
"""

import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from composer.spec.system_model import ContractName, SolidityIdentifier


# ---------------------------------------------------------------------------
# Interface declaration models
# ---------------------------------------------------------------------------


class InterfaceDeclModel(BaseModel, ABC):
    """A single interface declaration."""
    content: str = Field(
        description="The contents of `path`, which should hold a complete Solidity "
        "interface describing the external entry points of the described contract(s)"
    )
    solidity_identifier: SolidityIdentifier = Field(description="The solidity identifier of the interface")
    if TYPE_CHECKING:
        @property
        @abstractmethod
        def path(self) -> str:
            ...


class LocatedInterfaceDecl(InterfaceDeclModel):
    __doc__ = InterfaceDeclModel.__doc__
    path: str = Field(description=(
        "The project-relative path where this interface file lives. Must end in "
        "'.sol'; the basename should match ``{solidity_identifier}.sol``. The task "
        "prompt governs how to choose this path."
    ))


class AutoInterfaceDecl(InterfaceDeclModel):
    __doc__ = InterfaceDeclModel.__doc__

    @property
    def path(self) -> str:
        return f"{self.solidity_identifier}.sol"


class InterfaceResult[T: InterfaceDeclModel](BaseModel):
    """The result of your interface generation."""
    name_to_interface: dict[SolidityIdentifier, T] = Field(
        description="A mapping from the explicit contract's solidity identifier to "
        "the interface describing the behavior of that component"
    )

    def dump_to_path(self, p: pathlib.Path) -> list[pathlib.Path]:
        """Write each interface to its agent-chosen ``path`` under root ``p``."""
        to_ret: list[pathlib.Path] = []
        for (_, i) in self.name_to_interface.items():
            rel_path = pathlib.Path(i.path)
            to_ret.append(rel_path)
            full_path = p / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(i.content)
        return to_ret


# ---------------------------------------------------------------------------
# Stub declaration models
# ---------------------------------------------------------------------------


class StubDeclarationModel(BaseModel, ABC):
    """The generated stub."""
    solidity_identifier: SolidityIdentifier = Field(
        description="The contract identifier (solidity identifier) chosen for the stub"
    )
    content: str = Field(
        description="The complete Solidity file which declares the stub implementation"
    )

    if TYPE_CHECKING:
        @property
        @abstractmethod
        def path(self) -> str:
            ...

class LocatedStubDeclaration(StubDeclarationModel):
    __doc__ = StubDeclarationModel.__doc__
    path: str = Field(description=(
        "The project-relative path where this stub file lives. Must end in '.sol'; "
        "the basename should match ``{solidity_identifier}.sol``. The task prompt "
        "governs how to choose this path."
    ))


class AutoStubDeclaration(StubDeclarationModel):
    __doc__ = StubDeclarationModel.__doc__

    @property
    def path(self) -> str:
        return f"{self.solidity_identifier}.sol"

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


# ---------------------------------------------------------------------------
# Interface declaration models
# ---------------------------------------------------------------------------


class InterfaceDeclModel(BaseModel, ABC):
    """A single interface declaration."""
    content: str = Field(
        description="The contents of `path`, which should hold a complete Solidity "
        "interface describing the external entry points of the described contract(s)"
    )
    solidity_identifier: str = Field(description="The solidity identifier of the interface")
    if TYPE_CHECKING:
        @property
        @abstractmethod
        def path(self) -> str:
            ...


class LocatedInterfaceDecl(InterfaceDeclModel):
    __doc__ = InterfaceDeclModel.__doc__
    path: str = Field(description=
        "The project-relative path where this interface should be placed (e.g. "
        "'contracts/interfaces/IFoo.sol' or 'src/interfaces/IFoo.sol'). Choose a path "
        "that fits the project: when an existing codebase is provided, use the source "
        "tools to inspect the layout and follow whatever convention the project already "
        "uses for interfaces (e.g. an existing 'interfaces/', 'src/interfaces/' or "
        "'contracts/interfaces/' directory). When generating from scratch with no "
        "source code, default to 'contracts/interfaces/<IdentifierName>.sol'. The path "
        "must end in '.sol' and the basename should match the solidity_identifier."
    )


class AutoInterfaceDecl(InterfaceDeclModel):
    __doc__ = InterfaceDeclModel.__doc__

    @property
    def path(self) -> str:
        return f"{self.solidity_identifier}.sol"


class InterfaceResult[T: InterfaceDeclModel](BaseModel):
    """The result of your interface generation."""
    name_to_interface: dict[str, T] = Field(
        description="A mapping from the explicit contract name to the interface "
        "describing the behavior of that component"
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
    solidity_identifier: str = Field(
        description="The contract name (solidity identifier) chosen for the stub"
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
    path: str = Field(
        description=(
            "The project-relative path where this stub should be placed (e.g. "
            "'src/contracts/Foo.sol' or 'contracts/impl/Foo.sol'). Inspect the "
            "existing source tree with the source tools and follow whatever "
            "convention the project already uses for contract implementations. "
            "The path must end in '.sol' and the basename should match "
            "``{solidity_identifier}.sol``."
        )
    )


class AutoStubDeclaration(StubDeclarationModel):
    __doc__ = StubDeclarationModel.__doc__

    @property
    def path(self) -> str:
        return f"src/contracts/{self.solidity_identifier}.sol"

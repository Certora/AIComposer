import pathlib
from typing import Any, TypedDict

from langchain_core.tools import BaseTool

from graphcore.tools.vfs import DirBackend, FSBackend, Materializer, fs_tools_layered

from composer.spec.gen_types import TypedTemplate
from composer.spec.system_model import Application, FromSourceApplication
from composer.spec.natspec.models import (
    InterfaceResult,
    LocatedInterfaceDecl, AutoInterfaceDecl,
    LocatedStubDeclaration, AutoStubDeclaration,
)
from composer.spec.natspec.task_description import (
    MentalModel, AgentDescription, InterfaceGenCallParams, StubGenCallParams,
)

class PathChoosingConf(TypedDict):
    agent_chooses_path: bool

_InterfaceTemplate = TypedTemplate[PathChoosingConf]("interface_generation_prompt.j2")
_StubTemplate = TypedTemplate[PathChoosingConf]("stub_generation_prompt.j2")


def build_mental_model(
    *,
    source_root: pathlib.Path | None,
    config_init: dict | None,
) -> MentalModel:
    # In from-source (update) mode the agent picks file locations to fit the
    # existing project layout. In greenfield there is no layout to conform to,
    # so paths are derived automatically from the solidity identifier.
    agent_chooses_path = source_root is not None
    interface_prompt = _InterfaceTemplate.bind(
        {"agent_chooses_path": agent_chooses_path}
    ).depends(InterfaceGenCallParams)
    stub_prompt = _StubTemplate.bind(
        {"agent_chooses_path": agent_chooses_path}
    ).depends(StubGenCallParams)

    if source_root is not None:
        return MentalModel(
            model_ty=FromSourceApplication,
            interface_desc=AgentDescription(
                output_ty=InterfaceResult[LocatedInterfaceDecl],
                prompt=interface_prompt,
            ),
            stub_desc=AgentDescription(
                output_ty=LocatedStubDeclaration,
                prompt=stub_prompt,
            ),
            source_root=source_root,
            config_init=config_init,
        )
    return MentalModel(
        model_ty=Application,
        interface_desc=AgentDescription(
            output_ty=InterfaceResult[AutoInterfaceDecl],
            prompt=interface_prompt,
        ),
        stub_desc=AgentDescription(
            output_ty=AutoStubDeclaration,
            prompt=stub_prompt,
        ),
        source_root=None,
        config_init=config_init,
    )


# ---------------------------------------------------------------------------
# Source-tool factory (phase-dispatched)
# ---------------------------------------------------------------------------


def make_source_factory(
    source_root: pathlib.Path | None,
    forbidden_read: str | None,
):
    """Build the ``source_factory`` closure the pipeline calls at each phase.

    The factory layers any extra backends the pipeline supplies (generated
    interfaces, the stub registry, etc.) over the on-disk source root, with
    the extras winning on collision. In greenfield mode there is no source
    root, so the factory just wires the extras.
    """
    base_layer: list[FSBackend] = []
    if source_root is not None:
        base_layer.append(DirBackend(source_root))

    def factory(extra_backends: list[FSBackend]) -> tuple[list[BaseTool], Materializer]:
        # Extras first — first-hit reads and reverse-order dumps mean the
        # extras' content wins over the source tree on collision.
        return fs_tools_layered(
            [*extra_backends, *base_layer],
            forbidden_read=forbidden_read,
        )

    return factory

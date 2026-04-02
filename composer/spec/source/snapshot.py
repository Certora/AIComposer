"""Snapshot and restore for batch_cvl_generation inputs."""

import logging
from typing import Any, Annotated, Literal

from pydantic import BaseModel, Discriminator

from langgraph.store.base import BaseStore

from composer.spec.context import (
    WorkflowContext, CVLGeneration, SourceCode, SNAPSHOT_NAMESPACE,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import CVLResource
from composer.spec.system_model import (
    AnyApplication, Application, SourceApplication, HarnessedApplication,
    ContractComponentInstance,
)
from composer.spec.mnemonic import mnemonic_id

# ---------------------------------------------------------------------------
# Application discriminated union for serialization
# ---------------------------------------------------------------------------

class _BaseApp(BaseModel):
    kind: Literal["base"] = "base"
    app: Application

class _SourceApp(BaseModel):
    kind: Literal["source"] = "source"
    app: SourceApplication

class _HarnessedApp(BaseModel):
    kind: Literal["harnessed"] = "harnessed"
    app: HarnessedApplication

type AppSnapshot = Annotated[
    _BaseApp | _SourceApp | _HarnessedApp,
    Discriminator("kind"),
]


def _snapshot_app(app: AnyApplication) -> AppSnapshot:
    match app:
        case HarnessedApplication() as h:
            return _HarnessedApp(app=h)
        case SourceApplication() as s:
            return _SourceApp(app=s)
        case Application() as a:
            return _BaseApp(app=a)


# ---------------------------------------------------------------------------
# Snapshot models
# ---------------------------------------------------------------------------

class SourceSnapshot(BaseModel):
    """Serializable mirror of SourceCode."""
    content: str | dict[str, Any]
    project_root: str
    contract_name: str
    relative_path: str
    forbidden_read: str

    @staticmethod
    def from_source(source: SourceCode) -> "SourceSnapshot":
        return SourceSnapshot(
            content=source.content,
            project_root=source.project_root,
            contract_name=source.contract_name,
            relative_path=source.relative_path,
            forbidden_read=source.forbidden_read,
        )

    def restore(self) -> SourceCode:
        return SourceCode(**self.model_dump())


class ComponentSnapshot(BaseModel):
    """Serializable mirror of ContractComponentInstance."""
    component_index: int
    contract_index: int
    app: AppSnapshot

    @staticmethod
    def from_component(c: ContractComponentInstance) -> "ComponentSnapshot":
        return ComponentSnapshot(
            component_index=c.ind,
            contract_index=c.contract_index,
            app=_snapshot_app(c.app),
        )

    def restore(self) -> ContractComponentInstance:
        return ContractComponentInstance.from_app(
            app=self.app.app,
            contract_index=self.contract_index,
            component_index=self.component_index,
        )


class CVLGenSnapshot(BaseModel):
    """Complete snapshot of inputs to batch_cvl_generation."""
    mnemonic: str
    props: list[PropertyFormulation]
    resources: list[CVLResource]
    source: SourceSnapshot
    init_config: dict[str, Any]
    component: ComponentSnapshot | None
    description: str
    # WorkflowContext coordinates
    thread_id: str
    memory_namespace: str
    cache_namespace: tuple[str, ...] | None


# ---------------------------------------------------------------------------
# Take / load
# ---------------------------------------------------------------------------

async def take_snapshot(
    ctx: WorkflowContext[CVLGeneration],
    *,
    props: list[PropertyFormulation],
    resources: list[CVLResource],
    source: SourceCode,
    init_config: dict,
    component: ContractComponentInstance | None,
    description: str,
) -> str:
    """Capture and store a snapshot of batch_cvl_generation inputs.

    Returns the mnemonic key under which the snapshot is stored.
    """
    mnem = mnemonic_id()

    snapshot = CVLGenSnapshot(
        mnemonic=mnem,
        props=props,
        resources=resources,
        source=SourceSnapshot.from_source(source),
        init_config=init_config,
        component=ComponentSnapshot.from_component(component) if component else None,
        description=description,
        thread_id=ctx.thread_id,
        memory_namespace=ctx.memory_namespace,
        cache_namespace=ctx.cache_namespace,
    )

    await ctx.save_snapshot(mnem, snapshot)
    return mnem


async def load_snapshot(store: BaseStore, mnemonic: str) -> CVLGenSnapshot:
    """Load a snapshot from the store by mnemonic.

    Raises KeyError if no snapshot exists for the given mnemonic.
    """
    result = await store.aget(SNAPSHOT_NAMESPACE, mnemonic)
    if result is None:
        raise KeyError(f"No snapshot found for mnemonic: {mnemonic}")
    return CVLGenSnapshot.model_validate(result.value)

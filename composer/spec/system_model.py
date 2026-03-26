from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field
from functools import cached_property

type ContractSort = Literal["dynamic", "singleton", "multiple"]

class ExternalDependency(BaseModel):
    external_actor: str = Field(description="The name of the external actor interacted with")
    description : str = Field(description="A description of the interaction with the external actor")

class ComponentInteraction(BaseModel):
    contract_name: str = Field(description="The name of the contract interacted with")
    component : str | None = Field(description="The specific component within that contract interacted with")
    description : str = Field(description="A description of the interaction with the contract component.")

type Interaction = ComponentInteraction | ExternalDependency

class ContractComponent(BaseModel):
    """
    A single major "component" within a contract.
    """
    name: str = Field(description="A short, concise name of the component")
    description: str = Field(description="A longer description describing *what* this component does, not *how* it does it.")
    external_entry_points : list[str] = Field(description="The signatures/names of any external entry points explicitly part of this component")
    state_variables : list[str] = Field(description="The name & types of any storage/state variables explicitly linked to this entry point")
    interactions: list[Interaction] = Field(description="A list of interactions with other components; either external actors or other components described in this system")
    requirements : list[str] = Field(description="A list of natural language requirements for this component, i.e., it's behavioral specification")

    @cached_property
    def intra_application_interactions(self) -> list[ComponentInteraction]:
        to_ret : list[ComponentInteraction] = []
        for int in self.interactions:
            if isinstance(int, ComponentInteraction) and int.contract_name != self.name:
                to_ret.append(int)
        return to_ret

class ExplicitContract(BaseModel):
    """
    A concrete contract type in the system.
    """
    sort: ContractSort = Field(description=("The sort of the contract. `dynamic` if instances of this type are "
        "dynamically created by the system itself. `multiple` if multiple instances are expected to be "
        "deployed by some external actor/administrator. `singleton` if only one instance will exist in a deployed system."))
    name: str = Field(description="A short, unique identifier for this contract")
    description : str = Field(description="A short description of what this contract's role is in the system")
    components : list[ContractComponent] = Field(description="Components making up this contract.")

class SourceExplicitContract(ExplicitContract):
    """
    A concrete contract type in the system.
    """
    path: str = Field("The relative path to the file which defines the contract type this represents")

class HarnessDefinition(BaseModel):
    path: str
    name: str

class HarnessedExplicitContract(SourceExplicitContract):
    harnesses: list[HarnessDefinition]

class ExternalActor(BaseModel):
    """
    Some "external actor" to the system. This may be an administrator, an EOA,
    or some off-chain component, or a contract deployed and managed by someone else.
    """
    name: str = Field(description="A short, unique identifier for this external actor")
    description : str = Field(description="A short, technical description of this external actor")
    assumptions: list[str] = Field(description="A list of assumptions or requirements about this external actor's behavior")

class SourceExternalActor(ExternalActor):
    """
    Some "external actor" to the system. This may be an administrator, an EOA,
    or some off-chain component, or a contract deployed and managed by someone else.
    """
    path : str | None = Field(description="The relative path to the interface describing this external actor, if relevant.")

type SystemComponent = ExternalActor | ExplicitContract

class BaseApplication[T : SystemComponent](BaseModel):
    application_type: str = Field(description="A concise, description of the type of application (AMM/Liquidity Provider/etc.)")
    description: str = Field(description="A description of the application's main functionality (2 - 3 sentences max)")
    components : list[T] = Field(description="The system components (explicit contract & external actors) that comprise this application")

class Application(BaseApplication[SystemComponent]):
    """
    A description of the application.
    """

    @cached_property
    def contract_components(self) -> list[ExplicitContract]:
        to_ret = []
        for c in self.components:
            if not isinstance(c, ExplicitContract):
                continue
            to_ret.append(c)
        return to_ret
    
class SourceApplication(BaseApplication[SourceExplicitContract | SourceExternalActor]):
    """
    A description of the application.
    """
    @cached_property
    def contract_components(self) -> list[SourceExplicitContract]:
        to_ret = []
        for c in self.components:
            if not isinstance(c, SourceExplicitContract):
                continue
            to_ret.append(c)
        return to_ret

class HarnessedApplication(BaseApplication[HarnessedExplicitContract | SourceExternalActor]):
    @cached_property
    def contract_components(self) -> list[HarnessedExplicitContract]:
        to_ret = []
        for c in self.components:
            if not isinstance(c, HarnessedExplicitContract):
                continue
            to_ret.append(c)
        return to_ret


type AnyApplication = Application | SourceApplication | HarnessedApplication

@dataclass
class ContractInstance:
    ind: int
    app: AnyApplication

    @property
    def contract(self) -> ExplicitContract:
        return self.app.contract_components[self.ind]
    
    @cached_property
    def sibling_contracts(self) -> list[ExplicitContract]:
        to_ret : list[ExplicitContract] = []
        for (ind, c) in enumerate(self.app.contract_components):
            if ind == self.ind:
                continue
            to_ret.append(c)
        return to_ret

@dataclass
class ContractComponentInstance:
    ind: int
    _contract: ContractInstance

    @property
    def app(self) -> AnyApplication:
        return self._contract.app
    
    @property
    def contract(self) -> ExplicitContract:
        return self._contract.contract
    
    @property
    def ommer_contract(self) -> list[ExplicitContract]:
        return self._contract.sibling_contracts

    @property
    def component(self) -> ContractComponent:
        return self.contract.components[self.ind]

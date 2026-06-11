from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from functools import cached_property
from composer.spec.util import slugify_filename

type ContractSort = Literal["dynamic", "singleton", "multiple"]

# Nominal ``str`` subtypes for the two distinct contract-identity fields.
# Both phantom-typed (TYPE_CHECKING-only subclass; ``str`` at runtime) so they
# remain distinct at static-check time but pydantic ``Field`` validates them
# as plain strings.
#
# ``SolidityIdentifier``: a Solidity contract identifier (regex-validated where
# stored on a pydantic field).
# ``ContractName``: the conceptual / design-doc-readable name of a contract.
# May be a Solidity identifier when the design doc names contracts that way,
# but allowed to be anything human-readable.
#
# The two are **siblings under str**, not in a subtype relation with each
# other — passing a ``SolidityIdentifier`` where ``ContractName`` is expected
# (or vice-versa) is a type error, even though both are ``str`` at runtime.
if TYPE_CHECKING:
    class SolidityIdentifier(str): ...
    class ContractName(str): ...
else:
    SolidityIdentifier = str
    ContractName = str

class ExternalDependency(BaseModel):
    external_actor: str = Field(description="The name of the external actor interacted with")
    description : str = Field(description="A description of the interaction with the external actor")

class ComponentInteraction(BaseModel):
    contract_name: ContractName = Field(description="The conceptual name of the contract interacted with (matching the `name` field of an ExplicitContract in the application)")
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
    name: ContractName = Field(description=(
        "A short, conceptual label for this contract, used to refer to it across the "
        "system description. May be the same as solidity_identifier when the design "
        "doc names the contract by its Solidity identifier directly."
    ))
    solidity_identifier: SolidityIdentifier = Field(
        pattern=r"^[a-zA-Z_$][a-zA-Z0-9_$]*$",
        description=(
            "The Solidity identifier this contract will be deployed/compiled under. "
            "Derived from authoritative sources where possible (see system analysis "
            "prompt). Always a syntactically valid Solidity identifier."
        ),
    )
    description : str = Field(description="A short description of what this contract's role is in the system")
    components : list[ContractComponent] = Field(description="Components making up this contract.")

class SourceExplicitContract(ExplicitContract):
    """
    A concrete contract type in the system.
    """
    path: str = Field("The relative path to the file which defines the contract type this represents")

class HarnessDefinition(BaseModel):
    path: str
    name: SolidityIdentifier

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


class FromSourceContract(ExplicitContract):
    """Base for contracts in the ``from-current-source`` (natspec
    ``update``/``existing``) workflow.

    Split into two concrete variants by relationship to the change being
    built: :class:`ExistingFromSource` (contract is already in the source
    tree, either untouched or being edited) and :class:`FreshFromSource`
    (new contract being introduced by this task — no source path yet).
    """


class ExistingFromSource(FromSourceContract):
    """An already-present contract in the source tree."""
    path: str = Field(description="The relative path to the file defining this contract.")
    tag: Literal["unchanged", "edited"] = Field(description=(
        "Relationship of this contract to the change being built: "
        "'unchanged' if it's an existing dependency left as-is, "
        "'edited' if an existing contract is being modified for this task."
    ))


class FreshFromSource(FromSourceContract):
    """A new contract being introduced by this task — no source path yet."""
    tag: Literal["new"] = Field(description=(
        "This contract is being introduced by this task; there is no existing "
        "source file for it yet."
    ))


class FromSourceApplication(BaseApplication[ExistingFromSource | FreshFromSource | SourceExternalActor]):
    """Application variant for the ``from-current-source`` workflow — each
    explicit contract is tagged ``edited`` / ``unchanged`` / ``new`` via the
    :class:`FromSourceContract` subtype split.
    """
    @cached_property
    def contract_components(self) -> list[FromSourceContract]:
        to_ret: list[FromSourceContract] = []
        for c in self.components:
            if not isinstance(c, FromSourceContract):
                continue
            to_ret.append(c)
        return to_ret


type NatspecApplication = FromSourceApplication | Application
type AnyApplication = Application | SourceApplication | HarnessedApplication | FromSourceApplication

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
    def contract_index(self) -> int:
        """Index of the parent contract within the application."""
        return self._contract.ind

    @property
    def component(self) -> ContractComponent:
        return self.contract.components[self.ind]

    @property
    def slugified_name(self) -> str:
        """Filesystem-safe, collision-free slug for this component, used as an
        output filename base. Component names are unique within a contract, but
        slugifying can map distinct names onto the same slug, so the component
        index disambiguates when (and only when) the slug collides with a
        sibling's."""
        slug = slugify_filename(self.component.name)
        sibling_slugs = [slugify_filename(c.name) for c in self.contract.components]
        if sibling_slugs.count(slug) > 1:
            return f"{slug}_{self.ind}"
        return slug

    @staticmethod
    def from_app(
        app: AnyApplication,
        contract_index: int,
        component_index: int,
    ) -> "ContractComponentInstance":
        """Reconstruct from an application model and indices."""
        return ContractComponentInstance(
            ind=component_index,
            _contract=ContractInstance(ind=contract_index, app=app),
        )

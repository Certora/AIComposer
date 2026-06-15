from typing import Literal

from pydantic import BaseModel, Field

# The identifier of a property. The "foreign key" that ties a property to the
# rules it produced (``PropertyRuleMapping``) and to coverage tracking
# (``UncoveredProperty``). In the inference pipeline it is the LLM-generated
# snake_case ``PropertyFormulation.title``; in the known-properties pipeline it
# is the opaque input id (e.g. ``"001"``). Same string either way.
type PropertyId = str


class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    title: PropertyId = Field(description="A short, descriptive snake_case identifier for the property (e.g. 'total_supply_preserved'). Must be unique within the batch of properties.")
    methods: list[str] | Literal["invariant"] = Field(description="A list of external methods involved in the property, or 'invariant' if the property is an invariant on the contract state")
    sort: Literal["attack_vector", "safety_property", "invariant"] = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")


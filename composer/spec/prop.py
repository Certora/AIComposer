from typing import Literal

from pydantic import BaseModel, Field

type PropertyType = Literal["attack_vector", "safety_property", "invariant"]
"""The kind of a property: an attack vector, a safety property, or a state
invariant. Shared so every layer (inference, report, grouping) addresses the
same vocabulary instead of redeclaring the literal."""


class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    title: str = Field(description="A short, descriptive snake_case identifier for the property (e.g. 'total_supply_preserved'). Must be unique within the batch of properties.")
    methods: list[str] | Literal["invariant"] = Field(description="A list of external methods involved in the property, or 'invariant' if the property is an invariant on the contract state")
    sort: PropertyType = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")

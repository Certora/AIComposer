from typing import Literal

from pydantic import BaseModel, Field

class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    methods: list[str] | Literal["invariant"] = Field(description="A list of external methods involved in the property, or 'invariant' if the property is an invariant on the contract state")
    sort: Literal["attack_vector", "safety_property", "invariant"] = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")


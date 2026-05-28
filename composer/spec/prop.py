from typing import Literal

from pydantic import BaseModel, Field

class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    sort: Literal["attack_vector", "safety_property", "invariant"] = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")


from typing import Literal

from pydantic import BaseModel, Field

class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    methods: list[str] | Literal["invariant"] = Field(description="A list of external methods involved in the property, or 'invariant' if the property is an invariant on the contract state")
    sort: Literal["attack_vector", "safety_property", "invariant"] = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")

    def to_template_args(self) -> dict:
        thing : str
        what_formal : str
        prop = self
        match prop.sort:
            case "attack_vector":
                thing = "potential attack vector/exploit"
                what_formal = f"that a {thing} is not possible"
            case "invariant":
                thing = "invariant"
                what_formal = "that an invariant holds"
            case "safety_property":
                thing = "safety property"
                what_formal = "that a safety property holds"
        return {
            "thing": thing,
            "what_formal": what_formal,
            "thing_tag": self.sort,
            "thing_descr": self.description
        }

from typing import Literal, TypedDict, Annotated

from pydantic import Discriminator

class ProverSuccess(TypedDict):
    sort: Literal["success"]
    is_local_link: bool
    link: str | None

class ProverFailure(TypedDict):
    sort: Literal["failure"]
    exc_str: str

type ProverResultData = Annotated[ProverSuccess | ProverFailure, Discriminator("sort")]

type ProverResult = ProverResultData | None

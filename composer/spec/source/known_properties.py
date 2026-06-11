"""Input model + loader for the *known-properties* autoprove pipeline.

The properties pipeline takes a YAML file listing properties/invariants the user
already knows they want proven (instead of inferring them from the source). This
module defines the input schema and a loader that validates the YAML *up front*
— before any LLM work — surfacing clear errors for malformed input.

Input example::

    - sort: invariant
      property_desc: The protocol never becomes insolvent.
      property_id: "001"
    - sort: safety_property
      property_desc: A user can always withdraw their full balance.
      property_id: "002"

``sort`` accepts ``invariant``, ``safety_property``, ``attack_vector``, where
``safety_property`` is an alias for ``attack_vector`` (normalized on load). After
normalization ``sort`` is in ``{invariant, attack_vector}``, a subset of
``PropertyFormulation.sort`` (see ``composer/spec/prop.py``), so the downstream
conversion in the formalize phase is total.
"""

import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Input accepts ``safety_property`` as an alias for ``attack_vector``; normalized
# on load. Canonical sort matches ``PropertyFormulation.sort`` (prop.py).
InputSort = Literal["invariant", "safety_property", "attack_vector"]


class KnownProperty(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typo'd keys are errors

    sort: InputSort
    property_desc: str = Field(min_length=1)
    property_id: str = Field(min_length=1)

    @field_validator("sort")
    @classmethod
    def _normalize_sort(cls, v: str) -> str:
        # safety_property -> attack_vector
        return "attack_vector" if v == "safety_property" else v


class KnownProperties(BaseModel):
    properties: list[KnownProperty]


class KnownPropertiesError(Exception):
    """Raised when a ``--properties`` YAML file is missing, unreadable, or
    malformed. Carries a human-readable message suitable for ``parser.error``."""


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic ``ValidationError`` as a compact, user-facing message
    keyed by the offending list index / field."""
    lines: list[str] = []
    for err in exc.errors():
        # loc looks like (0, "sort") for properties[0].sort
        loc = err.get("loc", ())
        if loc and isinstance(loc[0], int):
            where = f"property #{loc[0]}"
            rest = ".".join(str(p) for p in loc[1:])
            if rest:
                where += f" (field '{rest}')"
        else:
            where = ".".join(str(p) for p in loc) or "<root>"
        lines.append(f"  - {where}: {err.get('msg', 'invalid')}")
    return "Invalid properties file:\n" + "\n".join(lines)


def load_known_properties(path: pathlib.Path) -> KnownProperties:
    """Load and validate a ``--properties`` YAML file.

    Surfaces clear ``KnownPropertiesError``s for: missing/unreadable file,
    invalid YAML, top-level not-a-list, missing/extra fields or bad ``sort``
    (reformatted ``ValidationError``), empty strings, and duplicate
    ``property_id``. Reads the file directly with ``read_text`` (not the
    document uploader). Raised before any LLM work.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise KnownPropertiesError(f"cannot read properties file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise KnownPropertiesError(f"malformed YAML in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise KnownPropertiesError(
            f"properties file {path} must contain a YAML list of properties, "
            f"got {type(data).__name__}"
        )

    try:
        parsed = [KnownProperty.model_validate(item) for item in data]
    except ValidationError as exc:
        raise KnownPropertiesError(_format_validation_error(exc)) from exc

    seen: set[str] = set()
    for prop in parsed:
        if prop.property_id in seen:
            raise KnownPropertiesError(
                f"duplicate property_id {prop.property_id!r} in {path}"
            )
        seen.add(prop.property_id)

    return KnownProperties(properties=parsed)

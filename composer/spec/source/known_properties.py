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

``sort`` accepts ``invariant``, ``safety_property``, ``attack_vector`` — exactly
``PropertyFormulation.sort`` (see ``composer/spec/prop.py``) — and is passed
through unchanged, so the downstream conversion in the formalize phase is total
and the distinction surfaced to the agents is preserved.
"""

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from composer.spec.prop import PropertyId, PropertySort, PropertyFormulation


class KnownProperty(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typo'd keys are errors

    sort: PropertySort
    property_desc: str = Field(min_length=1)
    property_id: PropertyId = Field(min_length=1)

    def to_formulation(self, methods: list[str]) -> PropertyFormulation:
        """Build the CVL-generation ``PropertyFormulation`` for this known property,
        given the external entry points the formalize agent mapped it to. ``sort``
        and ``description`` come from this (authoritative) YAML row; an invariant
        carries no methods."""
        return PropertyFormulation(
            title=self.property_id,
            sort=self.sort,
            description=self.property_desc,
            methods="invariant" if self.sort == "invariant" else methods,
        )


class KnownProperties(BaseModel):
    properties: list[KnownProperty]


_PROPERTIES_ADAPTER = TypeAdapter(list[KnownProperty])


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

    try:
        parsed = _PROPERTIES_ADAPTER.validate_python(data)
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

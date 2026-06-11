"""Tests for the known-properties YAML loader (``load_known_properties``)."""

import pathlib

import pytest

from composer.spec.source.known_properties import (
    KnownProperties, KnownPropertiesError, load_known_properties,
)


def _write(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    p = tmp_path / "props.yml"
    p.write_text(text)
    return p


def test_valid_file(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
- sort: invariant
  property_desc: The protocol never becomes insolvent.
  property_id: "001"
- sort: attack_vector
  property_desc: No user can withdraw more than they deposited.
  property_id: "002"
""")
    result = load_known_properties(p)
    assert isinstance(result, KnownProperties)
    assert [x.property_id for x in result.properties] == ["001", "002"]
    assert [x.sort for x in result.properties] == ["invariant", "attack_vector"]


def test_safety_property_normalizes_to_attack_vector(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
- sort: safety_property
  property_desc: A user can always withdraw their full balance.
  property_id: "001"
""")
    result = load_known_properties(p)
    assert result.properties[0].sort == "attack_vector"


def test_not_a_list(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
sort: invariant
property_desc: foo
property_id: "001"
""")
    with pytest.raises(KnownPropertiesError, match="must contain a YAML list"):
        load_known_properties(p)


def test_missing_field(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
- sort: invariant
  property_id: "001"
""")
    with pytest.raises(KnownPropertiesError, match="property_desc"):
        load_known_properties(p)


def test_extra_field_rejected(tmp_path: pathlib.Path) -> None:
    # A ``property:`` key (instead of ``property_desc``) is rejected by extra="forbid".
    p = _write(tmp_path, """
- sort: invariant
  property: The protocol never becomes insolvent.
  property_id: "001"
""")
    with pytest.raises(KnownPropertiesError):
        load_known_properties(p)


def test_bad_sort(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
- sort: liveness
  property_desc: foo
  property_id: "001"
""")
    with pytest.raises(KnownPropertiesError, match="sort"):
        load_known_properties(p)


def test_empty_string(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
- sort: invariant
  property_desc: ""
  property_id: "001"
""")
    with pytest.raises(KnownPropertiesError, match="property_desc"):
        load_known_properties(p)


def test_duplicate_property_id(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, """
- sort: invariant
  property_desc: foo
  property_id: "001"
- sort: attack_vector
  property_desc: bar
  property_id: "001"
""")
    with pytest.raises(KnownPropertiesError, match="duplicate property_id"):
        load_known_properties(p)


def test_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(KnownPropertiesError, match="cannot read"):
        load_known_properties(tmp_path / "does_not_exist.yml")

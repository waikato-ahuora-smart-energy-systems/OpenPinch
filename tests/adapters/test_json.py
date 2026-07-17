"""Behavioural contracts for the UTF-8 JSON filesystem adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch.adapters.io.json import parse_json, read_json, write_json

JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=40),
)
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=8),
        st.dictionaries(st.text(max_size=24), children, max_size=8),
    ),
    max_leaves=20,
)


@seed(20260715)
@given(value=JSON_VALUES)
@settings(max_examples=40)
def test_json_text_round_trip_preserves_json_values(value) -> None:
    """Every supported JSON value survives the adapter's text boundary."""
    import json

    assert parse_json(json.dumps(value)) == value


def test_json_file_round_trip_uses_utf8_and_returns_destination(tmp_path: Path) -> None:
    destination = tmp_path / "nested-δ.json"
    value = {"site": "Whangārei", "loads": [1.0, 2.5], "active": True}

    written = write_json(destination, value, indent=2)

    assert written == destination
    assert read_json(destination) == value
    assert "Whangārei" in destination.read_text(encoding="utf-8")


def test_json_adapter_propagates_invalid_document_errors(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(ValueError):
        read_json(source)

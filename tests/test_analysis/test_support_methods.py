"""Regression tests for support methods analysis routines."""

import pytest

from OpenPinch.lib import ValueWithUnit
from OpenPinch.services.common.miscellaneous import get_period_index
from OpenPinch.utils.value_resolution import (
    get_period_value,
    get_scalar_value,
)

"""Test cases for value resolution and period selection helpers."""


def test_resolve_scalar_value_with_float():
    assert get_scalar_value(3.14) == 3.14


def test_resolve_scalar_value_with_dict():
    assert get_scalar_value({"value": 42}) == 42


def test_resolve_scalar_value_with_valuewithunit():
    vwu = ValueWithUnit(value=99.9, unit="kW")
    assert get_scalar_value(vwu) == 99.9


def test_resolve_period_value_with_period_payload_defaults_to_period_zero():
    payload = {
        "values": [99.9, 88.8],
        "period_ids": ["0", "1"],
        "unit": "kW",
        "weights": [0.25, 0.75],
    }
    assert get_period_value(payload) == 99.9


def test_resolve_period_value_with_period_payload_uses_requested_period_idx():
    payload = {
        "values": [99.9, 88.8],
        "period_ids": ["0", "1"],
        "unit": "kW",
        "weights": [0.25, 0.75],
    }
    assert get_period_value(payload, period_idx=1) == 88.8


def test_get_period_index_resolves_period_id():
    idx, sid = get_period_index({"0": 0, "peak": 1}, {"period_id": "peak"})

    assert idx == 1
    assert sid == "peak"


def test_get_period_index_rejects_unknown_period_id():
    with pytest.raises(ValueError, match="period_id 'summer' was not found"):
        get_period_index({"0": 0, "peak": 1}, {"period_id": "summer"})


def test_get_period_index_accepts_explicit_idx():
    idx, sid = get_period_index({"0": 0, "peak": 1}, {"period_idx": 1})

    assert idx == 1
    assert sid is None


def test_get_period_index_rejects_conflicting_period_id_and_idx():
    with pytest.raises(ValueError, match="period_id 'peak' resolves to period_idx 1"):
        get_period_index({"0": 0, "peak": 1}, {"period_id": "peak", "period_idx": 0})


def test_resolve_scalar_value_with_int():
    assert get_scalar_value(5) == 5.0


def test_resolve_scalar_value_with_string():
    assert get_scalar_value("100") == 100.0

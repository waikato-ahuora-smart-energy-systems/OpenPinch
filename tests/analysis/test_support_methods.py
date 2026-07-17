"""Regression tests for support methods analysis routines."""

import pytest

import OpenPinch.analysis.numerics as miscellaneous
from OpenPinch.analysis.numerics import g_ineq_penalty, get_period_index
from OpenPinch.analysis.targeting.context import (
    apply_zone_config_overrides,
    format_selected_period_suffix,
)
from OpenPinch.contracts.common import ValueWithUnit
from OpenPinch.domain._value.resolution import (
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


def test_get_period_index_rejects_negative_and_unknown_explicit_idx():
    with pytest.raises(ValueError, match="non-negative"):
        get_period_index({"0": 0}, {"period_idx": -1})

    with pytest.raises(ValueError, match="period_idx 2 was not found"):
        get_period_index({"0": 0, "peak": 1}, {"period_idx": 2})


def test_get_period_index_rejects_conflicting_period_id_and_idx():
    with pytest.raises(ValueError, match="period_id 'peak' resolves to period_idx 1"):
        get_period_index({"0": 0, "peak": 1}, {"period_id": "peak", "period_idx": 0})


def test_resolve_scalar_value_with_int():
    assert get_scalar_value(5) == 5.0


def test_resolve_scalar_value_with_string():
    assert get_scalar_value("100") == 100.0


def test_service_orchestration_rejects_non_runtime_overrides_and_formats_suffixes():
    with pytest.raises(ValueError, match="Invalid key"):
        apply_zone_config_overrides(object(), {"THERMAL_DT_CONT": 20})

    assert format_selected_period_suffix(None) == ""
    assert format_selected_period_suffix({}) == ""
    assert format_selected_period_suffix({"period_idx": 2}) == " for period_idx 2"


def test_g_ineq_penalty_rejects_unrecognised_internal_penalty_type(monkeypatch):
    class FakeArray:
        def __pow__(self, exponent):
            return self

        def __rmul__(self, other):
            return self

    monkeypatch.setattr(
        miscellaneous.np, "asarray", lambda value, dtype=None: FakeArray()
    )

    with pytest.raises(ValueError, match="Return of the penalty function failed"):
        g_ineq_penalty(1.0)

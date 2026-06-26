"""Regression tests for explicit value-resolution helpers."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from OpenPinch.classes import Value
from OpenPinch.lib.schemas.common import (
    PeriodValueWithUnit,
    PeriodValueWithUnitAndWeights,
    ValueWithUnit,
)
from OpenPinch.utils.value_resolution import (
    evaluate_value_spec,
    get_period_value,
    get_scalar_value,
    resolve_value_array,
)


def test_resolve_period_value_scalar_passthrough():
    assert get_period_value(3.14) == pytest.approx(3.14)


def test_resolve_period_value_single_period_value_like():
    payload = PeriodValueWithUnit(values=[7.5], unit="kW")

    assert get_period_value(payload) == pytest.approx(7.5)


def test_resolve_period_value_defaults_to_period_zero():
    payload = {"values": [99.9, 88.8], "period_ids": ["0", "peak"], "unit": "kW"}

    assert get_period_value(payload) == pytest.approx(99.9)


def test_resolve_period_value_uses_explicit_idx():
    payload = {"values": [99.9, 88.8], "period_ids": ["0", "peak"], "unit": "kW"}

    assert get_period_value(payload, period_idx=1) == pytest.approx(88.8)


def test_resolve_period_value_rejects_missing_idx_when_default_disallowed():
    payload = {"values": [99.9, 88.8], "period_ids": ["0", "peak"], "unit": "kW"}

    with pytest.raises(ValueError, match="idx is required"):
        get_period_value(payload, default_allowed=False)


def test_resolve_period_value_rejects_negative_idx():
    payload = {"values": [99.9, 88.8], "period_ids": ["0", "peak"], "unit": "kW"}

    with pytest.raises(ValueError, match="non-negative"):
        get_period_value(payload, period_idx=-1)


def test_resolve_period_value_rejects_out_of_range_idx():
    payload = {"values": [99.9, 88.8], "period_ids": ["0", "peak"], "unit": "kW"}

    with pytest.raises(ValueError, match="out of range"):
        get_period_value(payload, period_idx=2)


@pytest.mark.parametrize(
    ("payload", "kwargs", "expected"),
    [
        pytest.param(3.14, {}, 3.14, id="float"),
        pytest.param(5, {}, 5.0, id="int"),
        pytest.param("100", {}, 100.0, id="numeric-string"),
        pytest.param(Value(99.9, unit="kW"), {}, 99.9, id="value"),
        pytest.param(
            ValueWithUnit(value=99.9, unit="kW"),
            {},
            99.9,
            id="value-with-unit",
        ),
        pytest.param(
            {"value": {"value": "42.0"}, "unit": "kW"},
            {},
            42.0,
            id="nested-scalar-payload",
        ),
        pytest.param(
            {"values": [99.9, 88.8], "period_ids": ["0", "peak"], "unit": "kW"},
            {"period_idx": 1},
            88.8,
            id="period_valued-payload",
        ),
        pytest.param(None, {}, None, id="none"),
    ],
)
def test_resolve_scalar_value_supported_inputs(payload, kwargs, expected):
    result = get_scalar_value(payload, **kwargs)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_resolve_scalar_value_rejects_bool():
    with pytest.raises(TypeError, match="Boolean values are not supported"):
        get_scalar_value(True)


@pytest.mark.parametrize(
    "payload",
    [
        {"value": 4.0, "add": 2.0},
        {"zone-a": 11.25},
    ],
)
def test_resolve_scalar_value_rejects_specs(payload):
    with pytest.raises(TypeError, match="evaluate_value_spec"):
        get_scalar_value(payload)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param({"value": 4.0, "multiplier": 2.5}, 10.0, id="multiplier"),
        pytest.param({"value": 4.0, "multiply": 2.5}, 10.0, id="multiply"),
        pytest.param({"value": 4.0, "add": 2.5}, 6.5, id="add"),
        pytest.param({"value": 4.0, "subtract": 2.5}, 1.5, id="subtract"),
        pytest.param({"value": 9.0, "divide": 3.0}, 3.0, id="divide"),
        pytest.param({"value": 0.0, "divide": 3.0}, 0.0, id="divide-zero"),
        pytest.param({"value": 3.0, "power": 2.0}, 9.0, id="power"),
        pytest.param({"value": 100.0, "log": 10.0}, 2.0, id="log"),
        pytest.param({"value": -10.0, "log": 10.0}, 0.0, id="log-negative"),
        pytest.param({"value": 3.0, "exp": 2.0}, 8.0, id="exp"),
        pytest.param({"value": -1.0, "exp": 2.0}, 0.0, id="exp-negative"),
        pytest.param({"value": -4.0, "abs": True}, 4.0, id="abs"),
        pytest.param({"value": 3.0, "min": 2.0}, 2.0, id="min"),
        pytest.param({"value": 3.0, "max": 7.0}, 7.0, id="max"),
    ],
)
def test_evaluate_value_spec_supports_operators(spec, expected):
    assert evaluate_value_spec(spec) == pytest.approx(expected)


def test_evaluate_value_spec_handles_zone_name_and_default_value():
    spec = {
        "zone-a": {"multiply": 0.5},
        "zone-b": {"add": 3.0},
    }

    assert evaluate_value_spec(spec, zone_name="zone-a", default_value=20.0) == 10.0
    assert evaluate_value_spec(spec, zone_name="zone-b", default_value=20.0) == 23.0


def test_evaluate_value_spec_resolves_nested_leaves_with_idx():
    spec = {
        "zone-a": {
            "value": {"values": [5.0, 8.0], "period_ids": ["0", "peak"], "unit": "kW"},
            "add": {"value": "2.0"},
        }
    }

    assert evaluate_value_spec(spec, zone_name="zone-a", period_idx=1) == pytest.approx(
        10.0
    )


def test_evaluate_value_spec_does_not_mutate_input():
    spec = {
        "zone-a": {
            "value": {"values": [5.0, 8.0], "period_ids": ["0", "peak"], "unit": "kW"},
            "add": 2.0,
        }
    }
    before = deepcopy(spec)

    assert evaluate_value_spec(spec, zone_name="zone-a", period_idx=1) == pytest.approx(
        10.0
    )
    assert spec == before


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(3.0, np.array([3.0]), id="scalar"),
        pytest.param(
            ValueWithUnit(value=4.2, unit="kW"),
            np.array([4.2]),
            id="value-with-unit",
        ),
        pytest.param(
            PeriodValueWithUnit(values=[1.0, 2.0], unit="kW"),
            np.array([1.0, 2.0]),
            id="period_valued",
        ),
        pytest.param(
            PeriodValueWithUnitAndWeights(
                values=[1.0, 2.0],
                weights=[0.4, 0.6],
                unit="kW",
            ),
            np.array([1.0, 2.0]),
            id="period_valued-with-weights",
        ),
        pytest.param(None, np.array([]), id="none"),
    ],
)
def test_resolve_value_array_supported_inputs(payload, expected):
    np.testing.assert_allclose(resolve_value_array(payload), expected)


def test_repo_no_longer_uses_removed_value_helpers():
    repo_root = Path(__file__).resolve().parents[2]
    current_file = Path(__file__).resolve()
    patterns = (
        re.compile(r"\bget_value\("),
        re.compile(r"\bget_values\("),
        re.compile(r"\bresolve_value_for_state\("),
    )
    search_roots = [
        repo_root / "OpenPinch",
        repo_root / "tests",
    ]

    offenders: list[str] = []
    for root in search_roots:
        for path in root.rglob("*"):
            if path == current_file or path.suffix not in {".py", ".ipynb"}:
                continue
            text = path.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in patterns):
                offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []

"""Regression tests for explicit value-resolution helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.utils.value_resolution as value_resolution
from OpenPinch.classes import Value
from OpenPinch.classes.value import Q_
from OpenPinch.lib.schemas.common import (
    PeriodValueWithUnit,
    PeriodValueWithUnitAndWeights,
    ValueWithUnit,
)
from OpenPinch.utils.value_resolution import (
    get_period_value,
    get_scalar_value,
    resolve_value_array,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "value_resolution_cases.json"
)


def _value_resolution_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_resolve_period_value_scalar_passthrough():
    assert get_period_value(3.14) == pytest.approx(3.14)


def test_resolve_period_value_none_passthrough():
    assert get_period_value(None) is None


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


def test_resolve_period_value_static_fixture_edge_cases():
    fixture = _value_resolution_fixture()

    assert get_period_value(fixture["period_payload"], period_idx=2) == pytest.approx(
        9.0
    )
    assert get_period_value(fixture["none_values_payload"]) is None
    with pytest.raises(KeyError, match="values"):
        get_period_value(fixture["missing_values_payload"])
    with pytest.raises(TypeError, match="Boolean values"):
        get_period_value({"values": True, "unit": "kW"})
    with pytest.raises(ValueError, match="cannot be empty"):
        get_period_value(fixture["empty_values_payload"])

    period_value = Value(fixture["period_payload"])
    assert get_period_value(period_value, period_idx=1) == pytest.approx(6.0)
    period_object = SimpleNamespace(**fixture["period_object"])
    assert get_period_value(period_object, period_idx=1) == pytest.approx(10.0)
    scalar_object = SimpleNamespace(**fixture["scalar_object"])
    assert get_period_value(scalar_object) == pytest.approx(12.5)
    assert get_period_value(Q_(2500.0, "W")) == pytest.approx(2500.0)

    with pytest.raises(TypeError, match="Unsupported string value"):
        get_period_value("not-a-number")
    with pytest.raises(TypeError, match="Unsupported type"):
        get_period_value(object())


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
    with pytest.raises(TypeError, match="Unsupported mapping"):
        get_scalar_value(payload)


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


def test_resolve_value_array_static_fixture_edge_cases():
    fixture = _value_resolution_fixture()

    np.testing.assert_allclose(
        resolve_value_array(fixture["period_payload"]),
        np.array([3.0, 6.0, 9.0]),
    )
    np.testing.assert_allclose(
        resolve_value_array(fixture["scalar_none_payload"]),
        np.array([]),
    )
    np.testing.assert_allclose(
        resolve_value_array(Value([1.0, 2.0], unit="kW")),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        resolve_value_array(SimpleNamespace(**fixture["period_object"])),
        np.array([5.0, 10.0]),
    )
    np.testing.assert_allclose(
        resolve_value_array(SimpleNamespace(**fixture["scalar_object"])),
        np.array([12.5]),
    )
    np.testing.assert_allclose(resolve_value_array("7.25"), np.array([7.25]))
    np.testing.assert_allclose(resolve_value_array(Q_(2.0, "kW")), np.array([2.0]))

    assert resolve_value_array(SimpleNamespace(value=None, unit="kW")).size == 0
    with pytest.raises(TypeError, match="Boolean values"):
        resolve_value_array(False)
    with pytest.raises(KeyError, match="values"):
        resolve_value_array(fixture["missing_values_payload"])
    with pytest.raises(TypeError, match="Unsupported mapping"):
        resolve_value_array({"zone-a": 11.25})
    with pytest.raises(TypeError, match="Unsupported type"):
        resolve_value_array(object())


def test_value_resolution_private_guards_cover_non_mapping_and_none_scalar():
    assert value_resolution._is_period_value_data(object()) is False
    assert value_resolution._coerce_optional_float(None) is None


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

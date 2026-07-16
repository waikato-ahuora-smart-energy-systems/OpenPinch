"""Regression tests for public I/O schema defaults."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import OpenPinch.lib.schemas.report_units as report_units
from OpenPinch.classes.value import Value
from OpenPinch.lib.schemas.common import (
    PeriodValueWithUnit,
)
from OpenPinch.lib.schemas.common import (
    ValueWithUnit as VU,
)
from OpenPinch.lib.schemas.report_units import split_report_value

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "report_units_cases.json"
)


def _report_units_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_split_vu_handles_none_plain_number_and_object():
    assert split_report_value(None) == (None, None)
    assert split_report_value(3) == (3.0, None)
    assert split_report_value(3.14) == (3.14, None)
    assert split_report_value(VU(value=12.3, unit="kg/s")) == (12.3, "kg/s")


def test_split_vu_non_numeric_unparsable():
    # Strings that cannot be coerced to float return (None, None)
    assert split_report_value("not-a-number") == (None, None)


def test_split_vu_uses_period_idx_for_period_values():
    payload = PeriodValueWithUnit(values=[5.0, 8.5], unit="kW")

    assert split_report_value(payload, period_idx=1) == (8.5, "kW")


def test_split_vu_preserves_array_payloads():
    payload = Value([5.0, 8.5], unit="kW")

    assert split_report_value(payload) == ([5.0, 8.5], "kW")


def test_split_vu_resolves_period_value_period_idx():
    payload = Value([5.0, 8.5], unit="kW")

    assert split_report_value(payload, period_idx=1) == (8.5, "kW")


def test_split_report_value_static_array_and_mapping_payloads():
    fixture = _report_units_fixture()

    assert split_report_value(fixture["array_payload"]) == ([1.0, 2.5, 4.0], "kW")
    assert split_report_value(fixture["scalar_none_payload"]) == (None, "kW")
    assert split_report_value(fixture["period_single_payload"]) == (7.0, "MW")
    assert split_report_value(fixture["period_multi_payload"]) == (7.0, "MW")
    assert split_report_value(fixture["period_multi_payload"], period_idx=1) == (
        9.0,
        "MW",
    )
    assert split_report_value(fixture["period_multi_payload"], period_idx=3) == (
        None,
        "MW",
    )
    assert split_report_value(fixture["period_empty_payload"]) == (None, "MW")

    try:
        split_report_value(fixture["empty_array_payload"])
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("empty array payload should fail")


def test_split_report_value_static_object_and_array_like_payloads():
    fixture = _report_units_fixture()

    assert split_report_value(np.array([1.0, 2.0])) == ([1.0, 2.0], None)
    assert split_report_value(Value(12.0, unit="kW")) == (12.0, "kW")
    assert split_report_value(SimpleNamespace(**fixture["scalar_object"])) == (
        6.5,
        "kg/s",
    )
    assert split_report_value(
        SimpleNamespace(**fixture["period_object"]),
        period_idx=1,
    ) == (6.0, "kW")

    class DumpableScalar:
        def model_dump(self, *, mode: str):
            assert mode == "python"
            return fixture["scalar_object"]

    assert split_report_value(DumpableScalar()) == (6.5, "kg/s")


def test_report_unit_private_scalar_and_array_guards():
    assert report_units._is_numeric_array_like(None) is False
    assert report_units._is_numeric_array_like(1.0) is False
    assert report_units._is_numeric_array_like([object()]) is False
    assert report_units._resolve_scalar_value(None) is None
    assert report_units._resolve_scalar_value(Value(2.0, unit="kW")) == 2.0
    assert report_units._resolve_scalar_value(Value([2.0, 4.0], unit="kW")) == 2.0
    assert report_units._resolve_scalar_value(3.5) == 3.5

    try:
        report_units._resolve_scalar_value(True)
    except TypeError as exc:
        assert "Boolean values" in str(exc)
    else:
        raise AssertionError("boolean scalar should fail")

    try:
        report_units._resolve_scalar_value(object())
    except TypeError as exc:
        assert "Unsupported scalar report value type" in str(exc)
    else:
        raise AssertionError("unsupported scalar should fail")

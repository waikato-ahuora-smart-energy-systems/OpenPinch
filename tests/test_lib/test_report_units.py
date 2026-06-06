"""Regression tests for public I/O schema defaults."""

from OpenPinch.classes.value import Value
from OpenPinch.lib.schemas.common import (
    StatefulValueWithUnit,
)
from OpenPinch.lib.schemas.common import (
    ValueWithUnit as VU,
)
from OpenPinch.lib.schemas.report_units import split_report_value


def test_split_vu_handles_none_plain_number_and_object():
    assert split_report_value(None) == (None, None)
    assert split_report_value(3) == (3.0, None)
    assert split_report_value(3.14) == (3.14, None)
    assert split_report_value(VU(value=12.3, unit="kg/s")) == (12.3, "kg/s")


def test_split_vu_non_numeric_unparsable():
    # Strings that cannot be coerced to float return (None, None)
    assert split_report_value("not-a-number") == (None, None)


def test_split_vu_uses_idx_for_stateful_values():
    payload = StatefulValueWithUnit(values=[5.0, 8.5], unit="kW")

    assert split_report_value(payload, idx=1) == (8.5, "kW")


def test_split_vu_preserves_array_payloads():
    payload = Value([5.0, 8.5], unit="kW")

    assert split_report_value(payload) == ([5.0, 8.5], "kW")


def test_split_vu_resolves_stateful_value_idx():
    payload = Value([5.0, 8.5], unit="kW")

    assert split_report_value(payload, idx=1) == (8.5, "kW")

"""Additional coverage tests for water property conversion edge cases."""

from OpenPinch.utils import water_properties as wp


def test_none_inputs_default_to_zero_for_unit_conversions():
    assert wp.toSIunit_p(None) == 0
    assert wp.fromSIunit_p(None) == 0
    assert wp.toSIunit_T(None) == 273.15
    assert wp.fromSIunit_T(None) == -273.15
    assert wp.toSIunit_h(None) == 0
    assert wp.fromSIunit_h(None) == 0

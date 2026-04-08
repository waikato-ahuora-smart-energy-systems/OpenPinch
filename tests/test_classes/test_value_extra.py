"""Additional branch coverage tests for Value."""

from OpenPinch.classes.value import Value


def test_value_eq_handles_conversion_errors():
    v = Value(1.0, "m")
    assert (v == object()) is False

    v._to_quantity = lambda other: (_ for _ in ()).throw(ValueError("boom"))
    assert (v == Value(2.0, "m")) is False


def test_value_reverse_arithmetic_paths():
    v = Value(3.0)
    assert (2 + v).value == 5.0
    assert (10 - v).value == 7.0
    assert (12 / v).value == 4.0

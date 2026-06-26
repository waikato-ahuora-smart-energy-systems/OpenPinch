"""Regression tests for the simplified state-array ``Value`` model."""

import pickle

import numpy as np
import pytest

from OpenPinch.classes.value import Value


def test_scalar_value_behaves_like_scalar_quantity():
    value = Value(12.5, "kW")

    assert value.value == pytest.approx(12.5)
    assert float(value) == pytest.approx(12.5)
    assert value.unit == "kW"
    assert value.to_dict() == {"value": 12.5, "unit": "kW"}


def test_period_value_stores_only_ordered_magnitudes():
    value = Value(
        {
            "values": [10.0, 4.0],
            "period_ids": ["base", "peak"],
            "weights": [0.25, 0.75],
            "unit": "kW",
        }
    )

    np.testing.assert_allclose(value.period_values, np.array([10.0, 4.0]))
    assert value.to_dict() == {"values": [10.0, 4.0], "unit": "kW"}


def test_period_value_requires_explicit_summary_for_scalar_face():
    value = Value([10.0, 4.0], unit="kW")

    assert value.mean.value == pytest.approx(7.0)
    with pytest.raises(TypeError):
        float(value)


def test_period_value_getitem_is_position_based():
    value = Value([10.0, 4.0], unit="kW")

    assert value["0"].value == pytest.approx(10.0)
    assert value[1].value == pytest.approx(4.0)
    with pytest.raises(KeyError):
        _ = value["peak"]


def test_value_supports_pytest_approx_comparisons():
    value = Value(5.0, "Δ°C")
    result = value * 0.5
    assert result * 2 == value
    assert pytest.approx(value * 0.5) == result


def test_dimensionless_values_use_dash_unit_representation():
    value = Value(0.5)

    assert value.unit == "-"
    assert value.to_dict() == {"value": 0.5, "unit": "-"}


def test_currency_values_round_trip_through_pint_pickle():
    value = Value(12.5, "$/y")

    round_tripped = pickle.loads(pickle.dumps(value))

    assert round_tripped.value == pytest.approx(12.5)
    assert round_tripped.unit == "$/y"


def test_period_arithmetic_requires_matching_period_count():
    left = Value([10.0, 4.0], unit="kW")
    right = Value([1.0, 2.0], unit="kW")

    result = left + right

    np.testing.assert_allclose(result.period_values, np.array([11.0, 6.0]))
    with pytest.raises(ValueError, match="identical period counts"):
        _ = left + Value([1.0, 2.0, 3.0], unit="kW")


def test_period_value_slice_preserves_magnitudes():
    value = Value([10.0, 4.0, 2.0], unit="kW")

    subset = value[1:]
    np.testing.assert_allclose(subset.period_values, np.array([4.0, 2.0]))


def test_value_array_protocol_accepts_copy_keyword():
    scalar = Value(12.5, "kW")
    period_valued = Value([10.0, 4.0], unit="kW")

    scalar_array = np.array(scalar, copy=True)
    period_array = np.array(period_valued, copy=True)

    assert scalar_array == pytest.approx(12.5)
    np.testing.assert_allclose(period_array, np.array([10.0, 4.0]))

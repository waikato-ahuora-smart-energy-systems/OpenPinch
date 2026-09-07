"""Value invariants found in the second repository audit cycle."""

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from OpenPinch.domain.value import Value


@settings(max_examples=30, deadline=None)
@example(values=[10.0, 20.0])
@given(
    values=st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_one_shot_iterable_matches_materialized_values(values):
    actual = Value((item for item in values), "kW")
    expected = Value(values, "kW")
    np.testing.assert_array_equal(actual.period_values, expected.period_values)
    assert actual.unit == expected.unit


def test_iterable_detection_does_not_consume_input():
    iterator = iter([10.0, 20.0])
    assert Value._is_array_like_input(iterator)
    assert list(iterator) == [10.0, 20.0]


@pytest.mark.parametrize(
    "derive",
    [
        lambda value: value[1],
        lambda value: value.mean,
        lambda value: value.min,
        lambda value: value.weighted_mean,
    ],
)
def test_scalar_derivations_have_compatible_weight_metadata(derive):
    value = Value(
        {"values": [10.0, 20.0, 40.0], "weights": [1.0, 2.0, 1.0], "unit": "kW"}
    )
    scalar = derive(value)
    assert scalar.weighted_mean.value == pytest.approx(scalar.value)
    assert scalar.weights is None or len(scalar.weights) == scalar.num_periods


@pytest.mark.parametrize(
    "derive",
    [
        lambda value: value.mutable_copy(),
        lambda value: value + value,
        lambda value: value[1:],
    ],
)
def test_derived_weights_do_not_expose_source_metadata(derive):
    value = Value(
        {"values": [10.0, 20.0, 40.0], "weights": [1.0, 2.0, 1.0], "unit": "kW"}
    )
    result = derive(value)
    before = value.weighted_mean.value
    result.weights[0] = 100.0
    assert value.weighted_mean.value == before


def test_weights_are_detached_from_input_and_public_view():
    weights = [1.0, 3.0]
    value = Value({"values": [10.0, 20.0], "weights": weights, "unit": "kW"})
    weights[0] = 100.0
    assert value.weighted_mean.value == pytest.approx(17.5)
    value.weights[0] = 100.0
    assert value.weighted_mean.value == pytest.approx(17.5)


def test_weight_length_is_validated_at_construction():
    with pytest.raises(ValueError, match="weights length"):
        Value({"values": [10.0, 20.0], "weights": [1.0], "unit": "kW"})

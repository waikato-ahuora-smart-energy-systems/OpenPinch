"""Regression tests for the simplified state-array ``Value`` model."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.classes.value import Q_, Value

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "value_and_stream_edge_cases.json"
)


def _edge_case_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


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


def test_value_normalises_unit_aliases_from_static_fixture():
    for case in _edge_case_fixture()["value_unit_aliases"]:
        assert Value(1.0, case["input"]).unit == case["expected_unit"]
    assert Value(1.0, "").unit == "-"
    assert Value(1.0, "C").unit == "degC"


def test_period_value_summary_iteration_and_weight_preserving_slice():
    value = Value(_edge_case_fixture()["weighted_period_value"])

    assert value.min.value == pytest.approx(10.0)
    assert value.max.value == pytest.approx(40.0)
    assert value.median.value == pytest.approx(20.0)
    assert value.weighted_mean.value == pytest.approx(22.5)
    assert list(value) == [10.0, 20.0, 40.0]

    sliced = value[1:]
    np.testing.assert_allclose(sliced.period_values, np.array([20.0, 40.0]))
    np.testing.assert_allclose(sliced.weights, np.array([2.0, 1.0]))


def test_scalar_value_rejects_iteration_and_bool_assignment():
    value = Value(1.0, "kW")

    value.value = 2.0
    assert value.value == pytest.approx(2.0)
    assert str(value) == "2.0 kW"
    assert format(value, ".1f") == "2.0"
    assert abs(Value(-2.0, "kW")) == pytest.approx(2.0)

    with pytest.raises(TypeError, match="Scalar Value is not iterable"):
        list(value)
    with pytest.raises(TypeError, match="Boolean values are not supported"):
        value.value = True


def test_period_value_assignment_accepts_scalar_broadcast_and_validates_lengths():
    value = Value([1.0, 2.0], unit="kW")

    value.value = 3.0
    np.testing.assert_allclose(value.period_values, np.array([3.0, 3.0]))
    assert value.values == [3.0, 3.0]
    assert repr(value) == "Value(values=[3.0, 3.0], unit='kW')"
    assert value.to_dict() == {"values": [3.0, 3.0], "unit": "kW"}

    with pytest.raises(ValueError, match="length must match"):
        value.value = [1.0]
    with pytest.raises(TypeError, match="Boolean values"):
        value.value = [True, False]


def test_value_scalar_magic_methods_and_period_scalar_guards():
    scalar = Value(2.4, "kW")
    period_valued = Value([1.0, 2.0], unit="kW")

    scalar.unit = "W"
    assert scalar.value == pytest.approx(2400.0)
    scalar.unit = "kW"
    assert scalar.value == pytest.approx(2.4)
    assert repr(scalar) == "Value(2.4, 'kW')"
    assert int(Value(2.9, "kW")) == 2
    assert round(scalar) == 2
    assert round(scalar, 1) == pytest.approx(2.4)
    assert (-scalar) == pytest.approx(-2.4)
    assert (+scalar) == pytest.approx(2.4)

    with pytest.raises(TypeError, match="multiperiod"):
        int(period_valued)
    with pytest.raises(TypeError, match="multiperiod"):
        round(period_valued)


def test_value_comparisons_return_false_for_incompatible_units_and_types():
    power = Value(1.0, "kW")
    temperature = Value(1.0, "degC")

    assert power == 1.0
    assert power < 2.0
    assert power <= 1.0
    assert power > 0.0
    assert power >= 1.0
    assert power == Value(1000.0, "W")
    assert power < Value(2.0, "kW")
    assert power <= Value(1.0, "kW")
    assert power > Value(500.0, "W")
    assert power >= Value(1.0, "kW")
    assert (power == temperature) is False
    assert (power < temperature) is False
    assert (power <= temperature) is False
    assert (power > temperature) is False
    assert (power >= temperature) is False
    assert (power == object()) is False
    assert (power < object()) is False
    assert (power <= object()) is False
    assert (power > object()) is False
    assert (power >= object()) is False


def test_value_reverse_numeric_operations_preserve_units():
    value = Value(2.0, "kW")

    assert (value + 1.0).value == pytest.approx(3.0)
    assert (value + Value(1.0, "kW")).value == pytest.approx(3.0)
    assert (value - 1.0).value == pytest.approx(1.0)
    assert (value - Value(500.0, "W")).value == pytest.approx(1.5)
    assert (value / 2.0).value == pytest.approx(1.0)
    assert (1.0 + value).value == pytest.approx(3.0)
    assert value.__radd__(Value(1.0, "kW")).value == pytest.approx(3.0)
    assert (5.0 - value).value == pytest.approx(3.0)
    assert value.__rsub__(Value(5.0, "kW")).value == pytest.approx(3.0)
    assert (3.0 * value).value == pytest.approx(6.0)
    assert (8.0 / value).value == pytest.approx(4.0)


def test_value_rejects_invalid_period_indexes_and_empty_values():
    value = Value([1.0, 2.0], unit="kW")

    with pytest.raises(IndexError):
        _ = value[-1]
    with pytest.raises(IndexError):
        _ = value[2]
    with pytest.raises(ValueError, match="cannot be empty"):
        Value([], unit="kW")
    with pytest.raises(ValueError, match="cannot be empty"):
        Value(1.0, unit="kW")._set_storage(Q_(np.array([]), "kW"))


def test_value_setitem_accepts_value_and_pint_quantities():
    value = Value([1.0, 2.0], unit="kW")

    value[0] = Value(0.5, "MW")
    value[1] = Q_(250.0, "W")

    np.testing.assert_allclose(value.period_values, np.array([500.0, 0.25]))


def test_value_accepts_object_and_model_dump_inputs_from_static_fixture():
    fixture = _edge_case_fixture()
    object_data = SimpleNamespace(**fixture["object_value"])

    quantity_value = Value(Q_(1.5, "MW"))
    assert quantity_value.value == pytest.approx(1.5)
    assert quantity_value.unit == "MW"

    object_value = Value(object_data)
    np.testing.assert_allclose(object_value.period_values, np.array([1000.0, 2000.0]))
    np.testing.assert_allclose(object_value.weights, np.array([0.25, 0.75]))
    assert object_value.unit == "W"

    class DumpableValue:
        def model_dump(self, *, mode: str):
            assert mode == "python"
            return fixture["model_dump_value"]

    dumped_value = Value(DumpableValue())
    assert dumped_value.value == pytest.approx(2.5)
    assert dumped_value.unit == "MW"

    scalar_object = Value(SimpleNamespace(value=None, unit="kW"))
    assert scalar_object.to_dict() == {"value": None, "unit": "kW"}


def test_value_mapping_and_dimensionless_fallback_paths():
    default = Value()
    assert default.value == pytest.approx(0.0)
    assert default.unit == "-"

    mapped = Value({"base": 1.0, "peak": 2.0}, unit="kW")
    np.testing.assert_allclose(mapped.period_values, np.array([1.0, 2.0]))

    dimensionless = Value(Value(5.0), unit="kW")
    assert dimensionless.unit == "kW"
    assert dimensionless.value == pytest.approx(5.0)

    same_dimensionality = Value(Value(5.0, "delta_degC"), unit="degC")
    assert same_dimensionality.unit == "degC"
    assert same_dimensionality.value == pytest.approx(5.0)

    with pytest.raises(Exception):
        Value(Value(5.0, "kW"), unit="degC")


def test_value_rejects_non_numeric_and_malformed_inputs():
    with pytest.raises(TypeError, match="Boolean values"):
        Value(True)
    with pytest.raises(TypeError, match="Boolean values"):
        Value([1.0, False], unit="kW")
    with pytest.raises(TypeError, match="Boolean values"):
        Value(1.0, unit="kW")._coerce_magnitude_array(
            True,
            expected_len=1,
            label="values",
        )
    assert Value(["1.5"], unit="kW").value == pytest.approx(1.5)
    np.testing.assert_allclose(
        Value(1.0, unit="kW")._coerce_magnitude_array(
            "1.5",
            expected_len=1,
            label="values",
        ),
        np.array([1.5]),
    )
    with pytest.raises(TypeError, match="must contain numeric values"):
        Value(["not-numeric"], unit="kW")
    with pytest.raises(TypeError, match="must be numeric scalar"):
        Value(1.0, unit="kW")._coerce_magnitude_array(
            object(),
            expected_len=1,
            label="values",
        )
    with pytest.raises(ValueError, match="length must match"):
        Value(1.0, unit="kW")._coerce_magnitude_array(
            [1.0, 2.0],
            expected_len=1,
            label="values",
        )


def test_value_serialisation_handles_nan_and_from_dict_validation():
    missing = Value({"value": None, "unit": "kW"})

    assert missing.to_dict() == {"value": None, "unit": "kW"}
    assert Value.from_dict({"value": 3.0, "unit": "kW"}).value == pytest.approx(3.0)
    np.testing.assert_allclose(
        Value.from_dict({"values": [1.0, 2.0], "unit": "kW"}).period_values,
        np.array([1.0, 2.0]),
    )
    with pytest.raises(TypeError, match="data must be a mapping"):
        Value.from_dict([1.0])


def test_value_private_type_helpers_cover_negative_paths():
    assert Value._is_array_like_input(iter([1.0, 2.0])) is True
    assert Value._is_array_like_input(None) is False
    assert Value._is_array_like_input("12") is False
    assert Value._is_array_like_input({"a": 1.0}) is False
    assert Value._is_array_like_input(object()) is False
    assert Value._quantity_is_dimensionless(Q_(1.0, "")) is True
    assert Value._same_dimensionality(Q_(1.0, "kW"), "not-a-unit") is False
    assert Value._normalise_weights(None, expected_len=1) is None
    np.testing.assert_allclose(
        Value._normalise_weights([1.0, 3.0], expected_len=2),
        np.array([0.25, 0.75]),
    )
    with pytest.raises(ValueError, match="weights length"):
        Value._normalise_weights([1.0, 2.0], expected_len=1)

"""Regression tests for the simplified state-array ``Value`` model."""

import copy
import json
import pickle
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pint
import pytest

from OpenPinch.domain.value import Q_, Value
from tests.support.paths import FIXTURES_ROOT
from tests.support.quantity_count import count_quantities

FIXTURE_PATH = FIXTURES_ROOT / "value_and_stream_edge_cases.json"


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
            "weights": [0.25, 0.75],
            "unit": "kW",
        }
    )

    np.testing.assert_allclose(value.period_values, np.array([10.0, 4.0]))
    assert value.to_dict() == {
        "values": [10.0, 4.0],
        "unit": "kW",
        "weights": [0.25, 0.75],
    }


def test_period_value_rejects_period_ids():
    # A Value holds magnitudes in order and has no period identity, so ids
    # given beside them used to be dropped silently: {"values": [10, 20],
    # "period_ids": ["b", "a"]} gave the first period 10 whatever its id.
    # It now raises, as a domain Stream does.
    with pytest.raises(ValueError, match="do not take period_ids"):
        Value(
            {
                "values": [10.0, 4.0],
                "period_ids": ["base", "peak"],
                "weights": [0.25, 0.75],
                "unit": "kW",
            }
        )


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
    assert not hasattr(value, "values")
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


@pytest.mark.parametrize("unit", ["k$/y", "k$/yr", "M$/y"])
def test_prefixed_annual_cost_units_parse(unit):
    value = Value(2.0, unit)

    scale = 1e6 if unit.startswith("M") else 1e3
    assert value.to("$/y").value == pytest.approx(2.0 * scale)
    assert Value(2000.0, "$").to("k$").value == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [("degC", 5.0), ("C", 5.0), ("°C", 5.0), ("degF", 5.0 / 1.8)],
)
def test_to_converts_an_offset_temperature_as_a_difference(unit, expected):
    converted = Value(5.0, unit).to("delta_degC")

    assert converted.unit == "delta_degC"
    assert converted.value == pytest.approx(expected, rel=1e-12)


def test_to_still_rejects_another_dimension():
    with pytest.raises(Exception, match="Cannot convert"):
        Value(5.0, "kW").to("delta_degC")


def test_to_does_not_convert_a_difference_to_an_offset_temperature():
    with pytest.raises(Exception, match="Cannot convert"):
        Value(5.0, "delta_degC").to("degC")


def test_stored_magnitudes_cannot_be_written_in_place():
    magnitudes = Value([1.0, 2.0], "kW")._quantity.magnitude

    with pytest.raises(ValueError, match="read-only"):
        magnitudes[0] = 5.0


def test_deepcopy_builds_no_quantity():
    value = Value([1.0, 2.0], "kW")

    with count_quantities() as built:
        copy.deepcopy(value)

    assert built[0] == 0


def test_deepcopy_is_isolated_and_keeps_weights_and_read_only_state():
    value = Value({"values": [1.0, 3.0], "weights": [1.0, 2.0], "unit": "kW"})
    frozen = copy.deepcopy(Value(1.0, "kW")._make_read_only("owned"))
    copied = copy.deepcopy(value)

    copied[0] = 9.0
    copied._weights[0] = 7.0

    assert value.period_values.tolist() == [1.0, 3.0]
    assert value.weights.tolist() == [1.0, 2.0]
    assert copied.period_values.tolist() == [9.0, 3.0]
    with pytest.raises(TypeError, match="owned"):
        frozen.value = 2.0


def test_value_from_value_in_the_same_unit_builds_no_quantity():
    value = Value([1.0, 2.0], "kW")

    with count_quantities() as built:
        Value(value)
        Value(value, "kW")

    assert built[0] == 0


def test_value_from_a_scalar_builds_one_quantity():
    with count_quantities() as built:
        value = Value(1.5, "kW")

    assert built[0] == 1
    assert value.to_dict() == {"value": 1.5, "unit": "kW"}


def test_scalar_shortcut_matches_the_general_scalar_path():
    class FloatSubclass(float):
        pass

    for unit in ("degC", "-", "$/y", None):
        plain = Value(1.5, unit)
        general = Value(FloatSubclass(1.5), unit)

        assert plain.to_dict() == general.to_dict()
        assert plain.unit == general.unit


def test_to_the_same_unit_builds_no_quantity_and_is_isolated():
    value = Value({"values": [1.0, 3.0], "weights": [1.0, 2.0], "unit": "kW"})
    value._make_read_only("owned")

    with count_quantities() as built:
        same = value.to("kW")

    assert built[0] == 0
    same[0] = 9.0
    same._weights[0] = 7.0
    assert value.period_values.tolist() == [1.0, 3.0]
    assert value.weights.tolist() == [1.0, 2.0]


def test_value_from_a_foreign_registry_quantity_in_the_same_unit():
    foreign = pint.UnitRegistry().Quantity(np.array([1.5]), "kW")

    assert Value(foreign, "kW").to_dict() == {"value": 1.5, "unit": "kW"}


@pytest.mark.parametrize("protocol", [4, 5])
def test_unpickled_magnitudes_stay_read_only(protocol):
    # Protocol 4 is the default before Python 3.14 and drops NumPy's read-only flag.
    restored = pickle.loads(pickle.dumps(Value([1.0, 2.0], "kW"), protocol=protocol))

    with pytest.raises(ValueError, match="read-only"):
        restored._quantity.magnitude[0] = 5.0


def test_unit_text_is_formatted_once_per_unit():
    first = Value(1.0, "kW/m^2")
    second = Value(2.0, "kW/m^2")
    assert first.unit == "kW/m^2"

    with mock.patch(
        "OpenPinch.domain._value.units.clean_unit_text",
        side_effect=AssertionError("unit text formatted again"),
    ):
        assert second.unit == "kW/m^2"

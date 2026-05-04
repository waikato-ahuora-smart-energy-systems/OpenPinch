"""Regression tests for the value classes."""

import numpy as np
import pandas as pd
import pytest
from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.classes.value import Value


def test_heatflow_value_behavior():
    v = Value(12.5, "kW")
    assert v.value == 12.5
    assert v.unit == "kW"
    assert float(v) == 12.5
    assert int(v) == 12
    assert str(v) == "12.5 kW"
    assert repr(v) == "Value(12.5, 'kW')"
    assert int(Value(7)) == 7
    assert round(Value(3.14159), 2) == 3.14


def test_temperature_value_behavior():
    v = Value(15, "degC")
    assert v == 15.0
    assert v.unit == "degC"
    assert float(v) == 15
    assert int(v) == 15


def test_value_setters():
    v = Value()
    v.value = 99
    v.unit = "m^3/h"
    assert v.value == 99
    assert v.unit in ("m^3/h", "m**3/h")


def test_value_equality_and_conversion():
    v1 = Value(1, "kg")
    v2 = Value(1000, "g")
    assert v1 == v2
    assert float(v1 + v2) == 2 or 2000


def test_serialization_round_trip():
    v1 = Value(42.0, "mol/s")
    d = v1.to_dict()
    v2 = Value.from_dict(d)
    assert v1 == v2
    assert v2.unit == "mol/s"


def test_multiply():
    cp = Value(4, "kJ/degC")
    t1 = Value(10, "degC")
    t2 = Value(11, "degC")
    assert cp * (t2 - t1) == Value(4000, "J")
    assert cp * (t2 - t1) == Value(4, "kJ")
    assert cp * (t2 - t1) == 4


def test_divide():
    q = Value(4, "kW")
    t1 = Value(10, "degK")
    t2 = Value(11, "degK")
    assert q / (t2 - t1) == Value(4000, "W/degK")
    assert q / (t2 - t1) == Value(4, "kW/degK")
    assert q / (t2 - t1) == 4


def test_pandas():
    df = pd.DataFrame(
        {
            "stream": ["A", "B"],
            "flow": [Value(10, "kW/degC"), Value(20, "kW/K")],
            "t_in": [Value(100, "degC"), Value(373.15, "K")],
            "t_out": [Value(120, "degC"), Value(383.15, "K")],
        }
    )
    df["q"] = df["flow"] * (df["t_out"] - df["t_in"])

    # Check resulting q values
    assert isinstance(df.loc[0, "q"], Value)
    assert isinstance(df.loc[1, "q"], Value)

    # For stream A: 10 kW/degC * (120 - 100) degC = 200 kW
    assert abs(float(df.loc[0, "q"]) - 200) < 1e-6
    assert df.loc[0, "q"].unit == "kW"

    # For stream B: 20 kW/K * (383.15 - 373.15) K = 200 kW
    assert abs(df.loc[1, "q"].value - 200) < 1e-6
    assert df.loc[1, "q"].unit == "kW"


def test_valuewithunit_with_mismatched_unit():
    vw = ValueWithUnit(value=100, units="degC")
    v = Value(vw, unit="K")  # mismatched unit

    # It should keep the original "degC" and silently ignore the mismatch
    assert v.unit == "degC"


def test_value_from_valuewithunit():
    # Simulate input from schema or external source
    vw = ValueWithUnit(value=123.45, units="kW")

    # Create a Value instance from it
    v = Value(vw)

    # Check value and unit
    assert isinstance(v, Value)
    assert v.value == 123.45
    assert v.unit == "kW"

    # Confirm it behaves like a quantity
    assert float(v) == 123.45
    assert str(v) == "123.45 kW"


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


def test_stateful_value_behavior():
    v = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 7.0, "off": 3.0})

    np.testing.assert_allclose(v.value, np.array([10.0, 0.0]))
    assert v.value[0] == pytest.approx(10.0)
    assert v.weighted_value == pytest.approx(7.0)
    assert float(v) == pytest.approx(7.0)
    assert int(v) == 7
    assert round(v, 1) == 7.0
    assert v.unit == "kW"
    assert v.state_ids == ["on", "off"]
    np.testing.assert_allclose(v.state_values, np.array([10.0, 0.0]))
    np.testing.assert_allclose(v.weights, np.array([0.7, 0.3]))
    assert "states=['on', 'off']" in str(v)
    assert "state_ids=['on', 'off']" in repr(v)


def test_stateful_value_defaults_to_uniform_weights():
    v = Value({"on": 10.0, "off": 4.0}, "kW")

    np.testing.assert_allclose(v.value, np.array([10.0, 4.0]))
    np.testing.assert_allclose(v.weights, np.array([0.5, 0.5]))
    assert v.weighted_value == pytest.approx(7.0)


def test_stateful_value_accepts_explicit_values_and_state_id_keywords():
    v = Value(
        values=np.array([1.0, 0.0]),
        state_id=["on", "off"],
        weights=np.array([0.5, 0.5]),
    )

    np.testing.assert_allclose(v.value, np.array([1.0, 0.0]))
    assert v.value[0] == pytest.approx(1.0)
    assert v.weighted_value == pytest.approx(0.5)
    assert v.state_ids == ["on", "off"]
    np.testing.assert_allclose(v.state_values, np.array([1.0, 0.0]))
    np.testing.assert_allclose(v.weights, np.array([0.5, 0.5]))


def test_stateful_value_add_states_with_explicit_weight():
    v = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 0.7, "off": 0.3})

    v.add_states("standby", 5.0, weight=0.2)

    assert v.state_ids == ["on", "off", "standby"]
    np.testing.assert_allclose(v.state_values, np.array([10.0, 0.0, 5.0]))
    np.testing.assert_allclose(v.weights, np.array([0.56, 0.24, 0.2]))
    assert v.value[2] == pytest.approx(5.0)
    assert v.weighted_value == pytest.approx(6.6)


def test_stateful_value_add_states_defaults_new_weights_by_uniform_state_count():
    v = Value({"on": 10.0, "off": 4.0}, "kW")

    v.add_states(["standby", "idle"], np.array([1.0, 2.0]))

    assert v.state_ids == ["on", "off", "standby", "idle"]
    np.testing.assert_allclose(v.state_values, np.array([10.0, 4.0, 1.0, 2.0]))
    np.testing.assert_allclose(v.weights, np.array([0.25, 0.25, 0.25, 0.25]))
    assert v.weighted_value == pytest.approx(4.25)


def test_stateful_value_serialization_round_trip():
    v1 = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 0.7, "off": 0.3})

    d = v1.to_dict()
    v2 = Value.from_dict(d)
    v3 = Value.from_dict(
        {
            "values": [10.0, 0.0],
            "state_ids": ["on", "off"],
            "weights": [7.0, 3.0],
            "units": "kW",
        }
    )

    assert d == {
        "values": [10.0, 0.0],
        "state_ids": ["on", "off"],
        "weights": [0.7, 0.3],
        "unit": "kW",
    }
    np.testing.assert_allclose(v2.value, v1.value)
    assert v2.unit == "kW"
    assert v2.state_ids == ["on", "off"]
    np.testing.assert_allclose(v2.state_values, v1.state_values)
    np.testing.assert_allclose(v2.weights, v1.weights)
    np.testing.assert_allclose(v3.value, v1.value)
    np.testing.assert_allclose(v3.weights, v1.weights)


def test_stateful_value_arithmetic_and_conversion():
    v = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 0.7, "off": 0.3})
    scalar = Value(2.0, "kW")

    added = v + scalar
    converted = v.to("W")

    np.testing.assert_allclose(added.value, np.array([12.0, 2.0]))
    np.testing.assert_allclose(added.state_values, np.array([12.0, 2.0]))
    np.testing.assert_allclose(added.weights, v.weights)
    assert added.state_ids == v.state_ids
    assert added.weighted_value == pytest.approx(9.0)

    np.testing.assert_allclose(converted.value, np.array([10_000.0, 0.0]))
    np.testing.assert_allclose(converted.state_values, np.array([10_000.0, 0.0]))
    np.testing.assert_allclose(converted.weights, v.weights)
    assert converted.unit == "W"

    other = Value({"on": 1.0, "off": 5.0}, "kW", weights={"on": 0.7, "off": 0.3})
    product = v * Value({"on": 2.0, "off": 4.0}, weights={"on": 0.7, "off": 0.3})

    np.testing.assert_allclose((v + other).state_values, np.array([11.0, 5.0]))
    np.testing.assert_allclose(product.state_values, np.array([20.0, 0.0]))
    assert product.weighted_value == pytest.approx(14.0)


def test_stateful_value_comparisons_use_weighted_scalar():
    v = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 0.7, "off": 0.3})
    other_weights = Value(
        {"high": 9.0, "low": 5.0},
        "kW",
        weights={"high": 0.5, "low": 0.5},
    )

    assert v == 7.0
    assert v >= Value(7.0, "kW")
    assert v > Value(6.0, "kW")
    assert v < Value(8.0, "kW")
    assert v == other_weights


def test_stateful_value_rejects_mismatched_arithmetic_inputs():
    v = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 0.7, "off": 0.3})

    with pytest.raises(ValueError, match="state_ids"):
        _ = v + Value({"off": 1.0, "on": 2.0}, "kW", weights={"off": 0.3, "on": 0.7})

    with pytest.raises(ValueError, match="weights"):
        _ = v + Value({"on": 1.0, "off": 2.0}, "kW", weights={"on": 0.5, "off": 0.5})


@pytest.mark.parametrize(
    ("state_id", "value", "weight", "error_type", "match"),
    [
        ("on", 1.0, 0.2, ValueError, "Duplicate state_ids"),
        (["standby", "idle"], [1.0], None, ValueError, "value length"),
        ("standby", 1.0, 1.2, ValueError, "sum to 1.0 or less"),
        ("standby", 1.0, -0.1, ValueError, "non-negative"),
    ],
)
def test_stateful_value_add_states_rejects_invalid_inputs(
    state_id, value, weight, error_type, match
):
    v = Value({"on": 10.0, "off": 0.0}, "kW", weights={"on": 0.7, "off": 0.3})

    with pytest.raises(error_type, match=match):
        v.add_states(state_id, value, weight=weight)


def test_scalar_value_add_states_rejected():
    v = Value(10.0, "kW")

    with pytest.raises(TypeError, match="stateful Value"):
        v.add_states("on", 1.0, weight=0.5)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        (
            {
                "data": {"on": 1.0, "off": 0.0},
                "unit": "kW",
                "weights": {"on": -1.0, "off": 1.0},
            },
            ValueError,
            "non-negative",
        ),
        (
            {
                "data": {"on": 1.0, "off": 0.0},
                "unit": "kW",
                "weights": {"on": 0.0, "off": 0.0},
            },
            ValueError,
            "positive",
        ),
        (
            {"data": {"on": 1.0, "off": 0.0}, "unit": "kW", "weights": {"on": 1.0}},
            ValueError,
            "match",
        ),
        (
            {"data": {}, "unit": "kW"},
            ValueError,
            "empty",
        ),
        (
            {"data": [1.0, 2.0], "unit": "kW"},
            TypeError,
            "state_ids",
        ),
        (
            {
                "data": [1.0, 2.0],
                "unit": "kW",
                "state_ids": ["on", "off"],
                "weights": [1.0],
            },
            ValueError,
            "weights length",
        ),
        (
            {"data": 1.0, "unit": "kW", "weights": [1.0]},
            TypeError,
            "weights can only",
        ),
        (
            {"data": 1.0, "values": [1.0, 0.0], "state_id": ["on", "off"]},
            TypeError,
            "either data or values",
        ),
        (
            {
                "values": [1.0, 0.0],
                "state_id": ["on", "off"],
                "state_ids": ["on", "off"],
            },
            TypeError,
            "either state_id or state_ids",
        ),
    ],
)
def test_stateful_value_rejects_invalid_inputs(kwargs, error_type, match):
    with pytest.raises(error_type, match=match):
        Value(
            kwargs.get("data"),
            kwargs.get("unit"),
            values=kwargs.get("values"),
            weights=kwargs.get("weights"),
            state_id=kwargs.get("state_id"),
            state_ids=kwargs.get("state_ids"),
        )


def test_value_arithmetic_comparison_and_conversion_paths():
    v = Value(10.0, "kW")
    assert str(v)
    assert repr(v)
    assert float(v) == 10.0
    assert int(Value(5.9, "kW")) == 5
    assert round(Value(5.49, "kW"), 1) == 5.5

    v.value = 20.0
    v.unit = "kW"
    assert v.value == 20.0
    assert v.to("W").value == pytest.approx(20_000.0)

    assert v == 20.0
    assert (v == object()) is False
    assert v > Value(10.0, "kW")
    assert v >= Value(20.0, "kW")
    assert v < Value(30.0, "kW")
    assert v <= Value(20.0, "kW")

    w = Value(5.0, "kW")
    assert (v + w).value == 25.0
    assert (w + v).value == 25.0
    assert (v - w).value == 15.0
    assert (Value(25.0, "kW") - v).value == 5.0
    assert (v * 2).value == 40.0
    assert (2 * v).value == 40.0
    assert (v / 2).value == 10.0
    assert (Value(40.0, "kW") / v).value == 2.0

    as_dict = v.to_dict()
    assert Value.from_dict(as_dict).value == v.value

    # Exercise ValueWithUnit branch with unit conversion failure fallback.
    vw = ValueWithUnit(value=3.0, units="m")
    fallback = Value(vw, unit="s")
    assert fallback.value == 3.0


def test_value_stateful_roundtrip_and_weighted_scalar_paths():
    v = Value(
        {"summer": 12.0, "winter": 6.0},
        "kW",
        weights={"summer": 0.25, "winter": 0.75},
    )

    assert v.value.tolist() == [12.0, 6.0]
    assert v.value[0] == pytest.approx(12.0)
    assert v.weighted_value == pytest.approx(7.5)
    assert float(v) == pytest.approx(7.5)
    assert int(v) == 7
    assert round(v, 1) == 7.5
    assert v.state_ids == ["summer", "winter"]
    assert v.state_values.tolist() == [12.0, 6.0]
    assert v.weights.tolist() == pytest.approx([0.25, 0.75])
    assert v.to_dict() == {
        "values": [12.0, 6.0],
        "state_ids": ["summer", "winter"],
        "weights": [0.25, 0.75],
        "unit": "kW",
    }
    assert Value.from_dict(v.to_dict()).weighted_value == pytest.approx(7.5)


# def test_series_unit_conversion():
#     series = pd.Series([
#         Value(10, "kW/degC"),
#         Value(20, "kW/K")
#     ])

#     converted = series.as_value.to("kW/K")

#     assert all(isinstance(v, Value) for v in converted)
#     assert converted[0].unit == "kW/K"
#     assert converted[1].unit == "kW/K"
#     assert round(float(converted[0]), 2) == 10.0  # conversion should preserve numeric value
#     assert round(float(converted[1]), 2) == 20.0

# def test_dataframe_column_conversion():
#     df = pd.DataFrame({
#         "stream": ["A", "B"],
#         "flow": [Value(10, "kW/degC"), Value(20, "kW/K")]
#     })

#     df["flow_kW_per_K"] = df["flow"].as_value.to("kW/K")

#     assert df["flow_kW_per_K"].iloc[0].unit == "kW/K"
#     assert df["flow_kW_per_K"].iloc[1].unit == "kW/K"
#     assert float(df["flow_kW_per_K"].iloc[0]) == 10.0
#     assert float(df["flow_kW_per_K"].iloc[1]) == 20.0

# def test_unit_preservation_after_conversion():
#     s = pd.Series([Value(100, "degC"), Value(373.15, "K")])
#     s_converted = s.as_value.to("K")

#     assert round(float(s_converted[0]), 2) == 373.15
#     assert round(float(s_converted[1]), 2) == 373.15
#     assert all(v.unit == "K" for v in s_converted)

# def test_invalid_unit_conversion_raises():
#     s = pd.Series([Value(100, "degC"), Value(373.15, "K")])
#     with pytest.raises(Exception):
#         s.as_value.to("m^3/h")  # incompatible unit
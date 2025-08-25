import pytest
import pandas as pd
from OpenPinch.classes import * 
from OpenPinch.lib import * 

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
    df = pd.DataFrame({
        "stream": ["A", "B"],
        "flow": [Value(10, "kW/degC"), Value(20, "kW/K")],
        "t_in": [Value(100, "degC"), Value(373.15, "K")],
        "t_out": [Value(120, "degC"), Value(383.15, "K")],
    })
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

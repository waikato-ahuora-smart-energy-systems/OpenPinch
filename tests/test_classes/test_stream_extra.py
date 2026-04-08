"""Additional branch coverage tests for Stream."""

import pytest

from OpenPinch.classes.stream import Stream
from OpenPinch.lib.enums import StreamType


def test_stream_pressure_and_enthalpy_property_getters():
    s = Stream(name="S", t_supply=120.0, t_target=80.0, heat_flow=100.0, htc=1.0)
    s.P_supply = 2.0
    s.P_target = 1.5
    s.h_supply = 900.0
    s.h_target = 700.0

    assert s.P_supply == 2.0
    assert s.P_target == 1.5
    assert s.h_supply == 900.0
    assert s.h_target == 700.0


def test_stream_equal_temperature_negative_heat_flow_sets_hot_profile():
    s = Stream(name="Iso", t_supply=100.0, t_target=100.0, heat_flow=-50.0, htc=1.0)
    assert s.type == StreamType.Hot.value
    assert s.t_target == pytest.approx(99.99, abs=1e-6)


def test_stream_update_attributes_uses_cp_when_heat_flow_non_numeric():
    s = Stream(name="CP", t_supply=120.0, t_target=80.0, heat_flow=100.0, htc=1.0)
    s._heat_flow = "unknown"
    s._CP = 3.0
    s._update_attributes()
    assert s.heat_flow == pytest.approx(120.0)

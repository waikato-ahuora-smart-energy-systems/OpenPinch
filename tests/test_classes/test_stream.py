"""Regression tests for the stream classes."""

import pytest

from OpenPinch.classes import *
from OpenPinch.lib import *


class DummyStream(Stream):
    """Concrete dummy subclass for testing Stream."""

    pass


@pytest.fixture
def hot_stream():
    """Return a representative hot stream used by this test module."""
    return DummyStream(
        name="Hot1",
        t_supply=300,
        t_target=200,
        dt_cont=10,
        heat_flow=5000,
        htc=2,
        price=50,
    )


@pytest.fixture
def cold_stream():
    """Return a representative cold stream used by this test module."""
    return DummyStream(
        name="Cold1",
        t_supply=100,
        t_target=250,
        dt_cont=5,
        heat_flow=4000,
        htc=1.5,
        price=30,
    )


# --- Basic Properties ---


def test_initialization(hot_stream):
    assert hot_stream.name == "Hot1"
    assert hot_stream.t_supply == 300
    assert hot_stream.t_target == 200
    assert hot_stream.dt_cont == 10
    assert hot_stream.heat_flow == 5000
    assert hot_stream.htc == 2
    assert hot_stream.price == 50
    assert hot_stream.active is True


def test_property_setters_getters(hot_stream):
    hot_stream.name = "NewName"
    hot_stream.price = 70
    hot_stream.dt_cont = 15
    assert hot_stream.name == "NewName"
    assert hot_stream.price == 70
    assert hot_stream.dt_cont == 15


def test_temperature_calculations(hot_stream, cold_stream):
    # Hot Stream
    assert hot_stream.t_min == 200
    assert hot_stream.t_max == 300
    assert hot_stream.t_min_star == 190
    assert hot_stream.t_max_star == 290
    assert hot_stream.type == StreamType.Hot.value

    # Cold Stream
    assert cold_stream.t_min == 100
    assert cold_stream.t_max == 250
    assert cold_stream.t_min_star == 105
    assert cold_stream.t_max_star == 255
    assert cold_stream.type == StreamType.Cold.value


def test_heat_capacity_flowrate_and_RCP(hot_stream):
    # CP = heat_flow / (t_max - t_min)
    expected_cp = 5000 / (300 - 200)
    assert pytest.approx(hot_stream.CP, rel=1e-6) == expected_cp

    # rCP = CP / htc
    expected_rcp = expected_cp / 2
    assert pytest.approx(hot_stream.rCP, rel=1e-6) == expected_rcp


def test_set_heat_flow(hot_stream):
    hot_stream.set_heat_flow(6000)
    assert hot_stream.heat_flow == 6000
    assert pytest.approx(hot_stream.CP) == 6000 / (300 - 200)
    assert pytest.approx(hot_stream.ut_cost) == (6000 / 1000) * 50


def test_active_flag(hot_stream):
    assert hot_stream.active is True
    hot_stream.active = False
    assert hot_stream.active is False


def test_manual_setters(hot_stream):
    hot_stream.t_min = 190
    hot_stream.t_max = 310
    hot_stream.t_min_star = 180
    hot_stream.t_max_star = 300
    hot_stream.CP = 100
    hot_stream.rCP = 50

    assert hot_stream.t_min == 190
    assert hot_stream.t_max == 310
    assert hot_stream.t_min_star == 180
    assert hot_stream.t_max_star == 300
    assert hot_stream.CP == 100
    assert hot_stream.rCP == 50


def test_zero_heat_flow_isothermal_stream_initialises_without_error():
    s = DummyStream(
        name="ZeroDuty",
        t_supply=100.0,
        t_target=100.0,
        heat_flow=0.0,
        dt_cont=5.0,
    )

    assert s.type == StreamType.Both.value
    assert s.t_min == 100.0
    assert s.t_max == 100.0
    assert s.t_min_star == 100.0
    assert s.t_max_star == 100.0
    assert s.CP == 0.0


# ===== Merged from test_stream_extra.py =====
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


def test_stream_invert_swaps_states_for_utility_stream():
    s = Stream(
        name="U",
        t_supply=180.0,
        t_target=140.0,
        P_supply=5.0,
        P_target=3.0,
        h_supply=1200.0,
        h_target=900.0,
        dt_cont=10.0,
        heat_flow=400.0,
        htc=2.0,
        is_process_stream=False,
    )

    s.invert()

    assert s.t_supply == 140.0
    assert s.t_target == 180.0
    assert s.P_supply == 3.0
    assert s.P_target == 5.0
    assert s.h_supply == 900.0
    assert s.h_target == 1200.0
    assert s.type == StreamType.Cold.value
    assert s.t_min == 140.0
    assert s.t_max == 180.0


def test_stream_invert_raises_for_process_stream():
    s = Stream(
        name="P",
        t_supply=180.0,
        t_target=140.0,
        heat_flow=400.0,
        htc=2.0,
        is_process_stream=True,
    )

    with pytest.raises(ValueError, match="Process streams cannot be inverted"):
        s.invert()

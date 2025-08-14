import pytest
from src.classes import *
from src.lib import * 

class DummyStream(Stream):
    """Concrete dummy subclass for testing Stream."""
    pass

@pytest.fixture
def hot_stream():
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

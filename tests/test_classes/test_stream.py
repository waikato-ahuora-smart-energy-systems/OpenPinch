"""Regression tests for the stream classes."""

import pytest

from OpenPinch.classes import *
from OpenPinch.classes._stream_value_view import StreamValueView as ExtractedStreamValueView
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream import StreamValueView
from OpenPinch.lib import *
from OpenPinch.lib.enums import ST
from OpenPinch.lib.schemas.common import StatefulValueWithUnit


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
    assert hot_stream.dt_cont_act == 10
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
    assert hot_stream.dt_cont_act == 15


def test_temperature_calculations(hot_stream, cold_stream):
    # Hot Stream
    assert hot_stream.t_min == 200
    assert hot_stream.t_max == 300
    assert hot_stream.t_min_star == 190
    assert hot_stream.t_max_star == 290
    assert hot_stream.type == ST.Hot.value

    # Cold Stream
    assert cold_stream.t_min == 100
    assert cold_stream.t_max == 250
    assert cold_stream.t_min_star == 105
    assert cold_stream.t_max_star == 255
    assert cold_stream.type == ST.Cold.value


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


def test_dt_cont_act_updates_shifted_temperatures_without_mutating_base_dt_cont():
    s = DummyStream(
        name="Shifted",
        t_supply=300.0,
        t_target=200.0,
        heat_flow=5000.0,
        dt_cont=10.0,
        dt_cont_act=15.0,
        htc=2.0,
    )

    assert s.dt_cont == 10.0
    assert s.dt_cont_act == 15.0
    assert s.t_min_star == 185.0
    assert s.t_max_star == 285.0

    s.dt_cont_act = 20.0
    assert s.dt_cont == 10.0
    assert s.dt_cont_act == 20.0
    assert s.t_min_star == 180.0
    assert s.t_max_star == 280.0


def test_zero_heat_flow_isothermal_stream_initialises_without_error():
    s = DummyStream(
        name="ZeroDuty",
        t_supply=100.0,
        t_target=100.0,
        heat_flow=0.0,
        dt_cont=5.0,
    )

    assert s.type == ST.Both.value
    assert s.t_min == 100.0
    assert s.t_max == 100.0
    assert s.t_min_star == 100.0
    assert s.t_max_star == 100.0
    assert s.CP == 0.0


def test_stream_stateful_value_view_defaults_to_state_zero_and_exposes_unit():
    s = Stream(
        name="Stateful",
        t_supply=StatefulValueWithUnit(
            values=[300.0, 280.0],
            unit="degC",
            state_ids=["0", "1"],
            weights=[0.6, 0.4],
        ),
        t_target=StatefulValueWithUnit(
            values=[200.0, 180.0],
            unit="degC",
            state_ids=["0", "1"],
            weights=[0.6, 0.4],
        ),
        heat_flow=StatefulValueWithUnit(
            values=[5000.0, 4000.0],
            unit="kW",
            state_ids=["0", "1"],
            weights=[0.6, 0.4],
        ),
        dt_cont=10.0,
        htc=2.0,
    )

    assert s.supply_temperature == 300.0
    assert s.supply_temperature.unit == "°C"
    assert s.supply_temperature.units == "°C"
    assert s.supply_temperature[0].value == pytest.approx(300.0)
    assert s.supply_temperature[1].value == pytest.approx(280.0)
    assert s.supply_temperature[1].unit == "°C"


def test_stream_value_view_import_from_stream_module_remains_supported():
    assert StreamValueView is ExtractedStreamValueView


def test_stream_scalar_value_view_allows_any_state_lookup_key():
    s = DummyStream(
        name="Scalar",
        t_supply=300.0,
        t_target=200.0,
        heat_flow=5000.0,
        dt_cont=10.0,
        htc=2.0,
    )

    assert float(s.supply_temperature) == pytest.approx(300.0)
    assert s.supply_temperature.value == pytest.approx(300.0)
    assert s.supply_temperature.unit == "°C"
    assert s.supply_temperature[0].value == pytest.approx(300.0)
    assert s.supply_temperature[1].value == pytest.approx(300.0)
    assert s.supply_temperature["summer"].value == pytest.approx(300.0)
    assert s.supply_temperature["summer"].unit == "°C"
    assert s.supply_temperature["summer"].state_ids is None
    assert isinstance(s.supply_temperature, StreamValueView)


def test_stream_stateful_derivatives_broadcast_scalar_inputs():
    s = Stream(
        name="Derived",
        t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    assert s.t_min.state_ids == ["0", "1"]
    assert s.t_min[0].value == pytest.approx(200.0)
    assert s.t_min[1].value == pytest.approx(180.0)
    assert s.t_max[0].value == pytest.approx(300.0)
    assert s.t_max[1].value == pytest.approx(280.0)
    assert s.t_min_star[0].value == pytest.approx(190.0)
    assert s.t_min_star[1].value == pytest.approx(170.0)
    assert s.CP[0].value == pytest.approx(50.0)
    assert s.CP[1].value == pytest.approx(40.0)
    assert s.rCP[0].value == pytest.approx(25.0)
    assert s.rCP[1].value == pytest.approx(20.0)


def test_stream_stateful_value_view_remains_strict_for_unknown_state_id():
    s = Stream(
        name="Stateful",
        t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
        t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
        heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
        dt_cont=10.0,
        htc=2.0,
    )

    with pytest.raises(KeyError, match="Unknown state_id 'summer'"):
        _ = s.supply_temperature["summer"]


def test_stream_rejects_mismatched_state_alignment():
    with pytest.raises(ValueError, match="state_ids for heat_flow must align with t_supply"):
        Stream(
            name="Misaligned",
            t_supply={"values": [300.0, 280.0], "state_ids": ["0", "1"], "unit": "degC"},
            t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
            heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "2"], "unit": "kW"},
            dt_cont=10.0,
            htc=2.0,
        )


def test_stream_rejects_mixed_hot_and_cold_state_directions():
    with pytest.raises(ValueError, match="Stream states must classify consistently"):
        Stream(
            name="Mixed",
            t_supply={"values": [300.0, 120.0], "state_ids": ["0", "1"], "unit": "degC"},
            t_target={"values": [200.0, 180.0], "state_ids": ["0", "1"], "unit": "degC"},
            heat_flow={"values": [5000.0, 4000.0], "state_ids": ["0", "1"], "unit": "kW"},
            dt_cont=10.0,
            htc=2.0,
        )


# ===== Merged from test_stream_extra.py =====
"""Additional branch coverage tests for Stream."""


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
    assert s.type == ST.Hot.value
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
    assert s.type == ST.Cold.value
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


def test_descriptive_aliases_update_canonical_stream_fields():
    s = Stream(name="Alias", t_supply=150.0, t_target=90.0, heat_flow=240.0, htc=2.0)

    s.supply_temperature = 160.0
    s.target_temperature = 100.0
    s.supply_pressure = 3.5
    s.target_pressure = 2.5
    s.supply_enthalpy = 1100.0
    s.target_enthalpy = 850.0
    s.delta_t_contribution = 8.0
    s.effective_delta_t_contribution = 12.0
    s.heat_duty = 300.0
    s.heat_transfer_coefficient = 4.0
    s.process_stream = False
    s.is_active = False

    assert s.t_supply == 160.0
    assert s.t_target == 100.0
    assert s.P_supply == 3.5
    assert s.P_target == 2.5
    assert s.h_supply == 1100.0
    assert s.h_target == 850.0
    assert s.dt_cont == 8.0
    assert s.dt_cont_act == 12.0
    assert s.heat_flow == 300.0
    assert s.htc == 4.0
    assert s.is_process_stream is False
    assert s.active is False
    assert s.supply_temperature == 160.0
    assert s.target_temperature == 100.0
    assert s.minimum_temperature == 100.0
    assert s.maximum_temperature == 160.0
    assert s.shifted_minimum_temperature == 88.0
    assert s.shifted_maximum_temperature == 148.0
    assert s.heat_transfer_resistance == pytest.approx(0.25)
    assert s.heat_capacity_flow_rate == pytest.approx(5.0)


def test_descriptive_aliases_cover_abbreviation_backed_fields():
    s = Stream(name="Readable", t_supply=120.0, t_target=80.0, heat_flow=200.0, htc=2.0)

    s.stream_type = ST.Both.value
    s.minimum_temperature = 70.0
    s.maximum_temperature = 130.0
    s.shifted_minimum_temperature = 65.0
    s.shifted_maximum_temperature = 125.0
    s.heat_transfer_resistance = 0.75
    s.utility_cost = 42.0
    s.heat_capacity_flow_rate = 9.0
    s.resistance_capacity_product = 6.75

    assert s.type == ST.Both.value
    assert s.t_min == 70.0
    assert s.t_max == 130.0
    assert s.t_min_star == 65.0
    assert s.t_max_star == 125.0
    assert s.htr == 0.75
    assert s.ut_cost == 42.0
    assert s.CP == 9.0
    assert s.rCP == 6.75

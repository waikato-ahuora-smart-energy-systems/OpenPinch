"""Property and edge-branch tests for class-level data models."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import pytest
from OpenPinch.classes.energy_target import EnergyTarget
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.value import Value
from OpenPinch.classes.zone import Zone
from OpenPinch.classes import stream_collection as sc_mod
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.schema import ValueWithUnit


def test_stream_property_roundtrip_and_mutation_paths():
    s = Stream(
        name="S1",
        t_supply=150.0,
        t_target=90.0,
        heat_flow=120.0,
        dt_cont=5.0,
        htc=2.0,
        price=10.0,
    )

    assert s.is_process_stream is True
    s.is_process_stream = False
    assert s.is_process_stream is False

    s.type = "Manual"
    assert s.type == "Manual"

    s.t_supply = 160.0
    s.t_target = 95.0
    s.P_supply = 200.0
    s.P_target = 180.0
    s.h_supply = 2_500.0
    s.h_target = 2_000.0
    s.dt_cont = 4.0
    s.heat_flow = 100.0
    s.htc = 3.0
    s.htr = 0.25
    s.price = 12.0
    s.ut_cost = 1_000.0
    s.CP = 2.5
    s.rCP = 0.4
    s.active = False
    s.t_min = 50.0
    s.t_max = 160.0
    s.t_min_star = 48.0
    s.t_max_star = 162.0

    assert s.htr == 0.25
    assert s.price == 12.0
    assert s.ut_cost == 1_000.0
    assert s.CP == 2.5
    assert s.rCP == 0.4
    assert s.active is False
    assert s.t_min == 50.0
    assert s.t_max == 160.0
    assert s.t_min_star == 48.0
    assert s.t_max_star == 162.0


def test_stream_collection_edge_paths_and_pickle_state(tmp_path):
    st1 = Stream(name="A", t_supply=120.0, t_target=80.0, heat_flow=20.0)
    st2 = Stream(name="B", t_supply=110.0, t_target=70.0, heat_flow=10.0)
    st3 = Stream(name="C", t_supply=60.0, t_target=90.0, heat_flow=5.0)

    sc = StreamCollection()
    sc.add(st1)
    sc.add_many([st2, st3], keys=["B", "C"])
    assert sc_mod._sort_by_attrs(("name",), st1) == ("A",)
    assert sc_mod._is_picklable(1) is True
    assert sc_mod._is_picklable(lambda x: x) is False

    with pytest.raises(ValueError, match="Length of streams and keys must match"):
        sc.add_many([st1], keys=["x", "y"])

    sc.set_sort_key(["t_supply", "name"])
    _ = list(sc)
    sc.set_sort_key(lambda stream: stream.name, reverse=True)
    _ = list(sc)

    assert sc.__add__(123) is NotImplemented
    with pytest.raises(ValueError, match="Stream not found"):
        sc.get_index(Stream(name="missing"))
    with pytest.raises(IndexError):
        _ = sc[999]
    with pytest.raises(TypeError):
        _ = sc[1.2]
    with pytest.raises(KeyError):
        sc.remove("unknown")

    state = sc.__getstate__()
    assert state["_sort_spec"] == ("attr", "t_supply")
    sc2 = StreamCollection()
    sc2.__setstate__(state)

    sc3 = StreamCollection()
    sc3.replace({"A": st1, "B": st2})
    assert sc3.get_hot_streams() is not None
    assert sc3.get_cold_streams() is not None

    csv_path = sc3.export_to_csv(filename="stream_collection_export_test")
    assert isinstance(csv_path, Path)
    assert csv_path.exists()

    assert (sc == 1) is False


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


def test_energy_target_and_zone_property_branches():
    t = EnergyTarget(name="T0")
    t.name = "T1"
    t.identifier = "DI"
    t.config = Configuration()
    t.parent_zone = "parent"
    with pytest.raises(TypeError):
        t.active = True
    t._active = Value(1)
    assert t.active == 1

    hot = StreamCollection()
    cold = StreamCollection()
    hu = Stream(
        name="HU",
        t_supply=180.0,
        t_target=170.0,
        heat_flow=20.0,
        is_process_stream=False,
    )
    cu = Stream(
        name="CU", t_supply=20.0, t_target=30.0, heat_flow=15.0, is_process_stream=False
    )
    hu.ut_cost = 8.0
    cu.ut_cost = 6.0
    hot.add(hu)
    cold.add(cu)

    t.hot_utilities = hot
    t.cold_utilities = cold
    t.graphs = {}
    t.area = 1.0
    t.capital_cost = 2.0
    t.total_cost = 3.0
    t.utility_cost = 4.0
    t.work_target = 5.0
    t.turbine_efficiency_target = 0.5
    t.exergy_sources = 6.0
    t.exergy_sinks = 7.0
    t.ETE = 0.8
    t.exergy_req_min = 8.0
    t.exergy_des_min = 9.0
    t.hot_utility_target = 10.0
    t.cold_utility_target = 11.0
    t.heat_recovery_target = 12.0
    t.degree_of_int = 0.75
    t.num_units = 2
    t.add_graph("g", {"x": 1})
    assert t.graphs["g"] == {"x": 1}
    assert t.calc_utility_cost() == pytest.approx(14.0)

    t.target_values = {"hot_pinch": 120.0, "cold_pinch": 120.0}
    assert t.hot_pinch == 120.0
    payload_same = t.serialize_json(isTotal=True)
    assert payload_same["temp_pinch"] == {"cold_temp": 120.0}

    t.hot_pinch = None
    t.cold_pinch = 95.0
    payload_cold = t.serialize_json(isTotal=False)
    assert payload_cold["temp_pinch"] == {"cold_temp": 95.0}

    t.hot_pinch = 135.0
    t.cold_pinch = None
    payload_hot = t.serialize_json(isTotal=False)
    assert payload_hot["temp_pinch"] == {"hot_temp": 135.0}

    t.config.DO_TURBINE_WORK = True
    t.config.DO_AREA_TARGETING = True
    t.config.DO_EXERGY_TARGETING = True
    payload_all = t.serialize_json()
    assert payload_all["work_target"] == 5.0
    assert payload_all["area"] == 1.0
    assert payload_all["exergy_sources"] == 6.0

    z = Zone(name="Root")
    z.name = "Root2"
    z.identifier = "P"
    z.config = Configuration()
    z.parent_zone = None
    z.active = True
    z.hot_streams = hot
    z.cold_streams = cold
    z.net_hot_streams = StreamCollection()
    z.net_cold_streams = StreamCollection()
    z.hot_utilities = hot
    z.cold_utilities = cold
    z.graphs = {}
    z.add_graph("k", {"v": 1})
    assert z.graphs["k"] == {"v": 1}
    assert z.process_streams is not None
    assert z.net_process_streams is not None
    assert z.utility_streams is not None
    assert z.all_streams is not None
    assert z.all_net_streams is not None

    child1 = Zone(name="Child")
    child2 = Zone(name="Child")
    child2.hot_streams = StreamCollection()
    child2.hot_streams.add(
        Stream(name="x", t_supply=120.0, t_target=100.0, heat_flow=5.0)
    )
    z.add_zone(child1)
    z.add_zone(child2)
    assert any(name.startswith("Child") for name in z.subzones.keys())

    z.add_target(t)
    z.add_targets([EnergyTarget(name="T2")])
    z.add_target_from_results(target_id="DI", results={"hot_pinch": 100.0})
    assert "Root2/DI" in z.targets

    assert z.get_subzone(next(iter(z.subzones.keys()))) is not None
    with pytest.raises(ValueError, match="Subzone"):
        z.get_subzone("missing/path")

    assert z.calc_utility_cost() == pytest.approx(14.0)

    parent = Zone(name="Parent")
    child = Zone(name="Child")
    child.hot_streams.add(
        Stream(name="H1", t_supply=120.0, t_target=80.0, heat_flow=10.0)
    )
    child.cold_streams.add(
        Stream(name="C1", t_supply=30.0, t_target=60.0, heat_flow=8.0)
    )
    parent.add_zone(child)
    parent.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=False, is_n_zone_depth=True, is_new_stream_collection=True
    )
    assert len(parent.hot_streams) >= 1
    assert len(parent.cold_streams) >= 1


# ===== Merged from test_energy_target_extra.py =====
"""Additional branch coverage tests for EnergyTarget properties."""


def test_energy_target_identifier_parent_active_and_cost_properties():
    target = EnergyTarget(name="T", identifier="id", parent_zone="Z")
    assert target.identifier == "id"
    assert target.parent_zone == "Z"

    target._active = True
    assert target.active is True
    target._active = Value(0.0)
    assert target.active == 0.0

    target.capital_cost = 123.0
    assert target.capital_cost == 123.0

    target.utility_heat_recovery_target = 44.0
    assert target.utility_heat_recovery_target == 44.0

    target.target_values = {"hot_utility_target": 10.0}
    assert target.target_values == {"hot_utility_target": 10.0}


def test_energy_target_capital_cost_descriptor_access():
    target = EnergyTarget(name="T", identifier="id", parent_zone="Z")
    EnergyTarget.capital_cost.fset(target, 321.0)
    assert EnergyTarget.capital_cost.fget(target) == 321.0

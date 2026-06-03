"""Property and edge-branch tests for class-level data models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from OpenPinch.classes import stream_collection as sc_mod
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.classes.zone import Zone
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import ProblemTableLabel as PT
from OpenPinch.lib.schemas.targets import BaseTargetModel, DirectIntegrationTarget


def _dummy_problem_table():
    return ProblemTable({PT.T: [0.0]})


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

    s.t_supply = 160.0
    s.t_target = 95.0
    s.p_supply = 200.0
    s.p_target = 180.0
    s.h_supply = 2_500.0
    s.h_target = 2_000.0
    s.dt_cont = 4.0
    s.dt_cont_act = 6.0
    s.heat_flow = 100.0
    s.htc = 3.0
    s.price = 12.0
    s.active = False

    assert s.type == "Hot"
    assert s.htr == pytest.approx(1.0 / 3.0)
    assert s.price == 12.0
    assert s.ut_cost == pytest.approx(1.2)
    assert s.CP == pytest.approx(100.0 / (160.0 - 95.0))
    assert s.rCP == pytest.approx((100.0 / (160.0 - 95.0)) / 3.0)
    assert s.active is False
    assert s.dt_cont == 4.0
    assert s.dt_cont_act == 6.0
    assert s.t_min == 95.0
    assert s.t_max == 160.0
    assert s.t_min_star == 89.0
    assert s.t_max_star == 154.0


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
    with pytest.warns(Warning):
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


def test_target_model_and_zone_property_branches():
    t = DirectIntegrationTarget(
        zone_name="T0",
        type="DI",
        pt=_dummy_problem_table(),
        pt_real=_dummy_problem_table(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
    )
    t.config = Configuration()
    t.parent_zone = "parent"
    t.active = True
    assert t.active is True

    hot = StreamCollection()
    cold = StreamCollection()
    hu = Stream(
        name="HU",
        t_supply=180.0,
        t_target=170.0,
        heat_flow=20.0,
        price=400.0,
        is_process_stream=False,
    )
    cu = Stream(
        name="CU",
        t_supply=20.0,
        t_target=30.0,
        heat_flow=15.0,
        price=400.0,
        is_process_stream=False,
    )
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

    t.hot_pinch = 120.0
    t.cold_pinch = 120.0
    assert t.hot_pinch == 120.0
    payload_same = t.serialize_json(isTotal=True)
    assert payload_same["temp_pinch"]["cold_temp"] == 120.0
    assert payload_same["temp_pinch"]["hot_temp"] == 120.0

    t.hot_pinch = None
    t.cold_pinch = 95.0
    payload_cold = t.serialize_json(isTotal=False)
    assert payload_cold["temp_pinch"]["cold_temp"] == 95.0

    t.hot_pinch = 135.0
    t.cold_pinch = None
    payload_hot = t.serialize_json(isTotal=False)
    assert payload_hot["temp_pinch"]["hot_temp"] == 135.0

    t.config.DO_TURBINE_WORK = True
    t.config.DO_AREA_TARGETING = True
    t.config.DO_EXERGY_TARGETING = True
    payload_all = t.serialize_json()
    assert payload_all["work_target"] == 5.0
    assert payload_all["area"] == 1.0
    assert payload_all["exergy_sources"] == 6.0

    z = Zone(name="Root", type="P")
    z.name = "Root2"
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
    z.add_targets(
        [
            DirectIntegrationTarget(
                zone_name="T2",
                type="DI",
                pt=_dummy_problem_table(),
                pt_real=_dummy_problem_table(),
                hot_utilities=StreamCollection(),
                cold_utilities=StreamCollection(),
                hot_utility_target=0.0,
                cold_utility_target=0.0,
                heat_recovery_target=0.0,
            )
        ]
    )
    assert "DI" in z.targets

    assert z.get_subzone(next(iter(z.subzones.keys()))) is not None
    with pytest.warns(Warning):
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


def test_target_model_requires_zone_name_and_identifier():
    with pytest.raises(ValidationError):
        BaseTargetModel()

    with pytest.raises(ValidationError):
        BaseTargetModel(zone_name="T")

    with pytest.raises(ValueError, match="zone_name is required"):
        BaseTargetModel(zone_name="", type="DI")

    with pytest.raises(ValueError, match="type is required"):
        BaseTargetModel(zone_name="T", type="")


def test_target_model_identifier_parent_active_and_cost_properties():
    target = BaseTargetModel(zone_name="T", type="id", parent_zone="Z")
    assert target.type == "id"
    assert target.parent_zone == "Z"

    target.active = True
    assert target.active is True

    direct_target = DirectIntegrationTarget(
        zone_name="T",
        type="DI",
        parent_zone="Z",
        pt=_dummy_problem_table(),
        pt_real=_dummy_problem_table(),
        hot_utilities=StreamCollection(),
        cold_utilities=StreamCollection(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
    )
    direct_target.capital_cost = 123.0
    assert direct_target.capital_cost == 123.0

    direct_target.utility_heat_recovery_target = 44.0
    assert direct_target.utility_heat_recovery_target == 44.0
    direct_target.hot_utility_target = 10.0
    assert direct_target.hot_utility_target == 10.0

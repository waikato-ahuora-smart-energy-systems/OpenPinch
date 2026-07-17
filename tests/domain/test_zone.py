"""Regression tests for the zone classes."""

import pytest

from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import ProblemTableLabel as ProblemTableLabel
from OpenPinch.domain.enums import (
    TargetType,
    ZoneType,
)
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.targets import DirectIntegrationTarget
from OpenPinch.domain.zone import Zone
from OpenPinch.presentation.reporting.results import serialize_target


def _dummy_problem_table():
    return ProblemTable({ProblemTableLabel.T: [0.0]})


@pytest.fixture
def dummy_zone():
    """Return dummy zone data used by this test module."""
    return Zone(name="Z1")


@pytest.fixture
def dummy_tar():
    """Return dummy tar data used by this test module."""
    return DirectIntegrationTarget(
        zone_name="Z1",
        type="DI",
        pt=_dummy_problem_table(),
        pt_real=_dummy_problem_table(),
        hot_utility_target=0.0,
        cold_utility_target=0.0,
        heat_recovery_target=0.0,
    )


@pytest.fixture
def sample_streams():
    """Return sample streams data used by this test module."""
    return [
        Stream(
            name="Hot1", supply_temperature=400, target_temperature=200, heat_flow=1
        ),
        Stream(
            name="Cold1", supply_temperature=100, target_temperature=300, heat_flow=1
        ),
    ]


@pytest.fixture
def sample_utilities():
    """Return sample utilities data used by this test module."""
    u1 = Stream(
        name="Steam",
        supply_temperature=450,
        target_temperature=250,
        heat_flow=250,
        price=400,
        is_process_stream=False,
    )
    u2 = Stream(
        name="CoolingWater",
        supply_temperature=25,
        target_temperature=40,
        heat_flow=125,
        price=400,
        is_process_stream=False,
    )
    return [u1, u2]


# === Basic Instantiation ===


def test_dummy_zone_instantiation(dummy_zone: Zone):
    assert dummy_zone.name == "Z1"
    assert isinstance(dummy_zone.hot_streams, StreamCollection)
    assert isinstance(dummy_zone.cold_streams, StreamCollection)
    assert dummy_zone.dt_cont_multiplier == 1.0


# === Property Setters and Getters ===


def test_property_setters_getters(dummy_zone: Zone):
    dummy_zone.area = 123.45
    assert dummy_zone.area == 123.45

    dummy_zone.capital_cost = 10000
    assert dummy_zone.capital_cost == 10000

    dummy_zone.work_target = 500
    assert dummy_zone.work_target == 500

    dummy_zone.ETE = 0.75
    assert dummy_zone.ETE == 0.75

    with pytest.warns(UserWarning, match="empty stream collection"):
        dummy_zone.dt_cont_multiplier = 1.5
    assert dummy_zone.dt_cont_multiplier == 1.5


# === Adding Utilities ===


def test_add_single_utility(dummy_zone: Zone, sample_utilities):
    dummy_zone.hot_utilities.add(sample_utilities[0])
    assert "Steam" in dummy_zone.hot_utilities


def test_add_multiple_utilities(dummy_zone: Zone, sample_utilities):
    dummy_zone.cold_utilities.add_many(sample_utilities)
    assert "Steam" in dummy_zone.cold_utilities
    assert "CoolingWater" in dummy_zone.cold_utilities


# === Adding Zones ===


def test_add_zone(dummy_zone: Zone):
    z = Zone(name="SubZone")
    dummy_zone.add_zone(z)
    assert "SubZone" in dummy_zone.subzones


def test_add_zone_conflict_naming(dummy_zone: Zone):
    z1 = Zone(name="ZoneX")
    z2 = Zone(name="ZoneX")
    dummy_zone.add_zone(z1)
    dummy_zone.add_zone(z2)
    assert "ZoneX" in dummy_zone.subzones
    assert "ZoneX_1" not in dummy_zone.subzones


def test_add_zone_invalid(dummy_zone: Zone):
    with pytest.raises(ValueError):
        dummy_zone.add_zone(object())


def test_get_subzone_prefers_direct_child_when_child_matches_parent_name():
    root = Zone(name="Plant")
    child = Zone(name="Plant", parent_zone=root)
    operation = Zone(name="O1", parent_zone=child)
    child.add_zone(operation)
    root.add_zone(child)

    assert root.get_subzone() is root
    assert root.get_subzone("Plant") is child
    assert root.get_subzone("Plant/O1") is operation
    assert root.get_subzone("Plant/Plant/O1") is operation


# === Site and Process Zones Filtering ===


def test_get_process_zones(dummy_zone: Zone):
    dummy_zone.add_zone(Zone(name="Area1"))
    dummy_zone.add_zone(Zone(name=TargetType.DI.value), sub=False)  # site-level
    assert "Area1" in dummy_zone.subzones
    assert TargetType.DI.value not in dummy_zone.subzones


def test_get_site_zones(dummy_zone: Zone):
    dummy_zone.add_zone(Zone(name="Area1"))
    dummy_zone.add_zone(Zone(name=TargetType.DI.value), sub=False)
    assert TargetType.DI.value in dummy_zone.targets
    assert "Area1" not in dummy_zone.targets


# === Utility Cost Calculation ===


def test_calc_zonal_utility_cost(dummy_tar: DirectIntegrationTarget, sample_utilities):
    for u in sample_utilities:
        dummy_tar.hot_utilities.add(u)
    dummy_tar.calc_utility_cost()
    assert dummy_tar.utility_cost == 150  # 100 + 50


# === Process Streams and Utility Streams Properties ===


def test_stream_collections(dummy_zone: Zone, sample_streams, sample_utilities):
    dummy_zone.hot_streams.add(sample_streams[0])
    dummy_zone.cold_streams.add(sample_streams[1])
    dummy_zone.hot_utilities.add(sample_utilities[0])
    dummy_zone.cold_utilities.add(sample_utilities[1])

    assert len(dummy_zone.process_streams) == 2
    assert len(dummy_zone.utility_streams) == 2
    assert len(dummy_zone.all_streams) == 4


# === Gather Streams from Subzones ===


def test_get_hot_and_cold_streams_from_subzones(dummy_zone, sample_streams):
    z = Zone(name="Sub1")
    z.hot_streams.add(sample_streams[0])
    z.cold_streams.add(sample_streams[1])
    dummy_zone.add_zone(z)

    dummy_zone.import_hot_and_cold_streams_from_sub_zones()
    assert "Sub1.Hot1" in dummy_zone.hot_streams
    assert "Sub1.Cold1" in dummy_zone.cold_streams


# === New Tests for Zone ===


def test_add_graph(dummy_tar):
    dummy_tar.add_graph("test_graph", {"some": "result"})
    assert "test_graph" in dummy_tar.graphs
    assert dummy_tar.graphs["test_graph"] == {"some": "result"}


def test_zone_is_equal(dummy_zone):
    z1 = Zone(name="Z_same")
    z2 = Zone(name="Z_same")

    # Should be equal since both are fresh and empty
    assert dummy_zone._zone_is_equal(z1, z2)

    # Add a stream to one zone -> now not equal
    z1.hot_streams.add(
        Stream(
            name="H1",
            supply_temperature=100,
            target_temperature=50,
            heat_flow=1,
            is_process_stream=False,
        )
    )
    assert not dummy_zone._zone_is_equal(z1, z2)


def test_serialize_json_basic(dummy_tar):
    dummy_tar.hot_utility_target = 100
    dummy_tar.cold_utility_target = 50
    dummy_tar.heat_recovery_target = 30
    dummy_tar.utility_cost = 500

    json_data = serialize_target(dummy_tar)

    assert json_data["name"] == "Z1/DI"
    assert json_data["Qh"] == {"value": 100.0, "unit": "kW"}
    assert json_data["Qc"] == {"value": 50.0, "unit": "kW"}
    assert json_data["Qr"] == {"value": 30.0, "unit": "kW"}
    assert json_data["utility_cost"] == {"value": 500.0, "unit": "$/h"}
    assert "pinch_temp" in json_data


def test_serialize_json_with_area_and_exergy(dummy_tar):
    dummy_tar.area = 120
    dummy_tar.num_units = 3
    dummy_tar.capital_cost = 2000
    dummy_tar.total_cost = 5000
    dummy_tar.exergy_sources = 800
    dummy_tar.exergy_sinks = 700
    dummy_tar.ETE = 0.85
    dummy_tar.exergy_req_min = 100
    dummy_tar.exergy_des_min = 90

    json_data = serialize_target(dummy_tar)

    assert json_data["area"] == {"value": 120.0, "unit": "m^2"}
    assert json_data["num_units"] == 3
    assert json_data["capital_cost"] == {"value": 2000.0, "unit": "$"}
    assert json_data["total_cost"] == {"value": 5000.0, "unit": "$/y"}
    assert json_data["exergy_sources"] == {"value": 800.0, "unit": "kW"}
    assert json_data["exergy_sinks"] == {"value": 700.0, "unit": "kW"}
    assert json_data["ETE"] == {"value": 85.0, "unit": "%"}
    assert json_data["exergy_req_min"] == {"value": 100.0, "unit": "kW"}
    assert json_data["exergy_des_min"] == {"value": 90.0, "unit": "kW"}


# ===== Merged from test_zone_extra.py =====
"""Additional branch coverage tests for Zone."""


def test_zone_active_property_and_duplicate_suffix_increment():
    root = Zone("Root")
    assert root.active is True

    first = Zone("Child")
    first.hot_streams.add(
        Stream(
            name="H1", supply_temperature=120.0, target_temperature=80.0, heat_flow=10.0
        )
    )
    root.add_zone(first, sub=True)

    already_taken = Zone("Child_1")
    already_taken.hot_streams.add(
        Stream(
            name="H9", supply_temperature=110.0, target_temperature=70.0, heat_flow=9.0
        )
    )
    root.add_zone(already_taken, sub=True)

    duplicate = Zone("Child")
    duplicate.hot_streams.add(
        Stream(
            name="H2", supply_temperature=130.0, target_temperature=90.0, heat_flow=8.0
        )
    )
    root.add_zone(duplicate, sub=True)

    assert "Child_2" in root.subzones


def test_zone_property_setters_and_stream_container_type_guard():
    root = Zone("Root")
    child = Zone("Child")
    config = Configuration({"THERMAL_DT_CONT": 8})

    root.type = ZoneType.S.value
    root.config = config
    child.parent_zone = root

    assert root.type == ZoneType.S.value
    assert root.config is config
    assert child.parent_zone is root
    with pytest.raises(TypeError, match="StreamCollection"):
        root._attach_stream_collection(object())


def test_zone_period_context_none_list_weights_and_target_defaults(dummy_tar):
    root = Zone("Root")
    child = Zone("Child", parent_zone=root)
    root.add_zone(child)

    root.set_period_context(None, None, None)
    assert root.period_ids is None
    assert child.period_ids is None

    root.set_period_context(["base", "peak"], [0.25, 0.75], 2)
    assert root.period_ids == {"base": 0, "peak": 1}
    assert root.weights.tolist() == [0.25, 0.75]
    assert child.period_ids == {"base": 0, "peak": 1}

    root.set_period_context(["base", "peak"], (0.4, 0.6), 2)
    assert root.weights.tolist() == [0.4, 0.6]

    root.add_targets()
    assert root.targets == {}
    root.add_targets([dummy_tar])
    assert dummy_tar.type in root.targets


def test_zone_subzone_and_target_zone_resolution_edges():
    root = Zone("Root")
    area = Zone("Area", parent_zone=root)
    unit = Zone("Unit", parent_zone=area)
    area.add_zone(unit)
    root.add_zone(area)

    assert root.get_subzone("Root") is root
    assert root.get_subzone("Root/Area/Unit") is unit
    with pytest.warns(UserWarning, match="not found"):
        assert root.get_subzone("Missing") is None

    assert root.get_target_zone(None) is root
    assert root.get_target_zone("Root") is root
    assert root.get_target_zone("Root/Area") is area


def test_zone_imports_net_streams_and_locks_dt_multiplier():
    root = Zone("Root")
    area = Zone("Area", parent_zone=root)
    area.net_hot_streams.add(
        Stream(
            name="NH",
            supply_temperature=200.0,
            target_temperature=120.0,
            heat_flow=10.0,
        )
    )
    area.net_cold_streams.add(
        Stream(
            name="NC", supply_temperature=50.0, target_temperature=100.0, heat_flow=8.0
        )
    )
    root.add_zone(area)

    root.import_hot_and_cold_streams_from_sub_zones(get_net_streams=True)

    assert "Area.NH" in root.net_hot_streams
    assert "Area.NC" in root.net_cold_streams
    with pytest.warns(UserWarning, match="empty stream collection"):
        Zone("Empty").lock_dt_cont_multiplier()

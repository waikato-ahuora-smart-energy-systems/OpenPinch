import pytest

from OpenPinch.classes import *
from OpenPinch.lib import *


@pytest.fixture
def dummy_zone():
    return Zone(name="Z1")


@pytest.fixture
def dummy_tar():
    return EnergyTarget("Z1")


@pytest.fixture
def sample_streams():
    return [
        Stream(name="Hot1", t_supply=400, t_target=200, heat_flow=1),
        Stream(name="Cold1", t_supply=100, t_target=300, heat_flow=1),
    ]


@pytest.fixture
def sample_utilities():
    u1 = Stream(name="Steam", t_supply=450, t_target=250, is_process_stream=False)
    u1.ut_cost = 100
    u2 = Stream(name="CoolingWater", t_supply=25, t_target=40, is_process_stream=False)
    u2.ut_cost = 50
    return [u1, u2]


# === Basic Instantiation ===


def test_dummy_zone_instantiation(dummy_zone: Zone):
    assert dummy_zone.name == "Z1"
    assert isinstance(dummy_zone.hot_streams, StreamCollection)
    assert isinstance(dummy_zone.cold_streams, StreamCollection)


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


def test_calc_zonal_utility_cost(dummy_tar: EnergyTarget, sample_utilities):
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
            name="H1", t_supply=100, t_target=50, heat_flow=1, is_process_stream=False
        )
    )
    assert not dummy_zone._zone_is_equal(z1, z2)


def test_serialize_json_basic(dummy_tar):
    dummy_tar.hot_utility_target = 100
    dummy_tar.cold_utility_target = 50
    dummy_tar.heat_recovery_target = 30
    dummy_tar.utility_cost = 500

    json_data = dummy_tar.serialize_json()

    assert json_data["name"] == "Z1"
    assert json_data["Qh"] == 100
    assert json_data["Qc"] == 50
    assert json_data["Qr"] == 30
    assert json_data["utility_cost"] == 500
    assert "temp_pinch" in json_data


def test_serialize_json_with_area_and_exergy(dummy_tar):
    dummy_tar.config.AREA_BUTTON = True
    dummy_tar.config.EXERGY_BUTTON = True

    dummy_tar.area = 120
    dummy_tar.num_units = 3
    dummy_tar.capital_cost = 2000
    dummy_tar.total_cost = 5000
    dummy_tar.exergy_sources = 800
    dummy_tar.exergy_sinks = 700
    dummy_tar.ETE = 0.85
    dummy_tar.exergy_req_min = 100
    dummy_tar.exergy_des_min = 90

    json_data = dummy_tar.serialize_json()

    assert json_data["area"] == 120
    assert json_data["num_units"] == 3
    assert json_data["capital_cost"] == 2000
    assert json_data["total_cost"] == 5000
    assert json_data["exergy_sources"] == 800
    assert json_data["exergy_sinks"] == 700
    assert json_data["ETE"] == 85.0
    assert json_data["exergy_req_min"] == 100
    assert json_data["exergy_des_min"] == 90

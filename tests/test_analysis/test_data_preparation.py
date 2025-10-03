"""
Tests for `prepare_problem` in OpenPinch.

This test suite validates the core functionality and edge behavior of the
prepare_problem function, which transforms raw stream and utility input 
data into structured Zone objects with zones, utilities, and heat integration attributes.

Test Categories:
----------------
1. Core Functionality Tests
   - Validates correct assignment of streams and utilities.
   - Verifies zone creation and utility processing logic.

2. Stream and Zone Validations
   - Ensures correct calculation of derived stream properties.
   - Confirms expected zone naming and presence of synthetic zones (TI, TS, etc.).

3. Unit Handling Tests
   - Validates support for mixed unit and unitless values via ValueWithUnit.

4. Invalid Input Tests
   - Asserts appropriate errors for schema validation or missing configuration.

5. Edge Case Tests
   - Covers 16 stream, utility, and configuration edge cases including:
     • Flat temperature gradients (t_supply = t_target)
     • Zero HTC, zero DT values
     • Duplicate stream/utility names
     • Inactive utilities
     • Zone ordering
     • Extreme turbine configuration auto-correction
     • Reuse of site object without duplication

Fixtures:
---------
- `dummy_config`: basic configuration object
- `dummy_site`: Zone using `dummy_config`
- `dummy_streams`: one hot and one cold stream
- `dummy_utilities`: one hot and one cold utility

Note:
-----
All edge cases aim to test both robustness and realistic behavior. Some tests intentionally include borderline inputs 
(e.g. zero values or mismatched heat flow directions) to verify internal fallbacks or fail-safes.

"""

import pytest
from pydantic import ValidationError

from OpenPinch.analysis.data_preparation import _validate_input_data, prepare_problem
from OpenPinch.classes import *
from OpenPinch.lib import *

# ---------------- Fixtures ---------------- #


@pytest.fixture
def dummy_config():
    return Configuration()


@pytest.fixture
def dummy_site(dummy_config):
    return Zone(name="test_site", config=dummy_config)


@pytest.fixture
def dummy_streams():
    return [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": ValueWithUnit(value=250, units="C"),
                "t_target": ValueWithUnit(value=100, units="C"),
                "heat_flow": ValueWithUnit(value=10000, units="kW"),
                "dt_cont": ValueWithUnit(value=10, units="K"),
                "htc": ValueWithUnit(value=1, units="kW/m2.K"),
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "C1",
                "zone": "Z1",
                "t_supply": ValueWithUnit(value=50, units="C"),
                "t_target": ValueWithUnit(value=150, units="C"),
                "heat_flow": ValueWithUnit(value=-8000, units="kW"),
                "dt_cont": ValueWithUnit(value=10, units="K"),
                "htc": ValueWithUnit(value=1, units="kW/m2.K"),
            }
        ),
    ]


@pytest.fixture
def dummy_utilities():
    return [
        UtilitySchema.model_validate(
            {
                "name": "HU1",
                "type": "Hot",
                "t_supply": ValueWithUnit(value=300, units="C"),
                "t_target": ValueWithUnit(value=250, units="C"),
                "heat_flow": 0,
                "dt_cont": ValueWithUnit(value=10, units="K"),
                "price": ValueWithUnit(value=100, units="$/MWh"),
                "htc": ValueWithUnit(value=1, units="kW/m2.K"),
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU1",
                "type": "Cold",
                "t_supply": ValueWithUnit(value=20, units="C"),
                "t_target": ValueWithUnit(value=80, units="C"),
                "heat_flow": 0,
                "dt_cont": ValueWithUnit(value=10, units="K"),
                "price": ValueWithUnit(value=100, units="$/MWh"),
                "htc": ValueWithUnit(value=1, units="kW/m2.K"),
            }
        ),
    ]


# ---------------- Core Functionality Tests ---------------- #


def test_prepare_site_stream_data(dummy_streams, dummy_utilities):
    site = prepare_problem(streams=dummy_streams, utilities=dummy_utilities)
    assert len(site.subzones) > 0
    assert any(z.hot_streams or z.cold_streams for z in site.subzones.values())
    assert len(site.hot_utilities) > 0
    assert len(site.cold_utilities) > 0


def test_prepare_site_stream_data_raises_with_no_streams(dummy_utilities):
    with pytest.raises(ValueError, match="At least one stream is required"):
        _validate_input_data(utilities=dummy_utilities)


def test_prepare_site_stream_data_adds_default_utilities(dummy_streams):
    site = prepare_problem(streams=dummy_streams)
    hu_names = [u.name for u in site.hot_utilities]
    cu_names = [u.name for u in site.cold_utilities]
    assert "HU" in hu_names
    assert "CU" in cu_names


def test_prepare_site_stream_data_handles_extreme_temperatures():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotExtreme",
                "zone": "Zext",
                "t_supply": 1000,
                "t_target": 500,
                "heat_flow": 5000,
                "dt_cont": 5,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "ColdExtreme",
                "zone": "Zext",
                "t_supply": 10,
                "t_target": 300,
                "heat_flow": -4000,
                "dt_cont": 5,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams)
    assert len(site.hot_utilities) > 0
    assert len(site.cold_utilities) > 0


# ---------------- Stream and Zone Validations ---------------- #


def test_stream_attributes_are_computed_correctly(dummy_streams):
    site = prepare_problem(streams=dummy_streams)
    z1 = site.subzones["Z1"]
    hot = next(s for s in z1.hot_streams if s.name == "H1")
    cold = next(s for s in z1.cold_streams if s.name == "C1")
    assert hot.t_max_star == hot.t_supply - hot.dt_cont
    assert cold.t_min_star == cold.t_supply + cold.dt_cont


def test_zone_names_and_ordering(dummy_streams):
    site = prepare_problem(streams=dummy_streams)
    zone_names = list(site.subzones.keys())
    expected_subzones = {"Z1"}
    expected_this_zone = {
        f"{TargetType.DI.value}",
        f"{TargetType.TS.value}",
        f"{TargetType.TZ.value}",
    }
    assert expected_subzones == set(zone_names)


# ---------------- Unit Handling Tests ---------------- #


def test_mixed_unit_and_unitless_inputs():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": ValueWithUnit(value=100, units="C"),
                "heat_flow": ValueWithUnit(value=10000, units="kW"),
                "dt_cont": 10,
                "htc": ValueWithUnit(value=1, units="kW/m2.K"),
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "C1",
                "zone": "Z1",
                "t_supply": ValueWithUnit(value=50, units="C"),
                "t_target": 150,
                "heat_flow": -8000,
                "dt_cont": ValueWithUnit(value=10, units="K"),
                "htc": 1,
            }
        ),
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU",
                "type": "Hot",
                "t_supply": 300,
                "t_target": 250,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": ValueWithUnit(value=100, units="$/MWh"),
                "htc": 1,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU",
                "type": "Cold",
                "t_supply": ValueWithUnit(value=20, units="C"),
                "t_target": ValueWithUnit(value=80, units="C"),
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams, utilities=utilities)
    assert any(u.name == "HU" for u in site.hot_utilities)
    assert any(u.name == "CU" for u in site.cold_utilities)


# ---------------- Invalid Input Tests ---------------- #


def test_invalid_utility_wrong_type():
    with pytest.raises(ValidationError):
        UtilitySchema.model_validate(
            {
                "name": "BadUtility",
                "type": "InvalidType",
                "t_supply": 300,
                "t_target": 250,
                "heat_flow": 0,
                "dt_cont": 10,
                "price": 100,
                "htc": 1,
            }
        )


# ---------------- Edge Case Tests ---------------- #


def test_equal_supply_target_temperature_adjustment():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "Flat",
                "zone": "Z1",
                "t_supply": 200,
                "t_target": 200,
                "heat_flow": 3000,
                "dt_cont": 5,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=streams)

    s = next(iter(site.subzones["Z1"].process_streams))
    assert s.t_supply != s.t_target


def test_duplicate_stream_names():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "DupStream",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": 100,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "DupStream",
                "zone": "Z1",
                "t_supply": 60,
                "t_target": 120,
                "heat_flow": -3000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams)
    names = [s.name for z in site.subzones.values() for s in z.process_streams]
    assert names.count("DupStream") == 2


def test_inactive_utilities_are_ignored(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_inactive",
                "type": "Hot",
                "t_supply": 300,
                "t_target": 250,
                "heat_flow": 0,
                "dt_cont": 10,
                "price": 100,
                "htc": 1,
                "active": False,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU_active",
                "type": "Cold",
                "t_supply": 20,
                "t_target": 80,
                "heat_flow": 0,
                "dt_cont": 10,
                "price": 100,
                "htc": 1,
                "active": True,
            }
        ),
    ]
    site = prepare_problem(streams=dummy_streams, utilities=utilities)

    hot_names = [u.name for u in site.hot_utilities]
    cold_names = [u.name for u in site.cold_utilities]

    assert "HU_inactive" not in hot_names
    assert "CU_active" in cold_names


def test_streams_split_across_multiple_zones():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "Hot1",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 200,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "Cold2",
                "zone": "Z2",
                "t_supply": 50,
                "t_target": 150,
                "heat_flow": -6000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams)
    assert "Z1" in site.subzones
    assert "Z2" in site.subzones
    assert any(s.name == "Hot1" for s in site.subzones["Z1"].hot_streams)
    assert any(s.name == "Cold2" for s in site.subzones["Z2"].cold_streams)


def test_heat_flow_negative_but_temps_suggest_hot():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "MismatchHot",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 200,
                "heat_flow": -5000,
                "dt_cont": 10,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=streams)
    z = site.subzones["Z1"]
    assert len(z.hot_streams) == 1
    assert z.hot_streams[0].name == "MismatchHot"


def test_htc_zero_stream():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "ZeroHTC",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 200,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 0.0,
            }
        )
    ]
    site = prepare_problem(streams=streams)
    stream = site.subzones["Z1"].hot_streams[0]
    assert stream.htc == 1.0
    assert stream.name == "ZeroHTC"


def test_zone_with_no_streams():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "S1",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 200,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=streams)
    assert "Z1" in site.subzones
    assert "Z2" not in site.subzones  # no Z2 given


def test_same_stream_name_different_zones():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "SharedName",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 200,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "SharedName",
                "zone": "Z2",
                "t_supply": 60,
                "t_target": 150,
                "heat_flow": -4000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams)
    assert all(
        any("SharedName" in s.name for s in z.hot_streams + z.cold_streams)
        for z in site.subzones.values()
        if z.name in ["Z1", "Z2"]
    )


def test_only_one_stream(dummy_site):
    streams = [
        StreamSchema.model_validate(
            {
                "name": "OnlyOne",
                "zone": "Z1",
                "t_supply": 150,
                "t_target": 100,
                "heat_flow": 4000,
                "dt_cont": 10,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=streams)
    assert "Z1" in site.subzones
    assert len(site.subzones["Z1"].hot_streams) == 1
    assert len(site.hot_utilities) > 0
    assert len(site.cold_utilities) > 0


def test_only_one_active_utility():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotStream",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": 150,
                "heat_flow": 8000,
                "dt_cont": 10,
                "htc": 1,
            }
        )
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_inactive",
                "type": "Hot",
                "t_supply": 300,
                "t_target": 250,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
                "active": False,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU_active",
                "type": "Cold",
                "t_supply": 20,
                "t_target": 80,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
                "active": True,
            }
        ),
    ]
    site = prepare_problem(streams=streams, utilities=utilities)
    hot_names = [u.name for u in site.hot_utilities]
    cold_names = [u.name for u in site.cold_utilities]
    assert "HU" in hot_names  # Default added
    assert "CU_active" in cold_names


def test_utility_equal_supply_target(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "FlatUtility",
                "type": "Hot",
                "t_supply": 250,
                "t_target": 250,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    names = [u.name for u in site.hot_utilities]
    assert "FlatUtility" in names


def test_utility_with_zero_price(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "FreeHU",
                "type": "Hot",
                "t_supply": 300,
                "t_target": 250,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 0,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    assert any(u.name == "FreeHU" for u in site.hot_utilities)


def test_all_default_utilities_added():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "BasicHot",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": 100,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "BasicCold",
                "zone": "Z1",
                "t_supply": 50,
                "t_target": 150,
                "heat_flow": -4000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams)
    hot_names = [u.name for u in site.hot_utilities]
    cold_names = [u.name for u in site.cold_utilities]
    assert "HU" in hot_names
    assert "CU" in cold_names


def test_duplicate_utility_names(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "Duplicate",
                "type": "Hot",
                "t_supply": 300,
                "t_target": 250,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "Duplicate",
                "type": "Cold",
                "t_supply": 20,
                "t_target": 80,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    names = [u.name for u in site.hot_utilities] + [u.name for u in site.cold_utilities]
    assert names.count("Duplicate") >= 2


def test_utility_sorting_by_temp(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_high",
                "type": "Hot",
                "t_supply": 400,
                "t_target": 300,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "HU_low",
                "type": "Hot",
                "t_supply": 310,
                "t_target": 200,
                "dt_cont": 10,
                "heat_flow": 0,
                "price": 100,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    temps = [u.t_supply for u in site.hot_utilities]
    assert temps == sorted(temps, reverse=True)


def test_zone_name_sort_order():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "Z10_Hot",
                "zone": "Z10",
                "t_supply": 300,
                "t_target": 150,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "Z2_Cold",
                "zone": "Z2",
                "t_supply": 50,
                "t_target": 150,
                "heat_flow": -3000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "Z1_Hot",
                "zone": "Z1",
                "t_supply": 280,
                "t_target": 180,
                "heat_flow": 5000,
                "dt_cont": 10,
                "htc": 1,
            }
        ),
    ]
    site = prepare_problem(streams=streams)
    zone_keys = list(site.subzones.keys())
    assert zone_keys[:3] == sorted(zone_keys[:3])  # check sorting is lexicographic


def test_prepare_site_stream_data_twice(dummy_site):
    streams = [
        StreamSchema.model_validate(
            {
                "name": "TwiceHot",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 150,
                "heat_flow": 6000,
                "dt_cont": 10,
                "htc": 1,
            }
        )
    ]
    site1 = prepare_problem(streams=streams)
    site2 = prepare_problem(streams=streams)
    assert len(site2.subzones) == len(site1.subzones)
    assert len(site2.hot_utilities) == len(site1.hot_utilities)
    assert len(site2.cold_utilities) == len(site1.cold_utilities)


def test_zero_dtcont_and_dtglide():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "ZeroDeltaT",
                "zone": "Z1",
                "t_supply": 300,
                "t_target": 100,
                "heat_flow": 6000,
                "dt_cont": 0,
                "htc": 1,
            }
        )
    ]
    site = prepare_problem(streams=streams)
    stream = site.subzones["Z1"].hot_streams[0]
    assert stream.t_min_star == stream.t_target
    assert stream.t_max_star == stream.t_supply


from OpenPinch.analysis.data_preparation import _validate_zone_tree_structure


def make_stream(zone: str) -> StreamSchema:
    return StreamSchema(
        name="TestStream",
        zone=zone,
        t_supply=150.0,
        t_target=100.0,
        heat_flow=1000.0,
        dt_cont=10.0,
        htc=500.0,
    )


def test_flat_single_zone():
    streams = [make_stream(zone="Boiler")]
    tree = _validate_zone_tree_structure(None, streams)
    assert tree.name == "Site"
    assert len(tree.children) == 1
    assert tree.children[0].name == "Boiler"
    assert tree.children[0].type == "Process Zone"
    assert tree.children[0].children is None


def test_nested_zone_with_slash():
    streams = [make_stream(zone="Plant/Line1")]
    tree = _validate_zone_tree_structure(None, streams)
    assert tree.children[0].name == "Plant"
    assert tree.children[0].children[0].name == "Line1"


def test_multiple_streams_shared_prefix():
    streams = [
        make_stream(zone="Site/Area1/Line1"),
        make_stream(zone="Site/Area1/Line2"),
        make_stream(zone="Site/Area2"),
    ]
    tree = _validate_zone_tree_structure(None, streams)
    site_node = tree.children[0]
    assert site_node.name == "Site"
    assert {c.name for c in site_node.children} == {"Area1", "Area2"}
    area1 = next(c for c in site_node.children if c.name == "Area1")
    assert {c.name for c in area1.children} == {"Line1", "Line2"}


def test_returns_existing_tree_if_valid():
    existing_tree = ZoneTreeSchema(
        name="Site",
        type="Site",
        children=[ZoneTreeSchema(name="ZoneA", type="Process Zone")],
    )
    result = _validate_zone_tree_structure(existing_tree, [])
    assert result == existing_tree


def test_raises_on_utility_zone_input():
    utility_tree = ZoneTreeSchema(name="U1", type="Utility Zone")
    with pytest.raises(
        ValueError, match="Pinch analysis does not apply to Utility Zones."
    ):
        _validate_zone_tree_structure(utility_tree, [])


def test_nested_path_with_whitespace_trimming():
    streams = [make_stream(zone="Plant / Line A ")]
    tree = _validate_zone_tree_structure(None, streams)
    assert tree.children[0].name == "Plant"
    assert tree.children[0].children[0].name == "Line A"


from OpenPinch.analysis.data_preparation import (
    _validate_config_data_completed,
)


def test_invalid_config_missing_op_time(dummy_config):
    dummy_config.ANNUAL_OP_TIME = 0
    out_config = _validate_config_data_completed(dummy_config)
    assert out_config.ANNUAL_OP_TIME > 0


from OpenPinch.analysis.data_preparation import _get_validated_zone_info


@pytest.mark.parametrize(
    "zone_type_str, expected_zone_type",
    [
        ("Zone", ZoneType.P.value),
        ("Sub-Zone", ZoneType.P.value),
        ("Process Zone", ZoneType.P.value),
        ("Site", ZoneType.S.value),
        ("Community", ZoneType.C.value),
        ("Region", ZoneType.R.value),
        ("Utility Zone", ZoneType.U.value),
    ],
)
def test_valid_zone_types(zone_type_str, expected_zone_type):
    zone_tree = ZoneTreeSchema.model_validate(
        {"name": "TestZone", "type": zone_type_str, "children": []}
    )
    _, actual_zone_type = _get_validated_zone_info(zone_tree)
    assert actual_zone_type == expected_zone_type


def test_unexpected_zone_type_raises():
    zone_tree = ZoneTreeSchema.model_validate(
        {"name": "UnknownZone", "type": "Invalid Type", "children": []}
    )
    with pytest.raises(
        ValueError, match="Zone name and type could not be identified correctly."
    ):
        _get_validated_zone_info(zone_tree)


def test_non_schema_input_returns_site_zone_type():
    _, actual_zone_type = _get_validated_zone_info("not_a_schema")
    assert actual_zone_type == ZoneType.S.value


from OpenPinch.analysis.data_preparation import _create_nested_zones


@pytest.fixture
def config():
    # Minimal mock config with required attributes
    class Config:
        TOP_ZONE_NAME = "MySite"
        TOP_ZONE_IDENTIFIER = ZoneType.S.value

    return Config()


def make_zone_tree_schema():
    # Example nested structure: Site -> Area1 -> Line1
    return ZoneTreeSchema(
        name="MySite",
        type=ZoneType.S.value,
        children=[
            ZoneTreeSchema(
                name="Area1",
                type=ZoneType.P.value,
                children=[ZoneTreeSchema(name="Line1", type=ZoneType.P.value)],
            )
        ],
    )


def test_creates_nested_zones_correctly(config: Configuration):
    zone_tree = make_zone_tree_schema()
    master_zone = Zone(
        name=config.TOP_ZONE_NAME, identifier=config.TOP_ZONE_IDENTIFIER, config=config
    )

    result = _create_nested_zones(master_zone, zone_tree, config)

    assert result.name == "MySite"
    assert len(result.subzones) == 1

    area1 = result.subzones["Area1"]
    assert area1.name == "Area1"
    assert area1.identifier == ZoneType.P.value
    assert len(area1.subzones) == 1

    line1 = area1.subzones["Line1"]
    assert line1.name == "Line1"
    assert line1.identifier == ZoneType.P.value
    assert line1.subzones == {}


def test_empty_zone_tree_returns_parent(config):
    zone_tree = ZoneTreeSchema(name="MySite", type=ZoneType.S.value, children=None)
    parent_zone = Zone(name="MySite", identifier=ZoneType.S.value, config=config)

    result = _create_nested_zones(parent_zone, zone_tree, config)

    assert result == parent_zone
    assert result.subzones == {}

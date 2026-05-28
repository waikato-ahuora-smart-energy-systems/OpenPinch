"""
Tests for `prepare_problem` in OpenPinch.

This test suite validates the core functionality and edge behavior of the
prepare_problem function, which transforms raw stream and utility input
data into structured Zone objects with zones, utilities, and heat integration
attributes.

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
All edge cases aim to test both robustness and realistic behavior. Some tests
intentionally include borderline inputs (e.g. zero values or mismatched heat
flow directions) to verify internal fallbacks or fail-safes.

"""

import pytest
from pydantic import ValidationError

from OpenPinch.classes import *
from OpenPinch.lib import *
from OpenPinch.services.input_data_processing.data_preparation import (
    _assign_process_streams_to_subzones,
    _build_prepared_stream_collection,
    _create_nested_zones,
    _get_validated_zone_info,
    _validate_config_data_completed,
    _validate_input_data,
    _validate_zone_tree_structure,
    prepare_problem,
)

# ---------------- Fixtures ---------------- #


@pytest.fixture
def dummy_config():
    """Return dummy config data used by this test module."""
    return Configuration()


@pytest.fixture
def dummy_site(dummy_config):
    """Return dummy site data used by this test module."""
    return Zone(name="test_site", zone_config=dummy_config)


@pytest.fixture
def dummy_streams():
    """Return dummy streams data used by this test module."""
    return [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": ValueWithUnit(value=250, unit="C"),
                "t_target": ValueWithUnit(value=100, unit="C"),
                "heat_flow": ValueWithUnit(value=10000, unit="kW"),
                "dt_cont": ValueWithUnit(value=10, unit="K"),
                "htc": ValueWithUnit(value=1, unit="kW/m2.K"),
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "C1",
                "zone": "Z1",
                "t_supply": ValueWithUnit(value=50, unit="C"),
                "t_target": ValueWithUnit(value=150, unit="C"),
                "heat_flow": ValueWithUnit(value=-8000, unit="kW"),
                "dt_cont": ValueWithUnit(value=10, unit="K"),
                "htc": ValueWithUnit(value=1, unit="kW/m2.K"),
            }
        ),
    ]


@pytest.fixture
def dummy_utilities():
    """Return dummy utilities data used by this test module."""
    return [
        UtilitySchema.model_validate(
            {
                "name": "HU1",
                "type": "Hot",
                "t_supply": ValueWithUnit(value=300, unit="C"),
                "t_target": ValueWithUnit(value=250, unit="C"),
                "heat_flow": 0,
                "dt_cont": ValueWithUnit(value=10, unit="K"),
                "price": ValueWithUnit(value=100, unit="$/MWh"),
                "htc": ValueWithUnit(value=1, unit="kW/m2.K"),
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU1",
                "type": "Cold",
                "t_supply": ValueWithUnit(value=20, unit="C"),
                "t_target": ValueWithUnit(value=80, unit="C"),
                "heat_flow": 0,
                "dt_cont": ValueWithUnit(value=10, unit="K"),
                "price": ValueWithUnit(value=100, unit="$/MWh"),
                "htc": ValueWithUnit(value=1, unit="kW/m2.K"),
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
    assert hot.t_max_star == hot.t_supply - hot.dt_cont_act
    assert cold.t_min_star == cold.t_supply + cold.dt_cont_act


def test_zone_names_and_ordering(dummy_streams):
    site = prepare_problem(streams=dummy_streams)
    zone_names = list(site.subzones.keys())
    expected_subzones = {"Z1"}
    assert expected_subzones == set(zone_names)


# ---------------- Unit Handling Tests ---------------- #


def test_mixed_unit_and_unitless_inputs():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": ValueWithUnit(value=100, unit="C"),
                "heat_flow": ValueWithUnit(value=10000, unit="kW"),
                "dt_cont": 10,
                "htc": ValueWithUnit(value=1, unit="kW/m2.K"),
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "C1",
                "zone": "Z1",
                "t_supply": ValueWithUnit(value=50, unit="C"),
                "t_target": 150,
                "heat_flow": -8000,
                "dt_cont": ValueWithUnit(value=10, unit="K"),
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
                "price": ValueWithUnit(value=100, unit="$/MWh"),
                "htc": 1,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU",
                "type": "Cold",
                "t_supply": ValueWithUnit(value=20, unit="C"),
                "t_target": ValueWithUnit(value=80, unit="C"),
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
    assert names.count("DupStream") == 1
    assert names.count("DupStream_1") == 1


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
    assert "O1" in site.subzones["Z1"].subzones
    assert "O1" in site.subzones["Z2"].subzones
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
                "name": "HotSeam",
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
    assert names.count("Duplicate") == 1
    assert names.count("Duplicate_1") == 1


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


def test_zone_tree_dt_cont_multiplier_validation():
    with pytest.raises(ValidationError, match="dt_cont_multiplier"):
        ZoneTreeSchema.model_validate(
            {
                "name": "Root",
                "type": "Site",
                "dt_cont_multiplier": -1.0,
            }
        )


def test_zone_dt_cont_multiplier_inherits_from_root_for_streams_and_utilities():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "Site/AreaA",
                "t_supply": 200.0,
                "t_target": 100.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        )
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "Steam",
                "type": "Hot",
                "t_supply": 250.0,
                "t_target": 220.0,
                "heat_flow": 0.0,
                "dt_cont": 8.0,
                "htc": 1.0,
                "price": 10.0,
            }
        )
    ]
    zone_tree = ZoneTreeSchema.model_validate(
        {
            "name": "Site",
            "type": "Site",
            "dt_cont_multiplier": 2.0,
            "children": [{"name": "AreaA", "type": "Process Zone"}],
        }
    )

    site = prepare_problem(streams=streams, utilities=utilities, zone_tree=zone_tree)
    area = site.get_subzone("AreaA")
    hot_stream = next(stream for stream in area.hot_streams if stream.name == "HotA")
    hot_utility = next(
        utility for utility in area.hot_utilities if utility.name == "Steam"
    )

    assert site.dt_cont_multiplier == 2.0
    assert area.dt_cont_multiplier == 2.0
    assert hot_stream.dt_cont == 10.0
    assert hot_stream.dt_cont_act == 20.0
    assert hot_stream.t_min_star == 80.0
    assert hot_utility.dt_cont == 8.0
    assert hot_utility.dt_cont_act == 16.0


def test_zone_dt_cont_multiplier_child_override_is_absolute():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "Site/AreaA",
                "t_supply": 220.0,
                "t_target": 120.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "HotB",
                "zone": "Site/AreaB",
                "t_supply": 220.0,
                "t_target": 120.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ),
    ]
    zone_tree = ZoneTreeSchema.model_validate(
        {
            "name": "Site",
            "type": "Site",
            "dt_cont_multiplier": 2.0,
            "children": [
                {"name": "AreaA", "type": "Process Zone"},
                {
                    "name": "AreaB",
                    "type": "Process Zone",
                    "dt_cont_multiplier": 0.5,
                },
            ],
        }
    )

    site = prepare_problem(streams=streams, zone_tree=zone_tree)
    area_a = site.get_subzone("AreaA")
    area_b = site.get_subzone("AreaB")

    hot_a = next(stream for stream in area_a.hot_streams if stream.name == "HotA")
    hot_b = next(stream for stream in area_b.hot_streams if stream.name == "HotB")

    assert area_a.dt_cont_multiplier == 2.0
    assert area_b.dt_cont_multiplier == 0.5
    assert hot_a.dt_cont == 10.0
    assert hot_a.dt_cont_act == 20.0
    assert hot_b.dt_cont == 10.0
    assert hot_b.dt_cont_act == 5.0


def test_default_utilities_use_zone_effective_dt_cont_multiplier():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "AreaA",
                "t_supply": 180.0,
                "t_target": 120.0,
                "heat_flow": 300.0,
                "dt_cont": 5.0,
                "htc": 1.0,
            }
        )
    ]
    zone_tree = ZoneTreeSchema.model_validate(
        {
            "name": "Site",
            "type": "Site",
            "dt_cont_multiplier": 3.0,
            "children": [
                {
                    "name": "AreaA",
                    "type": "Process Zone",
                }
            ],
        }
    )

    site = prepare_problem(streams=streams, zone_tree=zone_tree)
    area = site.get_subzone("AreaA")
    hu = next(utility for utility in area.hot_utilities if utility.name == "HU")
    cu = next(utility for utility in area.cold_utilities if utility.name == "CU")

    assert hu.dt_cont == pytest.approx(area.config.DT_CONT)
    assert hu.dt_cont_act == pytest.approx(area.config.DT_CONT * 3.0)
    assert cu.dt_cont == pytest.approx(area.config.DT_CONT)
    assert cu.dt_cont_act == pytest.approx(area.config.DT_CONT * 3.0)


def test_phase_change_utility_with_null_target_is_completed_near_supply_temperature():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "AreaA",
                "t_supply": 180.0,
                "t_target": 120.0,
                "heat_flow": 300.0,
                "dt_cont": 5.0,
                "htc": 1.0,
            }
        )
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HPS",
                "type": "Both",
                "t_supply": {"value": 280.0, "unit": "degC"},
                "t_target": {"value": None, "unit": "degC"},
                "dt_cont": {"value": 0.0, "unit": "degC"},
                "price": 30.0,
                "htc": {"value": 1.0, "unit": "kW/m^2/degC"},
                "heat_flow": None,
            }
        )
    ]

    site = prepare_problem(streams=streams, utilities=utilities)
    hot_hps = site.hot_utilities[".".join([StreamLoc.HotU.value, "HPS"])]
    cold_hps = site.cold_utilities[".".join([StreamLoc.ColdU.value, "HPS"])]

    assert hot_hps.t_supply == pytest.approx(280.0)
    assert hot_hps.t_target == pytest.approx(279.99)
    assert cold_hps.t_supply == pytest.approx(279.99)
    assert cold_hps.t_target == pytest.approx(280.0)


def test_process_stream_with_null_dt_cont_defaults_to_zero():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "AreaA",
                "t_supply": 180.0,
                "t_target": 120.0,
                "heat_flow": 300.0,
                "dt_cont": {"value": None, "unit": "degC"},
                "htc": 1.0,
            }
        )
    ]

    site = prepare_problem(streams=streams)
    hot_a = next(stream for stream in site.hot_streams if stream.name == "HotA")

    assert hot_a.dt_cont == pytest.approx(0.0)
    assert hot_a.dt_cont_act == pytest.approx(0.0)


def test_stateful_process_extrema_use_all_states_for_default_hot_utility():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "ColdA",
                "zone": "AreaA",
                "t_supply": {
                    "values": [40.0, 60.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [150.0, 200.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "heat_flow": {
                    "values": [120.0, 160.0],
                    "state_ids": ["0", "1"],
                    "unit": "kW",
                },
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "AreaA",
                "t_supply": 260.0,
                "t_target": 140.0,
                "heat_flow": 200.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ),
    ]

    site = prepare_problem(streams=streams)
    hu = next(utility for utility in site.hot_utilities if utility.name == "HU")

    assert hu.t_max_star == pytest.approx(210.0)


def test_stateful_hot_utility_sorting_uses_all_state_envelope(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_swing",
                "type": "Hot",
                "t_supply": {
                    "values": [250.0, 400.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [200.0, 350.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "dt_cont": 10.0,
                "heat_flow": 0.0,
                "price": 100.0,
                "htc": 1.0,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "HU_flat",
                "type": "Hot",
                "t_supply": {
                    "values": [300.0, 310.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [260.0, 270.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "dt_cont": 10.0,
                "heat_flow": 0.0,
                "price": 100.0,
                "htc": 1.0,
            }
        ),
    ]

    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    hot_names = [utility.name for utility in site.hot_utilities]

    assert hot_names[:2] == ["HU_swing", "HU_flat"]


def test_prepare_problem_rejects_cross_stream_state_mismatch():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "AreaA",
                "t_supply": {
                    "values": [220.0, 210.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [140.0, 130.0],
                    "state_ids": ["0", "1"],
                    "unit": "degC",
                },
                "heat_flow": {
                    "values": [200.0, 180.0],
                    "state_ids": ["0", "1"],
                    "unit": "kW",
                },
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        )
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_mismatch",
                "type": "Hot",
                "t_supply": {
                    "values": [300.0, 290.0],
                    "state_ids": ["0", "peak"],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [260.0, 250.0],
                    "state_ids": ["0", "peak"],
                    "unit": "degC",
                },
                "dt_cont": {
                    "values": [10.0, 10.0],
                    "state_ids": ["0", "peak"],
                    "unit": "delta_degC",
                },
                "heat_flow": 0.0,
                "price": 100.0,
                "htc": 1.0,
            }
        )
    ]

    with pytest.raises(ValueError, match="state_ids for stream 'Hot Utility.HU_mismatch'"):
        prepare_problem(streams=streams, utilities=utilities)


def test_build_prepared_stream_collection_rejects_neutral_process_stream():
    master_zone = Zone(name="Site", type=ZT.S.value, zone_config=Configuration())
    area = Zone(
        name="AreaA",
        type=ZT.P.value,
        zone_config=master_zone.config,
        parent_zone=master_zone,
    )
    master_zone.add_zone(area)
    streams = [
        StreamSchema.model_validate(
            {
                "name": "NeutralA",
                "zone": "Site/AreaA",
                "t_supply": 100.0,
                "t_target": 100.0,
                "heat_flow": 0.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        )
    ]

    with pytest.raises(ValueError, match="must classify as Hot or Cold"):
        _build_prepared_stream_collection(master_zone, streams, [])


def test_prepare_problem_process_streams_are_referenced_in_parent_imports():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "Site/AreaA",
                "t_supply": 220.0,
                "t_target": 120.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        )
    ]
    zone_tree = ZoneTreeSchema.model_validate(
        {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "AreaA", "type": "Process Zone"}],
        }
    )

    site = prepare_problem(streams=streams, zone_tree=zone_tree)
    area = site.get_subzone("AreaA")

    assert site.hot_streams["AreaA.HotA"] is area.hot_streams[0]


def test_prepare_problem_utilities_are_copied_per_zone():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "Site/AreaA",
                "t_supply": 220.0,
                "t_target": 120.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "ColdB",
                "zone": "Site/AreaB",
                "t_supply": 60.0,
                "t_target": 140.0,
                "heat_flow": -400.0,
                "dt_cont": 8.0,
                "htc": 1.0,
            }
        ),
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "Steam",
                "type": "Hot",
                "t_supply": 260.0,
                "t_target": 230.0,
                "heat_flow": 0.0,
                "dt_cont": 5.0,
                "htc": 1.0,
                "price": 10.0,
            }
        )
    ]
    zone_tree = ZoneTreeSchema.model_validate(
        {
            "name": "Site",
            "type": "Site",
            "children": [
                {"name": "AreaA", "type": "Process Zone"},
                {"name": "AreaB", "type": "Process Zone"},
            ],
        }
    )

    site = prepare_problem(streams=streams, utilities=utilities, zone_tree=zone_tree)
    root_utility = site.hot_utilities[".".join([StreamLoc.HotU.value, "Steam"])]
    area_a_utility = site.get_subzone("AreaA").hot_utilities[
        ".".join([StreamLoc.HotU.value, "Steam"])
    ]
    area_b_utility = site.get_subzone("AreaB").hot_utilities[
        ".".join([StreamLoc.HotU.value, "Steam"])
    ]

    assert root_utility is not area_a_utility
    assert area_a_utility is not area_b_utility
    assert root_utility.t_supply == area_a_utility.t_supply == area_b_utility.t_supply
    assert root_utility.t_target == area_a_utility.t_target == area_b_utility.t_target


def test_zone_dt_cont_multiplier_changes_only_zone_utility_copies():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotA",
                "zone": "Site/AreaA",
                "t_supply": 220.0,
                "t_target": 120.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 1.0,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "HotB",
                "zone": "Site/AreaB",
                "t_supply": 210.0,
                "t_target": 130.0,
                "heat_flow": 450.0,
                "dt_cont": 6.0,
                "htc": 1.0,
            }
        ),
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "Steam",
                "type": "Hot",
                "t_supply": 260.0,
                "t_target": 230.0,
                "heat_flow": 0.0,
                "dt_cont": 5.0,
                "htc": 1.0,
                "price": 10.0,
            }
        )
    ]
    zone_tree = ZoneTreeSchema.model_validate(
        {
            "name": "Site",
            "type": "Site",
            "children": [
                {"name": "AreaA", "type": "Process Zone"},
                {"name": "AreaB", "type": "Process Zone"},
            ],
        }
    )

    site = prepare_problem(streams=streams, utilities=utilities, zone_tree=zone_tree)
    area_a = site.get_subzone("AreaA")
    area_b = site.get_subzone("AreaB")
    root_hot_ref = site.hot_streams["AreaA.HotA"]

    area_a_hot = area_a.hot_streams[0]
    area_b_hot = area_b.hot_streams[0]
    root_utility = site.hot_utilities[".".join([StreamLoc.HotU.value, "Steam"])]
    area_a_utility = area_a.hot_utilities[".".join([StreamLoc.HotU.value, "Steam"])]
    area_b_utility = area_b.hot_utilities[".".join([StreamLoc.HotU.value, "Steam"])]

    area_a.dt_cont_multiplier = 2.0

    assert area_a_hot.dt_cont_act == pytest.approx(area_a_hot.dt_cont * 2.0)
    assert root_hot_ref is area_a_hot
    assert root_hot_ref.dt_cont_act == pytest.approx(area_a_hot.dt_cont * 2.0)
    assert area_b_hot.dt_cont_act == pytest.approx(area_b_hot.dt_cont)
    assert area_a_utility.dt_cont_act == pytest.approx(area_a_utility.dt_cont * 2.0)
    assert root_utility.dt_cont_act == pytest.approx(root_utility.dt_cont)
    assert area_b_utility.dt_cont_act == pytest.approx(area_b_utility.dt_cont)


def make_stream(zone: str) -> StreamSchema:
    """Build stream data used by this test module."""
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
    process_zone = tree.children[0]
    assert process_zone.name == "Boiler"
    assert process_zone.type == ZT.P.value
    assert process_zone.children is not None
    assert len(process_zone.children) == 1
    operation_zone = process_zone.children[0]
    assert operation_zone.name == "O1"
    assert operation_zone.type == ZT.O.value
    assert operation_zone.children is None


def test_nested_zone_with_slash():
    streams = [make_stream(zone="Plant/Line1")]
    tree = _validate_zone_tree_structure(None, streams)
    plant = tree.children[0]
    assert plant.name == "Plant"
    line1 = plant.children[0]
    assert line1.name == "Line1"
    assert line1.type == ZT.P.value
    assert len(line1.children) == 1
    assert line1.children[0].name == "O1"
    assert line1.children[0].type == ZT.O.value


def test_generic_zone_tree_defaults_by_depth():
    zone_tree = ZoneTreeSchema(
        name="Layer0",
        type="Zone",
        children=[
            ZoneTreeSchema(
                name="Layer1",
                type="Zone",
                children=[ZoneTreeSchema(name="Layer2", type="Zone")],
            )
        ],
    )
    normalised = _validate_zone_tree_structure(zone_tree, [])
    assert normalised.type == ZT.S.value
    assert normalised.children[0].type == ZT.P.value
    assert normalised.children[0].children[0].type == ZT.O.value


def test_site_level_streams_get_individual_process_zones():
    zone_tree = ZoneTreeSchema(name="SiteRoot", type="Zone", children=[])
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotStream",
                "zone": "SiteRoot",
                "t_supply": 200.0,
                "t_target": 100.0,
                "heat_flow": 500.0,
                "dt_cont": 10.0,
                "htc": 500.0,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "ColdStream",
                "zone": "SiteRoot",
                "t_supply": 50.0,
                "t_target": 150.0,
                "heat_flow": -400.0,
                "dt_cont": 10.0,
                "htc": 500.0,
            }
        ),
    ]

    normalised = _validate_zone_tree_structure(zone_tree, streams)
    child_names = {child.name for child in normalised.children}
    assert {"HotStream", "ColdStream"} <= child_names

    for child in normalised.children:
        if child.name in {"HotStream", "ColdStream"}:
            assert child.type == ZT.P.value
            assert child.children is None

    assert streams[0].zone == "SiteRoot/HotStream"
    assert streams[1].zone == "SiteRoot/ColdStream"


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
    line1 = next(c for c in area1.children if c.name == "Line1")
    assert {c.name for c in line1.children} == {"O1"}
    area2 = next(c for c in site_node.children if c.name == "Area2")
    assert {c.name for c in area2.children} == {"O1"}


def test_operation_zone_counter_and_zone_rewrite():
    stream_a1 = make_stream(zone="Area1")
    stream_a1.name = "StreamA1"
    stream_a1b = make_stream(zone="Area1")
    stream_a1b.name = "StreamA1B"
    stream_a2 = make_stream(zone="Area2")
    stream_a2.name = "StreamA2"
    streams = [stream_a1, stream_a1b, stream_a2]

    tree = _validate_zone_tree_structure(None, streams)

    area1 = next(c for c in tree.children if c.name == "Area1")
    assert {c.name for c in area1.children} == {"O1", "O2"}

    area2 = next(c for c in tree.children if c.name == "Area2")
    assert {c.name for c in area2.children} == {"O1"}

    assert stream_a1.zone == "Site/Area1/O1"
    assert stream_a1b.zone == "Site/Area1/O2"
    assert stream_a2.zone == "Site/Area2/O1"


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
    plant = tree.children[0]
    assert plant.name == "Plant"
    line_a = plant.children[0]
    assert line_a.name == "Line A"
    assert {c.name for c in line_a.children} == {"O1"}


def test_invalid_config_missing_op_time(dummy_config):
    dummy_config.ANNUAL_OP_TIME = 0
    out_config = _validate_config_data_completed(dummy_config)
    assert out_config.ANNUAL_OP_TIME > 0


def test_turbine_pressure_is_clamped_using_canonical_config_fields(dummy_config):
    dummy_config.DO_TURBINE_WORK = True
    dummy_config.TURB_P_IN = 250.0

    out_config = _validate_config_data_completed(dummy_config)

    assert out_config.TURB_P_IN == 200


def test_prepare_problem_preserves_passed_configuration_object(dummy_streams):
    cfg = Configuration(
        options={
            "DO_TURBINE_WORK": True,
            "TURB_P_IN": 12.0,
            "TURB_T_IN": 375.0,
            "IS_HIGH_P_COND_FLASH": True,
        }
    )

    site = prepare_problem(streams=dummy_streams, options=cfg)

    assert site.config is not cfg
    assert site.config.DO_TURBINE_WORK is True
    assert site.config.TURB_P_IN == 12.0
    assert site.config.TURB_T_IN == 375.0
    assert site.config.IS_HIGH_P_COND_FLASH is True


def test_prepare_problem_rejects_removed_legacy_turbine_gateway(dummy_streams):
    with pytest.raises(ValueError, match="Unknown configuration option"):
        prepare_problem(
            streams=dummy_streams,
            options={"turbine": [{"key": "PROP_TOP_0", "value": 450.0}]},
        )


@pytest.mark.parametrize(
    "zone_type_str, expected_zone_type",
    [
        ("Zone", ZT.S.value),
        ("Sub-Zone", ZT.P.value),
        ("Process Zone", ZT.P.value),
        ("Site", ZT.S.value),
        ("Community", ZT.C.value),
        ("Region", ZT.R.value),
        ("Utility Zone", ZT.U.value),
    ],
)
def test_valid_zone_types(zone_type_str, expected_zone_type):
    zone_tree = ZoneTreeSchema.model_validate(
        {"name": "TestZone", "type": zone_type_str, "children": []}
    )
    _, actual_zone_type = _get_validated_zone_info(zone_tree)
    assert actual_zone_type == expected_zone_type


def test_zone_type_defaults_with_depth():
    root = ZoneTreeSchema(
        name="Root", type="Zone", children=[ZoneTreeSchema(name="Child", type="Zone")]
    )
    _, child_type = _get_validated_zone_info(root.children[0], depth=1)
    _, grandchild_type = _get_validated_zone_info(
        ZoneTreeSchema(name="Grandchild", type="Zone"), depth=2
    )
    assert child_type == ZT.P.value
    assert grandchild_type == ZT.O.value


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
    assert actual_zone_type == ZT.S.value


@pytest.fixture
def config():
    # Minimal mock config with required attributes
    """Test helper for config."""

    class Config:
        TOP_ZONE_NAME = "MySite"
        TOP_ZONE_IDENTIFIER = ZT.S.value

    return Config()


def make_zone_tree_schema():
    # Example nested structure: Site -> Area1 -> Line1
    """Build zone tree schema data used by this test module."""
    return ZoneTreeSchema(
        name="MySite",
        type=ZT.S.value,
        children=[
            ZoneTreeSchema(
                name="Area1",
                type=ZT.P.value,
                children=[ZoneTreeSchema(name="Line1", type=ZT.P.value)],
            )
        ],
    )


def test_creates_nested_zones_correctly():
    zone_config = Configuration()
    zone_tree = make_zone_tree_schema()
    master_zone = Zone(
        name=zone_config.TOP_ZONE_NAME,
        type=zone_config.TOP_ZONE_IDENTIFIER,
        zone_config=zone_config,
    )

    result = _create_nested_zones(master_zone, zone_tree, zone_config)

    assert result.name == "Site"
    assert len(result.subzones) == 1

    area1 = result.subzones["Area1"]
    assert area1.name == "Area1"
    assert area1.type == ZT.P.value
    assert len(area1.subzones) == 1

    line1 = area1.subzones["Line1"]
    assert line1.name == "Line1"
    assert line1.type == ZT.P.value
    assert line1.subzones == {}


def test_empty_zone_tree_returns_parent():
    zone_config = Configuration()
    zone_tree = ZoneTreeSchema(name="MySite", type=ZT.S.value, children=None)
    parent_zone = Zone(name="MySite", type=ZT.S.value, zone_config=zone_config)

    result = _create_nested_zones(parent_zone, zone_tree, zone_config)

    assert result == parent_zone
    assert result.subzones == {}


def test_assign_process_streams_to_subzones_requires_zone_mapping():
    master_zone = Zone(name="Site", type=ZT.S.value, zone_config=Configuration())
    process_streams = StreamCollection()
    process_streams.add(
        Stream(
            name="H1",
            t_supply=200.0,
            t_target=100.0,
            heat_flow=10.0,
            dt_cont=5.0,
            htc=1.0,
            is_process_stream=True,
        ),
        key="Site.Hot Stream.H1",
    )

    with pytest.raises(RuntimeError, match="missing a zone mapping"):
        _assign_process_streams_to_subzones(
            master_zone=master_zone,
            process_streams=process_streams,
            process_zone_paths={},
        )

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

import warnings

import pytest
from pydantic import ValidationError

from OpenPinch.application._problem.input.canonicalization import (
    _apply_zone_dt_cont_multiplier,
    _build_zone_config,
    _create_nested_zones,
    _get_validated_zone_info,
    _resolve_zone_dt_cont_multiplier,
    _rewrite_stream_zones_from_tree,
    _validate_config_data_completed,
    _validate_input_data,
    _validate_zone_tree_structure,
)
from OpenPinch.application._problem.input.construction import (
    _assign_process_streams_to_subzones,
    _build_prepared_stream_collection,
    _find_extreme_process_temperatures,
    prepare_problem,
)
from OpenPinch.application._problem.input.utilities import (
    _orient_utility_temperatures,
    _utility_temperature_arrays,
)
from OpenPinch.contracts.common import ValueWithUnit
from OpenPinch.contracts.input import (
    StreamSchema,
    UtilitySchema,
    ZoneTreeSchema,
)
from OpenPinch.domain._stream.segment import StreamSegment
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.enums import (
    ST,
    ZT,
    StreamLoc,
)
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.zone import Zone

# ---------------- Fixtures ---------------- #


@pytest.fixture
def dummy_config():
    """Return dummy config data used by this test module."""
    return Configuration()


@pytest.fixture
def dummy_site(dummy_config):
    """Return dummy site data used by this test module."""
    return Zone(name="test_site", config=dummy_config)


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


def test_default_utility_extrema_use_authoritative_segment_shifted_temperatures():
    hot = Stream(
        name="Segmented hot",
        segments=[
            StreamSegment(
                name="S1",
                t_supply=200.0,
                t_target=150.0,
                heat_flow=50.0,
                dt_cont=20.0,
            ),
            StreamSegment(
                name="S2",
                t_supply=150.0,
                t_target=100.0,
                heat_flow=50.0,
                dt_cont=0.0,
            ),
        ],
    )
    hot_streams = StreamCollection()
    hot_streams.add(hot)

    hu_t_min, cu_t_max = _find_extreme_process_temperatures(
        hot_streams,
        StreamCollection(),
    )

    assert float(hot.t_min_star) == pytest.approx(80.0)
    assert min(float(segment.t_min_star) for segment in hot.segments) == pytest.approx(
        100.0
    )
    assert hu_t_min == pytest.approx(100.0)
    assert cu_t_max == pytest.approx(100.0)


# ---------------- Stream and Zone Validations ---------------- #


def test_stream_attributes_are_computed_correctly(dummy_streams):
    site = prepare_problem(streams=dummy_streams)
    z1 = site.subzones["Z1"]
    hot = next(s for s in z1.hot_streams if s.name == "H1")
    cold = next(s for s in z1.cold_streams if s.name == "C1")
    assert hot.t_max_star[0] == hot.t_supply[0] - hot.dt_cont_act[0]
    assert cold.t_min_star[0] == cold.t_supply[0] + cold.dt_cont_act[0]


def test_prepare_problem_preserves_stream_fluid_pressure_and_enthalpy_fields():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": 100,
                "p_supply": ValueWithUnit(value=2.0, unit="bar"),
                "p_target": ValueWithUnit(value=150.0, unit="kPa"),
                "h_supply": ValueWithUnit(value=2_800_000.0, unit="J/kg"),
                "h_target": ValueWithUnit(value=2_300.0, unit="kJ/kg"),
                "heat_flow": 10_000,
                "fluid_name": "HEOS::Water",
                "fluid_phase": "gas",
            }
        )
    ]

    site = prepare_problem(streams=streams)
    hot = next(s for s in site.subzones["Z1"].hot_streams if s.name == "H1")

    assert hot.fluid_name == "HEOS::Water"
    assert hot.fluid_phase == "gas"
    assert hot.p_supply.value == pytest.approx(200.0)
    assert hot.p_supply.unit == "kPa"
    assert hot.p_target.value == pytest.approx(150.0)
    assert hot.h_supply.value == pytest.approx(2_800.0)
    assert hot.h_supply.unit == "kJ/kg"
    assert hot.h_target.value == pytest.approx(2_300.0)


def test_prepare_problem_preserves_utility_fluid_pressure_and_enthalpy_fields(
    dummy_streams,
):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "Steam",
                "type": "Hot",
                "t_supply": 300,
                "t_target": 250,
                "p_supply": ValueWithUnit(value=8.0, unit="bar"),
                "p_target": ValueWithUnit(value=7.0, unit="bar"),
                "h_supply": ValueWithUnit(value=2_750.0, unit="kJ/kg"),
                "h_target": ValueWithUnit(value=700_000.0, unit="J/kg"),
                "fluid_name": "Water",
                "fluid_phase": "vle",
            }
        )
    ]

    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    steam = next(s for s in site.hot_utilities if s.name == "Steam")

    assert steam.fluid_name == "Water"
    assert steam.fluid_phase == "vle"
    assert steam.p_supply.value == pytest.approx(800.0)
    assert steam.p_target.value == pytest.approx(700.0)
    assert steam.h_supply.value == pytest.approx(2_750.0)
    assert steam.h_target.value == pytest.approx(700.0)


def test_prepare_problem_rejects_invalid_coolprop_stream_fluid_name():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": 250,
                "t_target": 100,
                "heat_flow": 10_000,
                "fluid_name": "NotARealCoolPropFluid",
            }
        )
    ]

    with pytest.raises(ValueError, match="Invalid CoolProp fluid_name"):
        prepare_problem(streams=streams)


def test_prepare_problem_orients_utility_pressure_and_enthalpy_with_temperature(
    dummy_streams,
):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "Cooling",
                "type": "Cold",
                "t_supply": 80,
                "t_target": 20,
                "p_supply": 800,
                "p_target": 700,
                "h_supply": 80,
                "h_target": 20,
            }
        )
    ]

    site = prepare_problem(streams=dummy_streams, utilities=utilities)
    cooling = next(s for s in site.cold_utilities if s.name == "Cooling")

    assert cooling.t_supply.value == pytest.approx(20.0)
    assert cooling.t_target.value == pytest.approx(80.0)
    assert cooling.p_supply.value == pytest.approx(700.0)
    assert cooling.p_target.value == pytest.approx(800.0)
    assert cooling.h_supply.value == pytest.approx(20.0)
    assert cooling.h_target.value == pytest.approx(80.0)


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


def test_prepare_problem_applies_configured_input_unit_defaults():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "H1",
                "zone": "Z1",
                "t_supply": 523.15,
                "t_target": 373.15,
                "heat_flow": 1000.0,
            }
        ),
        StreamSchema.model_validate(
            {
                "name": "C1",
                "zone": "Z1",
                "t_supply": 303.15,
                "t_target": 423.15,
                "heat_flow": -800.0,
            }
        ),
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_K",
                "type": "Hot",
                "t_supply": 573.15,
                "t_target": 523.15,
                "heat_flow": 0.0,
                "dt_cont": 10.0,
                "htc": 500.0,
            }
        ),
        UtilitySchema.model_validate(
            {
                "name": "CU_K",
                "type": "Cold",
                "t_supply": 293.15,
                "t_target": 313.15,
                "heat_flow": 0.0,
                "dt_cont": 10.0,
                "htc": 500.0,
            }
        ),
    ]

    site = prepare_problem(
        streams=streams,
        utilities=utilities,
        options={
            "INPUT_UNIT_TEMPERATURE": "K",
            "INPUT_UNIT_DELTA_T": "K",
            "INPUT_UNIT_HTC": "W/m^2/K",
        },
    )
    z1 = site.subzones["Z1"]
    hot = next(s for s in z1.hot_streams if s.name == "H1")
    cold = next(s for s in z1.cold_streams if s.name == "C1")
    hot_utility = next(u for u in site.hot_utilities if u.name == "HU_K")
    cold_utility = next(u for u in site.cold_utilities if u.name == "CU_K")

    assert float(hot.t_supply) == pytest.approx(250.0)
    assert float(hot.t_target) == pytest.approx(100.0)
    assert float(cold.t_supply) == pytest.approx(30.0)
    assert float(cold.t_target) == pytest.approx(150.0)
    assert float(hot_utility.t_supply) == pytest.approx(300.0)
    assert float(hot_utility.t_target) == pytest.approx(250.0)
    assert float(hot_utility.dt_cont) == pytest.approx(10.0)
    assert float(hot_utility.htc) == pytest.approx(0.5)
    assert float(cold_utility.t_supply) == pytest.approx(20.0)
    assert float(cold_utility.t_target) == pytest.approx(40.0)
    assert float(cold_utility.dt_cont) == pytest.approx(10.0)
    assert float(cold_utility.htc) == pytest.approx(0.5)


def test_prepare_problem_preserves_period_dt_cont_and_htc_values():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "H_period",
                "zone": "Z1",
                "t_supply": {"values": [250.0, 260.0]},
                "t_target": {"values": [100.0, 120.0]},
                "heat_flow": {"values": [10_000.0, 11_000.0]},
                "dt_cont": {"values": [2.0, 3.0]},
                "htc": {"values": [400.0, 500.0], "unit": "W/m^2/K"},
            }
        )
    ]
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_period",
                "type": "Hot",
                "t_supply": {"values": [300.0, 310.0]},
                "t_target": {"values": [250.0, 255.0]},
                "heat_flow": {"values": [0.0, 0.0]},
                "dt_cont": {"values": [4.0, 5.0]},
                "htc": {"values": [600.0, 700.0], "unit": "W/m^2/K"},
            }
        )
    ]

    site = prepare_problem(streams=streams, utilities=utilities)
    stream = next(s for s in site.subzones["Z1"].hot_streams if s.name == "H_period")
    utility = next(u for u in site.hot_utilities if u.name == "HU_period")

    assert stream.dt_cont.period_values.tolist() == pytest.approx([2.0, 3.0])
    assert stream.htc.period_values.tolist() == pytest.approx([0.4, 0.5])
    assert utility.dt_cont.period_values.tolist() == pytest.approx([4.0, 5.0])
    assert utility.htc.period_values.tolist() == pytest.approx([0.6, 0.7])


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
    with pytest.raises(ValueError, match="must classify as Hot or Cold"):
        prepare_problem(streams=streams)


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


def test_htc_zero_stream_is_preserved():
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
    assert stream.htc[0] == 0.0
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
    assert names.count("Duplicate") == 2
    assert names.count("Duplicate_1") == 0


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
    assert hot_stream.dt_cont[0] == 10.0
    assert hot_stream.dt_cont_act[0] == 20.0
    assert hot_stream.t_min_star[0] == 80.0
    assert hot_utility.dt_cont[0] == 8.0
    assert hot_utility.dt_cont_act[0] == 16.0


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
    assert hot_a.dt_cont[0] == 10.0
    assert hot_a.dt_cont_act[0] == 20.0
    assert hot_b.dt_cont[0] == 10.0
    assert hot_b.dt_cont_act[0] == 5.0


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

    dt_cont = area.config.thermal.dt_cont
    assert float(hu.dt_cont[0]) == pytest.approx(dt_cont)
    assert float(hu.dt_cont_act[0]) == pytest.approx(dt_cont * 3.0)
    assert float(cu.dt_cont[0]) == pytest.approx(dt_cont)
    assert float(cu.dt_cont_act[0]) == pytest.approx(dt_cont * 3.0)


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

    assert float(hot_hps.t_supply[0]) == pytest.approx(280.0)
    assert float(hot_hps.t_target[0]) == pytest.approx(279.99)
    assert float(cold_hps.t_supply[0]) == pytest.approx(279.99)
    assert float(cold_hps.t_target[0]) == pytest.approx(280.0)


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

    assert float(hot_a.dt_cont[0]) == pytest.approx(0.0)
    assert float(hot_a.dt_cont_act[0]) == pytest.approx(0.0)


def test_period_process_extrema_use_all_periods_for_default_hot_utility():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "ColdA",
                "zone": "AreaA",
                "t_supply": {
                    "values": [40.0, 60.0],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [150.0, 200.0],
                    "unit": "degC",
                },
                "heat_flow": {
                    "values": [120.0, 160.0],
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
    options = {
        "PROBLEM_PERIOD_IDS": ["one", "two"],
        "PROBLEM_PERIOD_WEIGHTS": [1, 1],
    }

    site = prepare_problem(streams=streams, options=options)
    hu = next(utility for utility in site.hot_utilities if utility.name == "HU")

    assert float(hu.t_max_star[0]) == pytest.approx(210.0)
    assert float(hu.t_max_star[1]) == pytest.approx(210.0)


def test_period_process_extrema_use_selected_period_for_default_hot_utility():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "ColdA",
                "zone": "AreaA",
                "t_supply": {
                    "values": [40.0, 60.0],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [150.0, 200.0],
                    "unit": "degC",
                },
                "heat_flow": {
                    "values": [120.0, 160.0],
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

    assert float(hu.t_max_star[0]) == pytest.approx(210.0)
    assert float(hu.t_max_star[1]) == pytest.approx(210.0)


def test_period_hot_utility_sorting_uses_all_period_envelope(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_swing",
                "type": "Hot",
                "t_supply": {
                    "values": [250.0, 400.0],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [200.0, 350.0],
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
                    "unit": "degC",
                },
                "t_target": {
                    "values": [260.0, 270.0],
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


def test_period_hot_utility_sorting_uses_selected_period(dummy_streams):
    utilities = [
        UtilitySchema.model_validate(
            {
                "name": "HU_swing",
                "type": "Hot",
                "t_supply": {
                    "values": [250.0, 400.0],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [200.0, 350.0],
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
                    "unit": "degC",
                },
                "t_target": {
                    "values": [260.0, 270.0],
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

    assert [utility.name for utility in list(site.hot_utilities)[:2]] == [
        "HU_swing",
        "HU_flat",
    ]


def test_build_prepared_stream_collection_rejects_neutral_process_stream():
    master_zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    area = Zone(
        name="AreaA",
        type=ZT.P.value,
        config=master_zone.config,
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

    sc, _ = _build_prepared_stream_collection(master_zone, streams, [])

    assert len(sc.get_hot_process_streams()) == 0


def test_utility_temperature_helpers_reject_missing_and_inconsistent_inputs():
    missing_temperature = UtilitySchema.model_validate(
        {
            "name": "HU_missing",
            "type": "Hot",
            "t_supply": 200.0,
            "t_target": None,
            "heat_flow": 0.0,
            "dt_cont": 10.0,
            "price": 100.0,
            "htc": 1.0,
        }
    )

    with pytest.raises(ValueError, match="missing supply or target"):
        _utility_temperature_arrays(missing_temperature, Configuration())

    inconsistent = UtilitySchema.model_validate(
        {
            "name": "HU_swing",
            "type": "Hot",
            "t_supply": {"values": [300.0, 200.0], "unit": "degC"},
            "t_target": {"values": [250.0, 260.0], "unit": "degC"},
            "heat_flow": 0.0,
            "dt_cont": 10.0,
            "price": 100.0,
            "htc": 1.0,
        }
    )

    with pytest.raises(ValueError, match="cannot be oriented consistently"):
        _orient_utility_temperatures(inconsistent, ST.Hot.value, Configuration())


def test_build_prepared_stream_collection_rejects_unresolved_stream_zone():
    master_zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    stream = StreamSchema.model_validate(
        {
            "name": "H_missing",
            "zone": "Site/Missing",
            "t_supply": 200.0,
            "t_target": 100.0,
            "heat_flow": 10.0,
            "dt_cont": 10.0,
            "htc": 1.0,
        }
    )

    with pytest.warns(UserWarning, match="Subzone 'Site/Missing' not found."):
        with pytest.raises(ValueError, match="could not resolve zone"):
            _build_prepared_stream_collection(master_zone, [stream], [])


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

    assert float(area_a_hot.dt_cont_act[0]) == pytest.approx(
        float(area_a_hot.dt_cont[0]) * 2.0
    )
    assert root_hot_ref is area_a_hot
    assert float(root_hot_ref.dt_cont_act[0]) == pytest.approx(
        float(area_a_hot.dt_cont[0]) * 2.0
    )
    assert float(area_b_hot.dt_cont_act[0]) == pytest.approx(
        float(area_b_hot.dt_cont[0])
    )
    assert float(area_a_utility.dt_cont_act[0]) == pytest.approx(
        float(area_a_utility.dt_cont[0]) * 2.0
    )
    assert float(root_utility.dt_cont_act[0]) == pytest.approx(
        float(root_utility.dt_cont[0])
    )
    assert float(area_b_utility.dt_cont_act[0]) == pytest.approx(
        float(area_b_utility.dt_cont[0])
    )


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


def test_prepare_problem_allows_stream_zone_matching_project_name_without_warning():
    streams = [
        StreamSchema.model_validate(
            {
                "name": "HotStream",
                "zone": "Plant",
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
                "zone": "Plant",
                "t_supply": 50.0,
                "t_target": 150.0,
                "heat_flow": -400.0,
                "dt_cont": 10.0,
                "htc": 500.0,
            }
        ),
    ]

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        master_zone = prepare_problem(streams=streams, project_name="Plant")

    assert not [
        warning
        for warning in caught_warnings
        if "Subzone" in str(warning.message) and "not found" in str(warning.message)
    ]
    assert "Plant" in master_zone.subzones
    process_zone = master_zone.subzones["Plant"]
    assert {"O1", "O2"} <= set(process_zone.subzones)
    assert process_zone.subzones["O1"].dt_cont_multiplier == pytest.approx(1.0)
    assert process_zone.subzones["O2"].dt_cont_multiplier == pytest.approx(1.0)


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


def test_canonical_zone_helpers_reject_invalid_config_and_multiplier_edges():
    with pytest.raises(TypeError, match="options must be"):
        _build_zone_config(
            options=object(),
            top_zone_name="Site",
            top_zone_identifier="Site",
        )
    with pytest.raises(ValueError, match="dt_cont_multiplier"):
        _resolve_zone_dt_cont_multiplier(
            ZoneTreeSchema.model_construct(
                name="Bad",
                type="Zone",
                dt_cont_multiplier=-1.0,
                children=None,
            ),
            inherited_multiplier=None,
        )

    blank_type_name, blank_type = _get_validated_zone_info(
        ZoneTreeSchema(name="Blank", type=" "),
        project_name="Project",
        depth=0,
    )
    assert blank_type_name == "Blank"
    assert blank_type == ZT.S.value


def test_apply_zone_dt_cont_multiplier_skips_missing_child_zone():
    parent_zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    zone_tree = ZoneTreeSchema(
        name="Site",
        type="Zone",
        children=[
            ZoneTreeSchema(
                name="MissingChild",
                type="Zone",
                dt_cont_multiplier=2.0,
            )
        ],
    )

    with pytest.warns(UserWarning) as caught:
        out = _apply_zone_dt_cont_multiplier(parent_zone, zone_tree)

    messages = [str(warning.message) for warning in caught]
    assert any("Subzone 'MissingChild' not found." in message for message in messages)
    assert any("empty stream collection" in message for message in messages)
    assert out is parent_zone
    assert parent_zone.dt_cont_multiplier == pytest.approx(1.0)


def test_apply_zone_dt_cont_multiplier_applies_existing_child_zone():
    parent_zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    child_zone = Zone(
        name="AreaA",
        type=ZT.P.value,
        config=parent_zone.config,
        parent_zone=parent_zone,
    )
    parent_zone.add_zone(child_zone)
    zone_tree = ZoneTreeSchema(
        name="Site",
        type="Zone",
        children=[
            ZoneTreeSchema(
                name="AreaA",
                type="Zone",
                dt_cont_multiplier=2.0,
            )
        ],
    )

    with pytest.warns(UserWarning, match="empty stream collection"):
        out = _apply_zone_dt_cont_multiplier(parent_zone, zone_tree)

    assert out is parent_zone
    assert parent_zone.dt_cont_multiplier == pytest.approx(1.0)
    assert child_zone.dt_cont_multiplier == pytest.approx(2.0)


def test_rewrite_stream_zones_handles_empty_labels_and_root_name_collisions():
    tree_without_children = ZoneTreeSchema(name="Site", type="Zone", children=None)
    missing_zone = make_stream(zone="Area")
    missing_zone.zone = None
    blank_zone = make_stream(zone=" / ")

    _rewrite_stream_zones_from_tree(
        tree_without_children,
        [missing_zone, blank_zone],
    )

    assert tree_without_children.children == []
    assert missing_zone.zone is None
    assert blank_zone.zone == " / "

    collision_tree = ZoneTreeSchema(
        name="Site",
        type="Zone",
        children=[ZoneTreeSchema(name="TestStream", type=ZT.P.value)],
    )
    site_stream = make_stream(zone="Site")
    site_stream.name = "TestStream"

    _rewrite_stream_zones_from_tree(collision_tree, [site_stream])

    assert site_stream.zone == "Site/TestStream_2"
    assert {child.name for child in collision_tree.children} == {
        "TestStream",
        "TestStream_2",
    }


def test_generated_zone_tree_handles_non_string_root_and_empty_zone_path():
    stream = make_stream(zone="/")

    normalised = _validate_zone_tree_structure(
        None,
        [stream],
        top_zone_name=object(),
    )

    assert normalised.name == ZT.S.value
    assert stream.zone == f"{ZT.S.value}/O1"


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
    dummy_config.costing.annual_op_time = 0
    out_config = _validate_config_data_completed(dummy_config)
    assert out_config.costing.annual_op_time > 0
    assert not hasattr(out_config, "COSTING_ANNUAL_OP_TIME")
    assert not hasattr(out_config, "ANNUAL_OP_TIME")


def test_turbine_pressure_is_clamped_using_canonical_config_fields(dummy_config):
    dummy_config.power.turbine_work_enabled = True
    dummy_config.power.turb_p_in = 250.0

    out_config = _validate_config_data_completed(dummy_config)

    assert out_config.power.turb_p_in == 200
    assert not hasattr(out_config, "POWER_TURB_P_IN")
    assert not hasattr(out_config, "TURB_P_IN")


def test_thermal_repairs_use_current_config_fields_without_legacy_attrs(dummy_config):
    dummy_config.thermal.dt_phase_change = 0
    dummy_config.thermal.dt_cont = -5.0

    out_config = _validate_config_data_completed(dummy_config)

    assert out_config.thermal.dt_phase_change == 0.01
    assert not hasattr(out_config, "THERMAL_DT_PHASE_CHANGE")
    assert out_config.thermal.dt_cont == 0.0
    assert not hasattr(out_config, "THERMAL_DT_CONT")
    assert not hasattr(out_config, "DT_PHASE_CHANGE")
    assert not hasattr(out_config, "DT_CONT")


def test_prepare_problem_preserves_passed_configuration_object(dummy_streams):
    cfg = Configuration(
        options={
            "PROBLEM_TOP_ZONE_NAME": "Original Site",
            "POWER_TURBINE_WORK_ENABLED": True,
            "POWER_TURB_P_IN": 12.0,
            "POWER_TURB_T_IN": 375.0,
            "POWER_HIGH_P_COND_FLASH_ENABLED": True,
        }
    )

    site = prepare_problem(
        streams=dummy_streams,
        options=cfg,
        project_name="Runtime Site",
    )

    assert site.config is not cfg
    assert site.config.problem.top_zone_name == "Runtime Site"
    assert not hasattr(site.config, "PROBLEM_TOP_ZONE_NAME")
    assert not hasattr(site.config, "PROBLEM_TOP_ZONE_IDENTIFIER")
    assert site.config.power.turbine_work_enabled is True
    assert site.config.power.turb_p_in == 12.0
    assert site.config.power.turb_t_in == 375.0
    assert site.config.power.high_p_cond_flash_enabled is True


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
        problem = type(
            "ProblemConfig",
            (),
            {
                "top_zone_name": "MySite",
                "top_zone_identifier": ZT.S.value,
            },
        )()

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
    config = Configuration()
    zone_tree = make_zone_tree_schema()
    master_zone = Zone(
        name=config.problem.top_zone_name,
        type=config.problem.top_zone_identifier,
        config=config,
    )

    result = _create_nested_zones(master_zone, zone_tree, config)

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
    config = Configuration()
    zone_tree = ZoneTreeSchema(name="MySite", type=ZT.S.value, children=None)
    parent_zone = Zone(name="MySite", type=ZT.S.value, config=config)

    result = _create_nested_zones(parent_zone, zone_tree, config)

    assert result == parent_zone
    assert result.subzones == {}


def test_assign_process_streams_to_subzones_requires_zone_mapping():
    master_zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
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


def test_assign_process_streams_to_subzones_rejects_unresolved_and_neutral_streams():
    master_zone = Zone(name="Site", type=ZT.S.value, config=Configuration())
    process_streams = StreamCollection()
    hot_stream = Stream(
        name="H1",
        t_supply=200.0,
        t_target=100.0,
        heat_flow=10.0,
        dt_cont=5.0,
        htc=1.0,
        is_process_stream=True,
    )
    process_streams.add(hot_stream, key="Site.Hot Stream.H1")

    with pytest.warns(UserWarning, match="Subzone 'Site/Missing' not found."):
        with pytest.raises(ValueError, match="could not resolve zone"):
            _assign_process_streams_to_subzones(
                master_zone=master_zone,
                process_streams=process_streams,
                process_zone_paths={"Site.Hot Stream.H1": "Site/Missing"},
            )

    area = Zone(
        name="AreaA",
        type=ZT.P.value,
        config=master_zone.config,
        parent_zone=master_zone,
    )
    master_zone.add_zone(area)
    neutral_streams = StreamCollection()
    neutral_streams.add(
        Stream(
            name="NeutralA",
            t_supply=100.0,
            t_target=100.0,
            heat_flow=0.0,
            dt_cont=5.0,
            htc=1.0,
            is_process_stream=True,
        ),
        key="Site.AreaA.NeutralA",
    )

    with pytest.raises(ValueError, match="must classify as Hot or Cold"):
        _assign_process_streams_to_subzones(
            master_zone=master_zone,
            process_streams=neutral_streams,
            process_zone_paths={"Site.AreaA.NeutralA": "Site/AreaA"},
        )

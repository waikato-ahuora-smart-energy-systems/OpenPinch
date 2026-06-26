"""Regression tests for public I/O schema defaults."""

import pytest
from pydantic import ValidationError

from OpenPinch.classes.value import Value
from OpenPinch.lib.enums import FluidPhase
from OpenPinch.lib.schemas.common import (
    PeriodValueWithUnit,
    ValueWithUnit,
)
from OpenPinch.lib.schemas.io import StreamSchema, TargetInput, UtilitySchema
from OpenPinch.lib.schemas.reporting import PinchTemp, TargetResults


def test_target_input_utilities_default_is_not_shared():
    first = TargetInput(streams=[])
    second = TargetInput(streams=[])

    first.utilities.append(
        UtilitySchema(
            name="Steam",
            type="Hot",
            t_supply=200.0,
            t_target=180.0,
            heat_flow=50.0,
        )
    )

    assert len(first.utilities) == 1
    assert second.utilities == []


def test_target_input_accepts_period_value_payloads():
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "H1",
                "t_supply": {
                    "values": [150.0, 140.0],
                    "unit": "degC",
                },
                "t_target": {
                    "values": [60.0, 55.0],
                    "unit": "degC",
                },
                "heat_flow": {
                    "values": [100.0, 90.0],
                    "unit": "kW",
                },
            }
        ]
    }

    validated = TargetInput.model_validate(payload)
    t_supply = validated.streams[0].t_supply
    t_target = validated.streams[0].t_target

    assert isinstance(t_supply, PeriodValueWithUnit)
    assert t_supply.values == [150.0, 140.0]
    assert t_supply.unit == "degC"
    assert isinstance(t_target, PeriodValueWithUnit)
    assert t_target.unit == "degC"


def test_target_input_accepts_stream_fluid_pressure_and_enthalpy_fields():
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "H1",
                "t_supply": 150.0,
                "t_target": 60.0,
                "p_supply": {"value": 2.0, "unit": "bar"},
                "p_target": {"value": 150.0, "unit": "kPa"},
                "h_supply": {"value": 2_800.0, "unit": "kJ/kg"},
                "h_target": {"value": 2_300_000.0, "unit": "J/kg"},
                "heat_flow": 100.0,
                "fluid_name": "HEOS::Water",
                "fluid_phase": "GAS",
            }
        ],
        "utilities": [
            {
                "name": "Steam",
                "type": "Hot",
                "t_supply": 180.0,
                "t_target": 160.0,
                "p_supply": 800.0,
                "p_target": 700.0,
                "h_supply": 2_750.0,
                "h_target": 700.0,
                "fluid_name": "Water",
                "fluid_phase": "vle",
            }
        ],
    }

    validated = TargetInput.model_validate(payload)
    stream = validated.streams[0]
    utility = validated.utilities[0]

    assert isinstance(stream.p_supply, ValueWithUnit)
    assert isinstance(stream.h_target, ValueWithUnit)
    assert stream.name == "H1"
    assert stream.fluid_name == "HEOS::Water"
    assert stream.fluid_phase == "gas"
    assert utility.p_supply == pytest.approx(800.0)
    assert utility.fluid_name == "Water"
    assert utility.fluid_phase == "vapour-liquid equilibrium"


@pytest.mark.parametrize(
    "alias_name",
    ["stream_name", "heat_capacity_flow_rate", "flow_heat_capacity"],
)
def test_stream_schema_rejects_retired_aliases(alias_name):
    payload = {
        "zone": "Zone A",
        "name": "H1",
        "t_supply": 150.0,
        "t_target": 60.0,
        "heat_flow": 100.0,
    }
    if alias_name == "stream_name":
        payload.pop("name")
        payload[alias_name] = "H1"
    else:
        payload[alias_name] = 10.0

    with pytest.raises(ValidationError):
        StreamSchema.model_validate(payload)


def test_target_input_accepts_fluid_phase_enum_instances():
    validated = TargetInput(
        streams=[
            StreamSchema(
                zone="Zone A",
                name="H1",
                t_supply=150.0,
                t_target=60.0,
                heat_flow=100.0,
                fluid_phase=FluidPhase.gas,
            )
        ]
    )

    assert validated.streams[0].fluid_phase == "gas"


def test_target_input_accepts_fluid_phase_description_strings():
    validated = TargetInput(
        streams=[
            StreamSchema(
                zone="Zone A",
                name="H1",
                t_supply=150.0,
                t_target=60.0,
                heat_flow=100.0,
                fluid_phase="liquid",
            )
        ]
    )

    assert validated.streams[0].fluid_phase == "liquid"


def test_target_input_accepts_standalone_vapour_phase_aliases():
    for phase in ("vapour", "vapor"):
        payload = {
            "streams": [
                {
                    "zone": "Zone A",
                    "name": "H1",
                    "t_supply": 100.0,
                    "t_target": 99.5,
                    "heat_flow": 100.0,
                    "fluid_name": "Water",
                    "fluid_phase": phase,
                }
            ],
            "utilities": [],
        }

        validated = TargetInput.model_validate(payload)

        assert validated.streams[0].fluid_phase == "vapour"


def test_target_input_rejects_invalid_stream_fluid_phase():
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "H1",
                "t_supply": 150.0,
                "t_target": 60.0,
                "heat_flow": 100.0,
                "fluid_phase": "plasma",
            }
        ]
    }

    with pytest.raises(ValidationError):
        TargetInput.model_validate(payload)


def test_target_input_defers_fluid_name_validation_to_stream_import():
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "H1",
                "t_supply": 150.0,
                "t_target": 60.0,
                "heat_flow": 100.0,
                "fluid_name": "NotARealCoolPropFluid",
            }
        ]
    }

    validated = TargetInput.model_validate(payload)

    assert validated.streams[0].fluid_name == "NotARealCoolPropFluid"


def test_target_input_rejects_legacy_units_alias():
    payload = {
        "streams": [
            {
                "zone": "Zone A",
                "name": "H1",
                "t_supply": {"value": 150.0, "units": "degC"},
                "t_target": {"value": 60.0, "unit": "degC"},
                "heat_flow": {"value": 100.0, "unit": "kW"},
            }
        ]
    }

    with pytest.raises(ValidationError):
        TargetInput.model_validate(payload)


def test_target_results_accept_unit_aware_hpr_scalar_and_array_metrics():
    results = TargetResults(
        name="Plant/Direct Heat Pump",
        Qh=ValueWithUnit(value=100.0, unit="kW"),
        Qc=ValueWithUnit(value=50.0, unit="kW"),
        Qr=ValueWithUnit(value=25.0, unit="kW"),
        pinch_temp=PinchTemp(),
        hpr_utility_total=PeriodValueWithUnit(values=[10.0, 12.0], unit="kW"),
        hpr_cop=ValueWithUnit(value=3.5, unit="-"),
        num_units=4,
        hpr_success=True,
    )

    assert isinstance(results.Qh, Value)
    assert results.Qh.value == pytest.approx(100.0)
    assert results.Qh.unit == "kW"
    assert isinstance(results.hpr_utility_total, Value)
    assert results.hpr_utility_total.values == [10.0, 12.0]
    assert results.hpr_utility_total.unit == "kW"
    assert isinstance(results.hpr_cop, Value)
    assert results.hpr_cop.value == pytest.approx(3.5)
    assert results.hpr_cop.unit == "-"
    assert results.num_units == 4
    assert results.hpr_success is True

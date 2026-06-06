"""Regression tests for public I/O schema defaults."""

import pytest

from OpenPinch.classes.value import Value
from OpenPinch.lib.schemas.common import (
    StatefulValueWithUnit,
    ValueWithUnit,
)
from OpenPinch.lib.schemas.io import (
    GetInputOutputData,
    ProblemTableDataSchema,
    StreamSchema,
    TargetInput,
    THSchema,
    UtilitySchema,
)
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


def test_get_input_output_data_options_default_is_not_shared():
    profile = ProblemTableDataSchema(name="Plant", data=THSchema(T=[100.0]))
    stream = StreamSchema(
        zone="Zone A",
        name="H1",
        t_supply=150.0,
        t_target=60.0,
        heat_flow=100.0,
    )

    first = GetInputOutputData(plant_profile_data=[profile], streams=[stream])
    second = GetInputOutputData(plant_profile_data=[profile], streams=[stream])

    first.options["mode"] = "baseline"

    assert first.options == {"mode": "baseline"}
    assert second.options == {}


def test_target_input_accepts_stateful_value_payloads():
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

    assert isinstance(t_supply, StatefulValueWithUnit)
    assert t_supply.values == [150.0, 140.0]
    assert t_supply.unit == "degC"
    assert isinstance(t_target, StatefulValueWithUnit)
    assert t_target.unit == "degC"


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

    with pytest.raises(Exception):
        TargetInput.model_validate(payload)


def test_target_results_accept_unit_aware_hpr_scalar_and_array_metrics():
    results = TargetResults(
        name="Plant/Direct Heat Pump",
        Qh=ValueWithUnit(value=100.0, unit="kW"),
        Qc=ValueWithUnit(value=50.0, unit="kW"),
        Qr=ValueWithUnit(value=25.0, unit="kW"),
        pinch_temp=PinchTemp(),
        hpr_utility_total=StatefulValueWithUnit(values=[10.0, 12.0], unit="kW"),
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

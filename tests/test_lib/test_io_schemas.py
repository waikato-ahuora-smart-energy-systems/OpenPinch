"""Regression tests for public I/O schema defaults."""

import pytest

from OpenPinch.lib.schemas.common import StatefulValueWithUnit
from OpenPinch.lib.schemas.io import (
    GetInputOutputData,
    ProblemTableDataSchema,
    StreamSchema,
    TargetInput,
    THSchema,
    UtilitySchema,
)


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

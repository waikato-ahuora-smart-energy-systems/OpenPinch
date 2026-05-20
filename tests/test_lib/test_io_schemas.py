"""Regression tests for public I/O schema defaults."""

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

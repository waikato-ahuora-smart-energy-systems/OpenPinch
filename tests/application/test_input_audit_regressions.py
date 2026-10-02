"""Regression tests for the input, unit and configuration audit fixes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from OpenPinch import PinchProblem
from OpenPinch.adapters.io.csv import get_problem_from_csv
from OpenPinch.application._problem.input import semantics, validation
from OpenPinch.contracts.input import TargetInput
from OpenPinch.domain.configuration_fields import validate_configuration_options
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.value import Value


def _stream(**overrides) -> dict:
    stream = {
        "zone": "Zone A",
        "name": "H1",
        "t_supply": 150.0,
        "t_target": 60.0,
        "heat_flow": 100.0,
        "dt_cont": 10.0,
        "htc": 1.0,
    }
    return stream | overrides


def _messages(payload: dict) -> list[tuple[str, str]]:
    problem = TargetInput.model_validate(payload)
    return [
        (issue.severity, issue.message)
        for issue in semantics.semantic_issues(problem, context={})
    ]


def test_null_utility_price_uses_the_default_price_without_warning():
    # Null and absent prices both mean "not given" (OpenPinch's own exports
    # write null), so neither is flagged.
    payload = {
        "streams": [_stream()],
        "utilities": [
            {"name": "Steam", "type": "Hot", "t_supply": 180.0, "price": None}
        ],
        "options": {"COSTING_UTILITY_PRICE": 50.0},
    }

    problem = PinchProblem(payload)
    steam = next(u for u in problem.hot_utilities if u.name == "Steam")

    assert float(steam.price) == pytest.approx(50.0)
    report = validation.build_validation_report(payload)
    assert not any("COSTING_UTILITY_PRICE" in issue.message for issue in report.issues)


@pytest.mark.parametrize("field", ["dt_cont", "htc"])
def test_null_value_objects_behave_like_absent_stream_fields(field):
    absent = TargetInput.model_validate({"streams": [_stream()], "utilities": []})
    nulled = TargetInput.model_validate(
        {
            "streams": [_stream(**{field: {"value": None, "unit": "K"}})],
            "utilities": [],
        }
    )

    assert getattr(nulled.streams[0], field) is None
    assert getattr(absent.streams[0], field) is not None


def test_null_required_stream_field_is_rejected():
    with pytest.raises(ValueError, match="require t_supply"):
        TargetInput.model_validate(
            {
                "streams": [_stream(t_supply={"value": None, "unit": "degC"})],
                "utilities": [],
            }
        )


def test_mismatched_period_lengths_are_reported_not_raised():
    payload = {
        "streams": [
            _stream(
                t_supply={"values": [150.0, 140.0], "unit": "degC"},
                t_target={"values": [60.0, 50.0], "unit": "degC"},
                heat_flow={"values": [100.0, 90.0, 80.0], "unit": "kW"},
            )
        ],
        "utilities": [],
    }

    report = validation.build_validation_report(payload)

    assert report.valid is False
    assert any("period values" in issue.message for issue in report.issues)


def test_period_lengths_must_match_declared_period_ids():
    messages = _messages(
        {
            "streams": [_stream(heat_flow={"values": [100.0, 90.0], "unit": "kW"})],
            "utilities": [],
            "options": {"PROBLEM_PERIOD_IDS": ["a", "b", "c"]},
        }
    )

    assert any("declares 3 periods" in message for _, message in messages)


def test_temperatures_below_absolute_zero_are_rejected():
    messages = _messages({"streams": [_stream(t_target=-300.0)], "utilities": []})

    assert any(
        severity == "error" and message.startswith("Temperature is below absolute zero")
        for severity, message in messages
    )


def test_duplicate_stream_names_are_renamed_with_a_warning():
    messages = _messages(
        {"streams": [_stream(), _stream(t_supply=140.0)], "utilities": []}
    )

    assert any(
        severity == "warning" and "used more than once" in message
        for severity, message in messages
    )


def test_construction_compares_stream_temperatures_in_canonical_units():
    payload = {
        "streams": [
            _stream(
                t_supply={"value": 100.0, "unit": "K"},
                t_target={"value": 100.0, "unit": "degC"},
            )
        ],
        "utilities": [],
    }

    problem = PinchProblem(payload)

    assert [stream.name for stream in problem.cold_streams] == ["H1"]


def test_zero_dt_cont_multiplier_is_kept():
    stream = Stream(
        supply_temperature=150.0,
        target_temperature=60.0,
        heat_flow=100.0,
        delta_t_contribution=10.0,
        delta_t_contribution_multiplier=0.0,
    )

    assert stream.delta_t_contribution_multiplier == 0.0


def test_offset_temperatures_convert_to_differences_with_scale():
    assert Value(Value(18.0, "degF"), "delta_degC").value == pytest.approx(10.0)
    assert Value({"value": 10.0, "unit": "degF"}, "degC").value == pytest.approx(
        -12.2222222, rel=1e-6
    )


def test_value_weights_are_validated_and_serialised():
    with pytest.raises(ValueError, match="positive sum"):
        Value({"values": [1.0, 2.0], "weights": [0.0, 0.0], "unit": "kW"})
    with pytest.raises(ValueError, match="non-negative"):
        Value({"values": [1.0, 2.0], "weights": [-1.0, 2.0], "unit": "kW"})

    value = Value({"values": [1.0, 2.0], "weights": [1.0, 3.0], "unit": "kW"})
    round_tripped = Value(value.to_dict())

    np.testing.assert_allclose(round_tripped.weights, [1.0, 3.0])


@pytest.mark.parametrize(
    "options, message",
    [
        ({"PROBLEM_PERIOD_IDS": ["a", "a"]}, "unique"),
        ({"PROBLEM_PERIOD_IDS": ["a", " "]}, "empty"),
        (
            {"PROBLEM_PERIOD_IDS": ["a"], "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5]},
            "2 values for 1 periods",
        ),
        ({"COSTING_ANNUAL_OP_TIME": 0.0}, "greater than 0"),
        ({"POWER_ETA_MECH": 0.0}, "greater than 0"),
        ({"HPR_ETA_COMP": 0.0}, "greater than 0"),
    ],
)
def test_configuration_rejects_degenerate_values(options, message):
    with pytest.raises(ValueError, match=message):
        validate_configuration_options(options)


def _write_csv(path: Path, rows: list[list]) -> None:
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_csv_keeps_superscript_units_and_text_names(tmp_path: Path):
    streams_csv = tmp_path / "streams.csv"
    utilities_csv = tmp_path / "utilities.csv"
    _write_csv(
        streams_csv,
        [
            ["ignored"] * 7,
            [None, None, "degC", "degC", "kW", "K", "kW/m^2/K"],
            ["Zone A", "007", 150.0, 60.0, 100.0, 10.0, 0.5],
        ],
    )
    _write_csv(
        utilities_csv,
        [
            ["ignored"] * 8,
            [None, None, "degC", "degC", "K", "$/MWh", "kW/m2/K", "kW"],
            ["1", "Hot", 260.0, 210.0, 10.0, 60.0, 0.9, 70.0],
        ],
    )

    with pytest.warns(UserWarning, match="'007' was changed to 'S007'"):
        out = get_problem_from_csv(streams_csv, utilities_csv, output_json=None)

    assert out["streams"][0]["name"] == "S007"
    assert out["streams"][0]["htc"]["unit"] == "kW/m^2/K"
    assert out["utilities"][0]["name"] == "1"


def test_absent_stream_dt_cont_uses_thermal_dt_cont():
    stream = _stream()
    del stream["dt_cont"]
    problem = PinchProblem(
        {"streams": [stream], "utilities": [], "options": {"THERMAL_DT_CONT": 7.0}}
    )

    assert float(problem.hot_streams[0].delta_t_contribution) == pytest.approx(7.0)


def test_absent_utility_price_uses_default_price_without_warning():
    payload = {
        "streams": [_stream()],
        "utilities": [{"name": "Steam", "type": "Hot", "t_supply": 180.0}],
        "options": {"COSTING_UTILITY_PRICE": 50.0},
    }

    problem = PinchProblem(payload)
    steam = next(u for u in problem.hot_utilities if u.name == "Steam")

    assert float(steam.price) == pytest.approx(50.0)
    report = validation.build_validation_report(payload)
    assert not any("COSTING_UTILITY_PRICE" in issue.message for issue in report.issues)


@pytest.mark.parametrize(
    "dt_cont, expected",
    [
        (5.0, 10.0),  # the THERMAL_DT_CONT default gives exactly dTmin / 2
        (2.0, 4.0),  # other contributions keep their ratio to the default
        (0.0, 10.0),  # an unset contribution falls back to dTmin / 2
    ],
)
def test_hen_dtmin_tier_scales_every_stream_contribution(dt_cont, expected):
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        _temperature_contribution,
    )

    stream = Stream(
        supply_temperature=200.0,
        target_temperature=100.0,
        heat_flow=100.0,
        delta_t_contribution=dt_cont,
    )

    assert _temperature_contribution(
        stream, 20.0, reference_dt_cont=5.0
    ) == pytest.approx(expected)


def test_absent_utility_dt_cont_uses_thermal_dt_cont():
    payload = {
        "streams": [_stream()],
        "utilities": [
            {"name": "Steam", "type": "Hot", "t_supply": 180.0, "price": 30.0}
        ],
        "options": {"THERMAL_DT_CONT": 7.0},
    }

    problem = PinchProblem(payload)
    steam = next(u for u in problem.hot_utilities if u.name == "Steam")

    assert float(steam.delta_t_contribution) == pytest.approx(7.0)

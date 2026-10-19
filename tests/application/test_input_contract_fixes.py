"""Regression tests for the input, contract and export audit fixes (WP9)."""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application._problem.input import validation


def _stream(**overrides) -> dict:
    stream = {
        "zone": "Zone A",
        "name": "H1",
        "t_supply": 150.0,
        "t_target": 60.0,
        "heat_flow": 180.0,
        "dt_cont": 10.0,
        "htc": 1.0,
    }
    return {key: value for key, value in (stream | overrides).items() if value != ...}


def _payload(*streams: dict) -> dict:
    return {
        "streams": list(streams),
        "utilities": [
            {"name": "HU", "type": "Hot", "t_supply": 250.0, "t_target": 249.0},
            {"name": "CU", "type": "Cold", "t_supply": 15.0, "t_target": 20.0},
        ],
    }


def test_heat_capacity_flowrate_gives_heat_flow_for_hot_and_cold_streams():
    problem = PinchProblem(
        _payload(
            _stream(heat_flow=..., heat_capacity_flowrate=2.0),
            _stream(
                name="C1",
                t_supply=40.0,
                t_target=120.0,
                heat_flow=...,
                heat_capacity_flowrate=1.5,
            ),
        )
    )

    assert float(problem.hot_streams[0].heat_flow) == pytest.approx(180.0)
    assert float(problem.cold_streams[0].heat_flow) == pytest.approx(120.0)


def test_heat_capacity_flowrate_converts_temperature_and_cp_units():
    problem = PinchProblem(
        _payload(
            _stream(
                t_supply={"value": 302.0, "unit": "degF"},
                t_target={"value": 140.0, "unit": "degF"},
                heat_flow=...,
                heat_capacity_flowrate={"value": 2000.0, "unit": "W/K"},
            )
        )
    )

    # 162 degF is a 90 K span.
    assert float(problem.hot_streams[0].heat_flow) == pytest.approx(180.0)


def test_heat_capacity_flowrate_per_period_values():
    payload = _payload(
        _stream(
            t_supply={"values": [150.0, 150.0], "unit": "degC"},
            t_target={"values": [60.0, 100.0], "unit": "degC"},
            heat_flow=...,
            heat_capacity_flowrate={"values": [2.0, 1.0], "unit": "kW/K"},
        )
    )
    payload["options"] = {"PROBLEM_PERIOD_IDS": ["a", "b"]}

    problem = PinchProblem(payload)

    assert list(problem.hot_streams[0].heat_flow.period_values) == pytest.approx(
        [180.0, 50.0]
    )


def test_disagreeing_heat_flow_and_heat_capacity_flowrate_are_reported():
    report = validation.build_validation_report(
        _payload(_stream(heat_capacity_flowrate=3.0))
    )

    assert report.valid is False
    issue = next(
        issue for issue in report.issues if issue.field == "heat_capacity_flowrate"
    )
    assert issue.severity == "error"
    assert "270" in issue.message and "180" in issue.message


def test_agreeing_heat_flow_and_heat_capacity_flowrate_are_accepted():
    report = validation.build_validation_report(
        _payload(_stream(heat_capacity_flowrate=2.0))
    )

    assert report.valid is True


def test_stream_without_heat_flow_or_heat_capacity_flowrate_is_rejected():
    report = validation.build_validation_report(_payload(_stream(heat_flow=...)))

    assert report.valid is False
    assert any("heat_flow" in issue.message for issue in report.issues)

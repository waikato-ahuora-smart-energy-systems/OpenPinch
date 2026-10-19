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


def test_compare_to_matches_rows_across_project_names():
    base = PinchProblem(source="basic_pinch.json", project_name="Base")
    other = PinchProblem(source="basic_pinch.json", project_name="Scenario")
    base.target.direct_heat_integration()
    other.target.direct_heat_integration()

    comparison = base.compare_to(other)

    assert comparison.loc["Change", "Hot Utility Target"] == 0.0
    assert comparison.loc["Scenario", "Scope"] == "Scenario"


def test_project_name_may_contain_a_slash():
    problem = PinchProblem(source="basic_pinch.json", project_name="Site A/2026")
    reference = PinchProblem(source="basic_pinch.json")

    problem.target.direct_heat_integration()
    reference.target.direct_heat_integration()

    assert problem.project_name == "Site A/2026"
    assert problem.summary_frame()["Hot Utility Target"].tolist() == (
        reference.summary_frame()["Hot Utility Target"].tolist()
    )


def _summary(*scopes: str):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "Scope": scope,
                "Zone Type": "Site" if "/" not in scope else "Process Zone",
                "Integration Type": "Process",
                "Target Method": "Heat Exchange",
                "Hot Utility Target": 10.0,
                "Hot Utility Target (unit)": "kW",
            }
            for scope in scopes
        ]
    )


@pytest.mark.parametrize(
    ("other_root", "other_scopes"),
    [
        # A zone-only run of a same-named problem (tutorial 14).
        ("Site", ("Site/Bleaching",)),
        ("Other", ("Other/Bleaching",)),
        ("Other", ("Other", "Other/Bleaching")),
    ],
)
def test_compare_summaries_maps_a_zone_scope_by_project_name(other_root, other_scopes):
    from OpenPinch.application._problem.output.reporting import (
        compare_problem_summaries,
    )

    comparison = compare_problem_summaries(
        _summary("Site", "Site/Bleaching"),
        _summary(*other_scopes),
        scope="Site/Bleaching",
        zone_type="Process Zone",
        integration_type=None,
        target_method=None,
        base_label="Base case",
        other_label="Scenario",
        base_root="Site",
        other_root=other_root,
    )

    assert comparison.loc["Scenario", "Scope"] == f"{other_root}/Bleaching"


def test_scalar_cp_agrees_with_per_period_heat_flow():
    payload = _payload(
        _stream(
            heat_flow={"values": [180.0, 180.0], "unit": "kW"},
            heat_capacity_flowrate=2.0,
        )
    )
    payload["options"] = {"PROBLEM_PERIOD_IDS": ["a", "b"]}

    report = validation.build_validation_report(payload)

    assert report.valid is True


def test_negative_heat_capacity_flowrate_is_rejected():
    report = validation.build_validation_report(
        _payload(_stream(heat_flow=..., heat_capacity_flowrate=-2.0))
    )

    assert report.valid is False
    assert any(
        issue.field == "heat_capacity_flowrate" and issue.severity == "error"
        for issue in report.issues
    )


def test_compare_to_uses_zone_tree_root_names():
    def problem(project_name: str) -> PinchProblem:
        payload = _payload(
            _stream(zone="Site/Plant"),
            _stream(name="C1", zone="Site/Plant", t_supply=40.0, t_target=120.0),
        )
        payload["zone_tree"] = {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Plant", "type": "Process Zone", "children": None}],
        }
        result = PinchProblem(payload, project_name=project_name)
        result.target.direct_heat_integration()
        return result

    comparison = problem("Site").compare_to(problem("Scenario"))

    assert comparison.loc["Change", "Hot Utility Target"] == pytest.approx(0.0)

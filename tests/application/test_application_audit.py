"""Regression tests for the application and reporting audit fixes (area 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.application._problem.periods.aggregation import (
    weighted_average_output,
)
from OpenPinch.contracts.output import TargetOutput
from OpenPinch.contracts.reporting import HeatUtility, PinchTemp, TargetResults
from OpenPinch.domain.value import Value
from OpenPinch.resources import read_sample_case


def _row(*, period_id: str, method: str = "Heat Exchange", **fields) -> TargetResults:
    values = {
        "scope": "Site",
        "zone_type": "Site",
        "integration_type": "Process",
        "target_method": method,
        "period_id": period_id,
        "Qh": Value(100.0, "kW"),
        "Qc": Value(50.0, "kW"),
        "Qr": Value(200.0, "kW"),
        "pinch_temp": PinchTemp(),
        "hot_utilities": [HeatUtility(name="Steam", heat_flow=Value(100.0, "kW"))],
        "cold_utilities": [HeatUtility(name="CW", heat_flow=Value(50.0, "kW"))],
    }
    return TargetResults(**(values | fields))


def _hpr_row(*, period_id: str, work: float, cop: float, **fields) -> TargetResults:
    return _row(
        period_id=period_id,
        method="Heat Pump",
        hpr_cycle="Carnot",
        hpr_work=Value(work, "kW"),
        hpr_cop=Value(cop, "dimensionless"),
        hpr_operating_cost=Value(10.0 * work, "$/y"),
        hpr_capital_cost=Value(1000.0 * work, "$"),
        hpr_annualized_capital_cost=Value(100.0 * work, "$/y"),
        hpr_success=True,
        **fields,
    )


# 4.1 Zero-load HPR period


def test_zero_load_period_counts_as_no_heat_pump():
    winter = TargetOutput(
        name="Site",
        period_id="winter",
        targets=[
            _row(period_id="winter"),
            _hpr_row(period_id="winter", work=10.0, cop=4.0),
        ],
    )
    summer = TargetOutput(  # no HPR row: the HPR load was zero
        name="Site", period_id="summer", targets=[_row(period_id="summer")]
    )

    output = weighted_average_output([winter, summer], [1.0, 1.0])

    hpr = next(t for t in output.targets if t.target_method == "Heat Pump")
    assert hpr.hpr_work.value == pytest.approx(5.0)
    assert hpr.hpr_capital_cost.value == pytest.approx(10000.0)  # peak period
    assert hpr.hpr_cop.value == pytest.approx(4.0)  # only the running period
    assert hpr.hpr_success is True


# 4.3 Physical aggregation


def test_seasonal_cop_is_total_heat_over_total_work():
    # 3 kW at COP 3 (9 kW heat) and 1 kW at COP 8 (8 kW heat): 17 / 4 = 4.25,
    # not the 5.5 average of the two COPs.
    periods = [
        TargetOutput(
            name="Site",
            period_id=pid,
            targets=[_row(period_id=pid), _hpr_row(period_id=pid, work=w, cop=c)],
        )
        for pid, w, c in (("a", 3.0, 3.0), ("b", 1.0, 8.0))
    ]

    output = weighted_average_output(periods, [1.0, 1.0])

    hpr = next(t for t in output.targets if t.target_method == "Heat Pump")
    assert hpr.hpr_cop.value == pytest.approx(17.0 / 4.0)


def test_area_and_exchanger_capital_take_the_peak_period():
    periods = [
        TargetOutput(
            name="Site",
            period_id=pid,
            targets=[
                _row(
                    period_id=pid,
                    area=Value(area, "m^2"),
                    num_units=units,
                    capital_cost=Value(10.0 * area, "$"),
                )
            ],
        )
        for pid, area, units in (("a", 100.0, 4.0), ("b", 300.0, 6.0))
    ]

    target = weighted_average_output(periods, [3.0, 1.0]).targets[0]

    assert target.area.value == pytest.approx(300.0)
    assert target.num_units == pytest.approx(6.0)
    assert target.capital_cost.value == pytest.approx(3000.0)


def test_one_failed_period_makes_hpr_success_false():
    periods = [
        TargetOutput(
            name="Site",
            period_id=pid,
            targets=[
                _row(period_id=pid),
                _hpr_row(period_id=pid, work=1.0, cop=3.0).model_copy(
                    update={"hpr_success": ok}
                ),
            ],
        )
        for pid, ok in (("a", True), ("b", False))
    ]

    hpr = next(
        t
        for t in weighted_average_output(periods, [1.0, 1.0]).targets
        if t.target_method == "Heat Pump"
    )
    assert hpr.hpr_success is False


# 4.2 and 4.6 Workspace


def _basic_payload() -> dict:
    return json.loads(read_sample_case("basic_pinch.json"))


def test_adding_a_case_keeps_solved_baseline_results():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    baseline = workspace.case("baseline")
    baseline.target.direct_heat_integration()
    other = PinchProblem(_basic_payload(), project_name="Other")

    workspace.add(other, name="alt")

    assert workspace.project_name == "Demo"
    assert workspace.case("baseline") is baseline
    assert baseline.results is not None


def test_scenario_refuses_an_existing_name_and_is_atomic():
    workspace = PinchWorkspace(_basic_payload(), project_name="Demo")
    workspace.scenario("wide")

    with pytest.raises(ValueError, match="already exists"):
        workspace.scenario("wide")
    with pytest.raises(Exception):
        workspace.scenario("broken", options={"NOT_AN_OPTION": 1})

    assert "broken" not in workspace.list_cases()
    workspace.scenario("wide", overwrite=True)


# 4.5 and 4.7 Exports


def test_export_excel_to_an_xlsx_name_writes_that_file(tmp_path: Path):
    problem = PinchProblem(_basic_payload(), project_name="Demo")
    problem.target.direct_heat_integration()

    path = problem.export_excel(tmp_path / "results.xlsx")

    assert Path(path) == tmp_path / "results.xlsx"
    assert Path(path).is_file()


def test_returned_results_do_not_alias_the_cache():
    problem = PinchProblem(_basic_payload(), project_name="Demo")
    problem.target.direct_heat_integration()

    first = problem.results
    first.targets[0].scope = "edited"

    assert problem.results.targets[0].scope != "edited"


def test_sheet_names_are_unique_ignoring_case():
    from OpenPinch.presentation.reporting.workbook import _unique_sheet_name

    used: set[str] = set()
    first = _unique_sheet_name("Plant", used)
    second = _unique_sheet_name("PLANT", used)

    assert first.casefold() != second.casefold()


def test_cli_notebook_refuses_to_overwrite_without_force(tmp_path: Path):
    from OpenPinch.__main__ import main
    from OpenPinch.resources import list_notebooks

    name = list_notebooks()[0]
    assert main(["notebook", "--name", name, "-o", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        main(["notebook", "--name", name, "-o", str(tmp_path)])
    assert main(["notebook", "--name", name, "-o", str(tmp_path), "--force"]) == 0

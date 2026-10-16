"""Area, cost and exergy columns in the summary once those targets have run."""

import pytest

from OpenPinch import PinchProblem


def test_summary_frame_shows_area_and_cost_after_the_area_target():
    problem = PinchProblem("basic_pinch.json")
    problem.target.heat_exchanger_area_and_cost()
    target = problem.results.targets[0]

    row = problem.summary_frame().iloc[0]

    assert row["Area"] == f"{target.area.value:.2f} m^2"
    assert row["Units"] == f"{target.num_units:.2f}"
    assert row["Capital Cost"] == f"{target.capital_cost.value:.2f} $"
    assert row["Total Cost"] == f"{target.total_cost.value:.2f} $/y"


def test_summary_frame_shows_exergy_after_the_exergy_target():
    problem = PinchProblem("basic_pinch.json")
    problem.target.direct_heat_integration()
    problem.target.exergy()
    target = problem.results.targets[0]

    row = problem.summary_frame().iloc[0]

    assert row["ETE"] == f"{target.ETE.value:.2f} {target.ETE.unit}"
    assert row["Exergy Sources"] == f"{target.exergy_sources.value:.2f} kW"
    assert row["Exergy Sinks"] == f"{target.exergy_sinks.value:.2f} kW"
    assert row["Min Exergy Requirement"] == f"{target.exergy_req_min.value:.2f} kW"
    assert row["Min Exergy Destruction"] == f"{target.exergy_des_min.value:.2f} kW"
    assert "Area" not in row


def test_summary_frame_columns_are_unchanged_without_those_targets():
    problem = PinchProblem("basic_pinch.json")
    problem.target.direct_heat_integration()

    assert list(problem.summary_frame().columns) == [
        "Scope",
        "Zone Type",
        "Integration Type",
        "Target Method",
        "Period ID",
        "Hot Utility Target",
        "Cold Utility Target",
        "Heat Recovery",
        "Hot Pinch",
        "Cold Pinch",
        "Hot Utilities",
        "Cold Utilities",
    ]


def test_report_metrics_include_area_and_cost_after_the_area_target():
    problem = PinchProblem("basic_pinch.json")
    problem.target.direct_heat_integration()
    before = {metric.label for metric in problem.report().metrics}
    problem.target.heat_exchanger_area_and_cost()
    target = problem.results.targets[0]

    metrics = {metric.label: metric for metric in problem.report().metrics}

    assert "Area" not in before
    assert metrics["Area"].value == pytest.approx(target.area.value, rel=1e-12)
    assert metrics["Area"].unit == "m^2"
    assert metrics["Capital Cost"].unit == "$"
    assert metrics["Total Cost"].unit == "$/y"
    assert metrics["Units"].unit is None


def test_weighted_summary_row_takes_the_peak_period_area():
    problem = PinchProblem("crude_preheat_train_multiperiod.json")
    problem.target.all_periods.heat_exchanger_area_and_cost()

    frame = problem.summary_frame(include_periods=True, include_weighted_average=True)
    areas = {
        row["Period ID"]: float(row["Area"].split()[0]) for _, row in frame.iterrows()
    }

    assert set(areas) == {"turndown", "base", "peak", "weighted_average"}
    assert areas["weighted_average"] == pytest.approx(
        max(areas[p] for p in ("turndown", "base", "peak")), abs=0.01
    )

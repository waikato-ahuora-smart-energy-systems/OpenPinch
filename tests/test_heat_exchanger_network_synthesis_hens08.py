"""HENS-08 stage-reduction and topology-evolution regression coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from OpenPinch import PinchProblem
from OpenPinch.classes.heat_exchanger import HeatExchangerKind
from OpenPinch.services.heat_exchanger_network_synthesis.array_adapter import (
    PreparedSolverArrays,
    problem_to_solver_arrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models.extraction import (
    extract_heat_exchanger_network,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "Four-stream-Yee-and-Grossmann-1990-1"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "openhens" / f"{CASE_ID}.json"
SNAPSHOT_PATH = (
    REPO_ROOT
    / "openhens_baseline_results"
    / "network_snapshots"
    / CASE_ID
    / "best-esm.json"
)


def test_four_stream_best_esm_snapshot_matches_identity_labelled_network() -> None:
    snapshot = _load_json(SNAPSHOT_PATH)
    source_artifact = _load_json(REPO_ROOT / snapshot["source_artifact"])
    source_summary = _load_json(REPO_ROOT / snapshot["source_summary_artifact"])
    solution = source_artifact["solution"]
    expected = snapshot["expected"]
    tolerances = snapshot["tolerances"]
    arrays = problem_to_solver_arrays(
        PinchProblem(source=FIXTURE_PATH),
        expected["best_dTmin"],
    )

    network = extract_heat_exchanger_network(
        _artifact_solved_model(solution, arrays),
        arrays,
        run_id=CASE_ID,
        task_id=snapshot["task_id"],
        method=snapshot["method"],
        stage_count=expected["stage_count"],
        tolerance=tolerances["duty_abs"],
    )

    assert snapshot["task_id"] == solution["task_id"]
    assert expected["total_annual_cost"] == source_summary["best_solution"]
    assert expected["best_dTmin"] == source_summary["best_dTmin"]
    assert expected["best_derivative_threshold"] == source_summary["best_min_dQ"]
    assert expected["stage_count"] == source_summary["best_stages"]
    assert expected["recovery_unit_count"] == source_summary["best_recovery_units"]
    assert expected["hot_utility_unit_count"] == source_summary["best_hu_units"]
    assert expected["cold_utility_unit_count"] == source_summary["best_cu_units"]

    assert network.stage_count == expected["stage_count"]
    assert network.summary_metrics["recovery_units"] == expected["recovery_unit_count"]
    assert (
        network.summary_metrics["hot_utility_units"]
        == expected["hot_utility_unit_count"]
    )
    assert (
        network.summary_metrics["cold_utility_units"]
        == expected["cold_utility_unit_count"]
    )
    _assert_close(
        network.total_annual_cost,
        expected["total_annual_cost"],
        abs_tol=tolerances["total_annual_cost_abs"],
        rel_tol=tolerances["total_annual_cost_rel"],
    )

    assert len(network.exchangers) == len(snapshot["exchangers"])
    for expected_exchanger in snapshot["exchangers"]:
        exchanger = network.exchanger_between(
            source_stream=expected_exchanger["source_stream"],
            sink_stream=expected_exchanger["sink_stream"],
            stage=expected_exchanger["stage"],
            kind=HeatExchangerKind(expected_exchanger["kind"]),
        )
        assert exchanger is not None, expected_exchanger
        assert exchanger.kind.value == expected_exchanger["kind"]
        assert exchanger.source_stream == expected_exchanger["source_stream"]
        assert exchanger.sink_stream == expected_exchanger["sink_stream"]
        assert exchanger.stage == expected_exchanger["stage"]
        _assert_close(
            exchanger.duty,
            expected_exchanger["duty"],
            abs_tol=tolerances["duty_abs"],
            rel_tol=tolerances["duty_rel"],
        )
        _assert_close(
            exchanger.area,
            expected_exchanger["area"],
            abs_tol=tolerances["area_abs"],
            rel_tol=tolerances["area_rel"],
        )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_solved_model(
    solution: dict,
    arrays: PreparedSolverArrays,
) -> SimpleNamespace:
    unit_counts = solution["unit_counts"]
    utility_loads = solution["utility_loads"]
    return SimpleNamespace(
        S=solution["stages"],
        stages=solution["stages"],
        Q_r=solution["recovery_heat_duties"],
        Q_h=solution["hot_utility_duties"],
        Q_c=solution["cold_utility_duties"],
        T_h=solution["hot_stream_temperatures"],
        T_c=solution["cold_stream_temperatures"],
        T_h_out_x=solution["hot_recovery_outlet_temperatures"],
        T_c_out_y=solution["cold_recovery_outlet_temperatures"],
        area_r=solution["recovery_areas"],
        area_hu=solution["hot_utility_areas"],
        area_cu=solution["cold_utility_areas"],
        TAC=solution["total_annual_cost"],
        TAC_model=solution["model_objective_value"],
        n_units=unit_counts["total"],
        n_recovery_units=unit_counts["recovery"],
        n_hu_units=unit_counts["hot_utility"],
        n_cu_units=unit_counts["cold_utility"],
        Q_hu_total=utility_loads["hot"],
        Q_cu_total=utility_loads["cold"],
        Q_r_total=_recovery_total(solution["recovery_heat_duties"]),
        T_h_out=arrays.arrays["T_h_out"].tolist(),
        T_c_out=arrays.arrays["T_c_out"].tolist(),
        T_hu_in=arrays.arrays["T_hu_in"].tolist(),
        T_hu_out=arrays.arrays["T_hu_out"].tolist(),
        T_cu_in=arrays.arrays["T_cu_in"].tolist(),
        T_cu_out=arrays.arrays["T_cu_out"].tolist(),
    )


def _recovery_total(values: list) -> float:
    return sum(
        duty
        for hot_stream in values
        for cold_stream in hot_stream
        for duty in cold_stream
    )


def _assert_close(
    actual: float | None,
    expected: float,
    *,
    abs_tol: float,
    rel_tol: float,
) -> None:
    assert actual is not None
    assert actual == pytest.approx(expected, abs=abs_tol, rel=rel_tol)

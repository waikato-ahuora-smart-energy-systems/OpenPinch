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
    expected = snapshot["expected"]
    tolerances = snapshot["tolerances"]
    arrays = problem_to_solver_arrays(
        PinchProblem(source=FIXTURE_PATH),
        expected["best_dTmin"],
    )

    network = extract_heat_exchanger_network(
        _snapshot_solved_model(snapshot, arrays),
        arrays,
        run_id=CASE_ID,
        task_id=snapshot["task_id"],
        method=snapshot["method"],
        stage_count=expected["stage_count"],
        tolerance=tolerances["duty_abs"],
    )

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


def _snapshot_solved_model(
    snapshot: dict,
    arrays: PreparedSolverArrays,
) -> SimpleNamespace:
    expected = snapshot["expected"]
    solution_arrays = _snapshot_solution_arrays(snapshot, arrays)
    return SimpleNamespace(
        S=expected["stage_count"],
        stages=expected["stage_count"],
        Q_r=solution_arrays["Q_r"],
        Q_h=solution_arrays["Q_h"],
        Q_c=solution_arrays["Q_c"],
        T_h=solution_arrays["T_h"],
        T_c=solution_arrays["T_c"],
        theta_1=solution_arrays["theta"],
        theta_2=solution_arrays["theta"],
        area_r=solution_arrays["area_r"],
        area_hu=solution_arrays["area_hu"],
        area_cu=solution_arrays["area_cu"],
        TAC=expected["total_annual_cost"],
        TAC_model=expected["total_annual_cost"],
        n_units=expected["total_unit_count"],
        n_recovery_units=expected["recovery_unit_count"],
        n_hu_units=expected["hot_utility_unit_count"],
        n_cu_units=expected["cold_utility_unit_count"],
        Q_hu_total=expected["hot_utility_load"],
        Q_cu_total=expected["cold_utility_load"],
        Q_r_total=_recovery_total(solution_arrays["Q_r"]),
        T_h_out=arrays.arrays["T_h_out"].tolist(),
        T_c_out=arrays.arrays["T_c_out"].tolist(),
        T_hu_in=arrays.arrays["T_hu_in"].tolist(),
        T_hu_out=arrays.arrays["T_hu_out"].tolist(),
        T_cu_in=arrays.arrays["T_cu_in"].tolist(),
        T_cu_out=arrays.arrays["T_cu_out"].tolist(),
    )


def _snapshot_solution_arrays(snapshot: dict, arrays: PreparedSolverArrays) -> dict:
    expected = snapshot["expected"]
    stages = expected["stage_count"]
    hot_count = len(arrays.axis_maps["hot_process_streams"])
    cold_count = len(arrays.axis_maps["cold_process_streams"])
    q_r = _zeros(hot_count, cold_count, stages)
    area_r = _zeros(hot_count, cold_count, stages)
    theta = _zeros(hot_count, cold_count, stages)
    q_h = [0.0 for _ in range(cold_count)]
    q_c = [0.0 for _ in range(hot_count)]
    area_hu = [0.0 for _ in range(cold_count)]
    area_cu = [0.0 for _ in range(hot_count)]

    hot_axis = arrays.axis_maps["hot_process_streams"]
    cold_axis = arrays.axis_maps["cold_process_streams"]
    for exchanger in snapshot["exchangers"]:
        kind = exchanger["kind"]
        if kind == "recovery":
            hot_index = hot_axis[exchanger["source_stream"]]
            cold_index = cold_axis[exchanger["sink_stream"]]
            stage_index = exchanger["stage"] - 1
            q_r[hot_index][cold_index][stage_index] = exchanger["duty"]
            area_r[hot_index][cold_index][stage_index] = exchanger["area"]
            theta[hot_index][cold_index][stage_index] = expected["best_dTmin"]
        elif kind == "hot_utility":
            cold_index = cold_axis[exchanger["sink_stream"]]
            q_h[cold_index] = exchanger["duty"]
            area_hu[cold_index] = exchanger["area"]
        elif kind == "cold_utility":
            hot_index = hot_axis[exchanger["source_stream"]]
            q_c[hot_index] = exchanger["duty"]
            area_cu[hot_index] = exchanger["area"]

    return {
        "Q_r": q_r,
        "Q_h": q_h,
        "Q_c": q_c,
        "T_h": _stage_temperatures(
            arrays.arrays["T_h_in"].tolist(),
            arrays.arrays["T_h_out"].tolist(),
            stages,
        ),
        "T_c": _stage_temperatures(
            arrays.arrays["T_c_out"].tolist(),
            arrays.arrays["T_c_in"].tolist(),
            stages,
        ),
        "theta": theta,
        "area_r": area_r,
        "area_hu": area_hu,
        "area_cu": area_cu,
    }


def _zeros(*dimensions: int) -> list:
    if len(dimensions) == 1:
        return [0.0 for _ in range(dimensions[0])]
    return [_zeros(*dimensions[1:]) for _ in range(dimensions[0])]


def _stage_temperatures(inlets: list[float], outlets: list[float], stages: int) -> list:
    return [
        [
            inlet + (outlet - inlet) * stage / stages
            for stage in range(stages + 1)
        ]
        for inlet, outlet in zip(inlets, outlets, strict=True)
    ]


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

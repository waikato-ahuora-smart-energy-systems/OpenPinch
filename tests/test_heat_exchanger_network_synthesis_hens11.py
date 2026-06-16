"""HENS-11 regression expansion and retirement-gate coverage."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from shutil import which
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.classes.heat_exchanger import HeatExchangerKind
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.services.heat_exchanger_network_synthesis.array_adapter import (
    PreparedSolverArrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models.extraction import (
    extract_heat_exchanger_network,
)
from OpenPinch.services.heat_exchanger_network_synthesis.service import (
    heat_exchanger_network_synthesis_service,
)
from OpenPinch.services.heat_exchanger_network_synthesis.workflow import (
    LocalSynthesisExecutor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "openhens_baseline_results"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
NETWORK_SNAPSHOT_ROOT = BASELINE_ROOT / "network_snapshots"

TAC_ABS_TOL = 1.0
TAC_REL_TOL = 1e-4
MAX_REGRESSION_REL_TOL = 1e-2
TEMPERATURE_ABS_TOL = 1.0

FOUR_STREAM = "Four-stream-Yee-and-Grossmann-1990-1"
NINE_STREAM = "Nine-stream-Linnhoff-and-Ahmad-1999-1"

BASELINE_EXPECTATIONS = {
    FOUR_STREAM: {
        "role": "routine",
        "best_solution": 154853.8518602861,
        "quartile_1": 154853.8601589499,
        "quartile_2": 159038.71986626228,
        "quartile_3": 163556.12518498677,
        "solved_esm_count": 100,
        "total_cases_attempted": 1210,
        "total_cases_solved": 1210,
        "within_2_percent": 37,
        "within_5_percent": 64,
        "within_10_percent": 100,
        "best_dTmin": 14,
        "best_min_dQ": 0.5,
        "best_stages": 3,
        "best_recovery_units": 3,
        "best_cu_units": 2,
        "best_hu_units": 1,
    },
    NINE_STREAM: {
        "role": "final-verification",
        "best_solution": 2905807.275299348,
        "quartile_1": 2947087.026205118,
        "quartile_2": 2954844.749379795,
        "quartile_3": 2969362.7969760075,
        "solved_esm_count": 71,
        "total_cases_attempted": 1155,
        "total_cases_solved": 886,
        "within_2_percent": 46,
        "within_5_percent": 68,
        "within_10_percent": 70,
        "best_dTmin": 18,
        "best_min_dQ": 1.7,
        "best_stages": 4,
        "best_recovery_units": 11,
        "best_cu_units": 3,
        "best_hu_units": 3,
    },
}

INTEGER_SUMMARY_FIELDS = (
    "solved_esm_count",
    "total_cases_attempted",
    "total_cases_solved",
    "within_2_percent",
    "within_5_percent",
    "within_10_percent",
    "best_stages",
    "best_recovery_units",
    "best_cu_units",
    "best_hu_units",
)

SUMMARY_QUARTILE_FIELDS = ("quartile_1", "quartile_2", "quartile_3")


@pytest.mark.parametrize("case_id", [FOUR_STREAM, NINE_STREAM])
def test_openhens_baseline_summary_preserves_solver_metric_contract(
    case_id: str,
) -> None:
    expected = BASELINE_EXPECTATIONS[case_id]
    summary = _summary(case_id)
    snapshot = _network_snapshot(case_id)
    best_row = _best_solution_metrics_row(case_id, snapshot["task_id"])
    artifact = _artifact(snapshot["source_artifact"])
    solution = artifact["solution"]

    _assert_close(
        summary["best_solution"],
        expected["best_solution"],
        abs_tol=TAC_ABS_TOL,
        rel_tol=TAC_REL_TOL,
    )
    for field in SUMMARY_QUARTILE_FIELDS:
        _assert_close(
            summary[field],
            expected[field],
            abs_tol=0.0,
            rel_tol=MAX_REGRESSION_REL_TOL,
        )
    for field in INTEGER_SUMMARY_FIELDS:
        assert summary[field] == expected[field]
    assert summary["best_dTmin"] == expected["best_dTmin"]
    assert summary["best_min_dQ"] == expected["best_min_dQ"]

    assert best_row["Method"] == "ESM"
    assert best_row["Task ID"] == snapshot["task_id"]
    assert float(best_row["ESM TAC"]) == pytest.approx(expected["best_solution"])
    assert float(best_row["dTmin"]) == expected["best_dTmin"]
    assert float(best_row["min_dQ"]) == expected["best_min_dQ"]
    assert int(best_row["Stages"]) == expected["best_stages"]
    assert int(best_row["N Recovery Units"]) == expected["best_recovery_units"]
    assert int(best_row["N CU Units"]) == expected["best_cu_units"]
    assert int(best_row["N HU Units"]) == expected["best_hu_units"]
    assert best_row["Solver Status"] == "1"
    assert best_row["Verification Failures"] == ""

    assert artifact["success"] is True
    assert artifact["verification_failures"] == []
    assert solution["verification_failures"] == []
    assert solution["utility_loads"]["hot"] == pytest.approx(
        snapshot["expected"]["hot_utility_load"],
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )
    assert solution["utility_loads"]["cold"] == pytest.approx(
        snapshot["expected"]["cold_utility_load"],
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )


@pytest.mark.parametrize("case_id", [FOUR_STREAM, NINE_STREAM])
def test_best_esm_network_snapshot_matches_identity_labelled_network(
    case_id: str,
) -> None:
    snapshot, source_summary, source_artifact, _arrays, network = _network_context(
        case_id
    )
    solution = source_artifact["solution"]
    expected = snapshot["expected"]
    tolerances = snapshot["tolerances"]

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
    assert network.summary_metrics["hot_utility_load"] == pytest.approx(
        expected["hot_utility_load"],
        abs=tolerances["duty_abs"],
        rel=tolerances["duty_rel"],
    )
    assert network.summary_metrics["cold_utility_load"] == pytest.approx(
        expected["cold_utility_load"],
        abs=tolerances["duty_abs"],
        rel=tolerances["duty_rel"],
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


@pytest.mark.parametrize("case_id", [FOUR_STREAM, NINE_STREAM])
def test_best_esm_network_satisfies_numerical_invariants(case_id: str) -> None:
    snapshot, _source_summary, source_artifact, arrays, network = _network_context(
        case_id
    )
    fixture = _fixture(case_id)
    solution = source_artifact["solution"]

    assert network.total_duty(kind=HeatExchangerKind.HOT_UTILITY) == pytest.approx(
        solution["utility_loads"]["hot"],
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )
    assert network.total_duty(kind=HeatExchangerKind.COLD_UTILITY) == pytest.approx(
        solution["utility_loads"]["cold"],
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )
    assert network.total_duty(kind=HeatExchangerKind.RECOVERY) == pytest.approx(
        _nested_sum(solution["recovery_heat_duties"]),
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )

    _assert_process_stream_heat_balances(network, arrays)
    _assert_stage_heat_balances(network, solution)
    _assert_temperature_feasibility(network, solution["dTmin"])
    _assert_cost_recomputation(
        network,
        fixture,
        solution,
        expected_total=snapshot["expected"]["total_annual_cost"],
    )


@pytest.mark.solver
@pytest.mark.parametrize("case_id", [FOUR_STREAM, NINE_STREAM])
def test_marked_solver_baseline_matches_checked_in_summary(case_id: str) -> None:
    _require_live_solver_environment()
    expected = BASELINE_EXPECTATIONS[case_id]
    problem = PinchProblem(source=FIXTURE_ROOT / f"{case_id}.json")

    design = heat_exchanger_network_synthesis_service(
        problem,
        executor=LocalSynthesisExecutor(print_output=False),
    )
    network = design.network

    _assert_close(
        design.objective_values["total_annual_cost"],
        expected["best_solution"],
        abs_tol=TAC_ABS_TOL,
        rel_tol=TAC_REL_TOL,
    )
    assert len(design.task_outcomes) == expected["total_cases_attempted"]
    assert (
        sum(outcome.status == "success" for outcome in design.task_outcomes)
        == expected["total_cases_solved"]
    )
    assert (
        sum(
            outcome.status == "success"
            and outcome.task.method == "energy_stage_refinement"
            for outcome in design.task_outcomes
        )
        == expected["solved_esm_count"]
    )
    assert network.stage_count == expected["best_stages"]
    assert network.summary_metrics["recovery_units"] == expected["best_recovery_units"]
    assert network.summary_metrics["hot_utility_units"] == expected["best_hu_units"]
    assert network.summary_metrics["cold_utility_units"] == expected["best_cu_units"]


def _network_context(
    case_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    PreparedSolverArrays,
    HeatExchangerNetwork,
]:
    snapshot = _network_snapshot(case_id)
    source_artifact = _artifact(snapshot["source_artifact"])
    source_summary = _load_json(REPO_ROOT / snapshot["source_summary_artifact"])
    fixture = _fixture(case_id)
    solution = source_artifact["solution"]
    arrays = _source_artifact_solver_arrays(fixture, snapshot["expected"]["best_dTmin"])
    network = extract_heat_exchanger_network(
        _artifact_solved_model(solution, arrays),
        arrays,
        run_id=case_id,
        task_id=snapshot["task_id"],
        method=snapshot["method"],
        stage_count=snapshot["expected"]["stage_count"],
        tolerance=snapshot["tolerances"]["duty_abs"],
    )
    return snapshot, source_summary, source_artifact, arrays, network


def _source_artifact_solver_arrays(
    fixture: dict[str, Any],
    d_tmin: float,
) -> PreparedSolverArrays:
    hot_streams = [
        stream
        for stream in fixture["streams"]
        if stream["t_supply"]["value"] > stream["t_target"]["value"]
    ]
    cold_streams = [
        stream
        for stream in fixture["streams"]
        if stream["t_supply"]["value"] < stream["t_target"]["value"]
    ]
    hot_utilities = [
        utility for utility in fixture["utilities"] if utility["type"] == "Hot"
    ]
    cold_utilities = [
        utility for utility in fixture["utilities"] if utility["type"] == "Cold"
    ]

    return PreparedSolverArrays(
        arrays={
            "T_c_in": _float_array(
                stream["t_supply"]["value"] for stream in cold_streams
            ),
            "T_c_out": _float_array(
                stream["t_target"]["value"] for stream in cold_streams
            ),
            "T_cu_in": _float_array(
                utility["t_supply"]["value"] for utility in cold_utilities
            ),
            "T_cu_out": _float_array(
                utility["t_target"]["value"] for utility in cold_utilities
            ),
            "T_h_in": _float_array(
                stream["t_supply"]["value"] for stream in hot_streams
            ),
            "T_h_out": _float_array(
                stream["t_target"]["value"] for stream in hot_streams
            ),
            "T_hu_in": _float_array(
                utility["t_supply"]["value"] for utility in hot_utilities
            ),
            "T_hu_out": _float_array(
                utility["t_target"]["value"] for utility in hot_utilities
            ),
            "f_c": _float_array(_heat_capacity_flow(stream) for stream in cold_streams),
            "f_h": _float_array(_heat_capacity_flow(stream) for stream in hot_streams),
        },
        axis_maps={
            "cold_process_streams": {
                _stream_identity(stream): index
                for index, stream in enumerate(cold_streams)
            },
            "cold_utilities": {
                _utility_identity(utility): index
                for index, utility in enumerate(cold_utilities)
            },
            "hot_process_streams": {
                _stream_identity(stream): index
                for index, stream in enumerate(hot_streams)
            },
            "hot_utilities": {
                _utility_identity(utility): index
                for index, utility in enumerate(hot_utilities)
            },
        },
        configuration={"active_dTmin": float(d_tmin)},
        preparation={
            "artifact_decoder": "OpenHENS source artifact order",
            "pinch_problem_class": "PinchProblem",
        },
        stream_identities={
            "cold_process_streams": [
                _stream_identity(stream) for stream in cold_streams
            ],
            "hot_process_streams": [_stream_identity(stream) for stream in hot_streams],
        },
        unit_conventions={
            "heat_capacity_flowrate": "kW/K",
            "temperature": "K",
        },
        utility_identities={
            "cold_utilities": [
                _utility_identity(utility) for utility in cold_utilities
            ],
            "hot_utilities": [_utility_identity(utility) for utility in hot_utilities],
        },
    )


def _artifact_solved_model(
    solution: dict[str, Any],
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
        Q_r_total=_nested_sum(solution["recovery_heat_duties"]),
        T_h_out=arrays.arrays["T_h_out"].tolist(),
        T_c_out=arrays.arrays["T_c_out"].tolist(),
        T_hu_in=arrays.arrays["T_hu_in"].tolist(),
        T_hu_out=arrays.arrays["T_hu_out"].tolist(),
        T_cu_in=arrays.arrays["T_cu_in"].tolist(),
        T_cu_out=arrays.arrays["T_cu_out"].tolist(),
    )


def _assert_process_stream_heat_balances(
    network: HeatExchangerNetwork,
    arrays: PreparedSolverArrays,
) -> None:
    for stream_id, index in arrays.axis_maps["hot_process_streams"].items():
        expected_heat_flow = (
            arrays.arrays["T_h_in"][index] - arrays.arrays["T_h_out"][index]
        ) * arrays.arrays["f_h"][index]
        actual_heat_flow = network.total_duty(stream=stream_id)
        assert actual_heat_flow == pytest.approx(
            expected_heat_flow,
            abs=TAC_ABS_TOL,
            rel=TAC_REL_TOL,
        )

    for stream_id, index in arrays.axis_maps["cold_process_streams"].items():
        expected_heat_flow = (
            arrays.arrays["T_c_out"][index] - arrays.arrays["T_c_in"][index]
        ) * arrays.arrays["f_c"][index]
        actual_heat_flow = sum(
            exchanger.duty
            for exchanger in network.exchangers
            if exchanger.sink_stream == stream_id
        )
        assert actual_heat_flow == pytest.approx(
            expected_heat_flow,
            abs=TAC_ABS_TOL,
            rel=TAC_REL_TOL,
        )


def _assert_stage_heat_balances(
    network: HeatExchangerNetwork,
    solution: dict[str, Any],
) -> None:
    for stage in range(1, solution["stages"] + 1):
        expected_stage_duty = sum(
            cold_stream[stage - 1]
            for hot_stream in solution["recovery_heat_duties"]
            for cold_stream in hot_stream
        )
        assert network.total_duty(
            kind=HeatExchangerKind.RECOVERY,
            stage=stage,
        ) == pytest.approx(expected_stage_duty, abs=TAC_ABS_TOL, rel=TAC_REL_TOL)


def _assert_temperature_feasibility(
    network: HeatExchangerNetwork,
    d_tmin: float,
) -> None:
    for exchanger in network.exchangers:
        assert exchanger.area is not None
        assert exchanger.area >= 0.0
        if exchanger.source_inlet_temperature is not None:
            assert exchanger.source_outlet_temperature is not None
            assert (
                exchanger.source_inlet_temperature + TEMPERATURE_ABS_TOL
                >= exchanger.source_outlet_temperature
            )
        if exchanger.sink_inlet_temperature is not None:
            assert exchanger.sink_outlet_temperature is not None
            assert (
                exchanger.sink_outlet_temperature + TEMPERATURE_ABS_TOL
                >= exchanger.sink_inlet_temperature
            )
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            approach_temperatures = exchanger.approach_temperatures or (
                exchanger.source_inlet_temperature - exchanger.sink_outlet_temperature,
                exchanger.source_outlet_temperature - exchanger.sink_inlet_temperature,
            )
            assert min(approach_temperatures) + TEMPERATURE_ABS_TOL >= d_tmin


def _assert_cost_recomputation(
    network: HeatExchangerNetwork,
    fixture: dict[str, Any],
    solution: dict[str, Any],
    *,
    expected_total: float,
) -> None:
    options = fixture["options"]
    fixed_cost = float(options["FIXED_COST"])
    variable_cost = float(options["VARIABLE_COST"])
    cost_exp = float(options["COST_EXP"])
    hot_utility_price = _utility_price(fixture, "Hot")
    cold_utility_price = _utility_price(fixture, "Cold")

    utility_cost = (
        hot_utility_price * solution["utility_loads"]["hot"]
        + cold_utility_price * solution["utility_loads"]["cold"]
    )
    area_cost = variable_cost * sum(
        float(exchanger.area) ** cost_exp
        for exchanger in network.exchangers
        if exchanger.area is not None
    )
    area_cost += fixed_cost * solution["unit_counts"]["total"]

    assert solution["total_annual_cost"] == pytest.approx(
        expected_total,
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )
    assert solution["total_annual_cost"] - area_cost == pytest.approx(
        utility_cost,
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )
    assert solution["total_annual_cost"] - utility_cost == pytest.approx(
        area_cost,
        abs=TAC_ABS_TOL,
        rel=TAC_REL_TOL,
    )


def _require_live_solver_environment() -> None:
    missing = [binary for binary in ("couenne", "ipopt") if which(binary) is None]
    if missing:
        pytest.skip(
            "HENS-11 solver baseline requires external solver binaries on PATH; "
            f"missing {', '.join(missing)}. Rerun with: "
            "rtk uv run pytest -m solver"
        )
    pytest.importorskip(
        "gekko",
        reason='install "openpinch[synthesis]" before running solver tests',
    )
    pytest.importorskip(
        "pyomo.environ",
        reason='install "openpinch[synthesis]" before running solver tests',
    )


def _stream_identity(stream: dict[str, Any]) -> str:
    zone = str(stream["zone"]).rsplit("/", maxsplit=1)[-1]
    return f"{zone}.{stream['name'].strip()}"


def _utility_identity(utility: dict[str, Any]) -> str:
    role = "Hot Utility" if utility["type"] == "Hot" else "Cold Utility"
    return f"{role}.{utility['name'].strip()}"


def _heat_capacity_flow(stream: dict[str, Any]) -> float:
    heat_flow = float(stream["heat_flow"]["value"])
    temperature_delta = abs(
        float(stream["t_supply"]["value"]) - float(stream["t_target"]["value"])
    )
    return heat_flow / temperature_delta


def _float_array(values) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _utility_price(fixture: dict[str, Any], utility_type: str) -> float:
    return next(
        float(utility["price"]["value"])
        for utility in fixture["utilities"]
        if utility["type"] == utility_type
    )


def _best_solution_metrics_row(case_id: str, task_id: str) -> dict[str, str]:
    with (BASELINE_ROOT / "refactor" / case_id / "solution_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["Task ID"] == task_id)


def _artifact(source_artifact: str) -> dict[str, Any]:
    return _load_json(REPO_ROOT / source_artifact)


def _fixture(case_id: str) -> dict[str, Any]:
    return _load_json(FIXTURE_ROOT / f"{case_id}.json")


def _summary(case_id: str) -> dict[str, Any]:
    return _load_json(BASELINE_ROOT / "refactor" / case_id / "summary.json")


def _network_snapshot(case_id: str) -> dict[str, Any]:
    return _load_json(NETWORK_SNAPSHOT_ROOT / case_id / "best-esm.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_sum(values: list[Any]) -> float:
    total = 0.0
    stack = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(value)
        else:
            total += float(value)
    return total


def _assert_close(
    actual: float | None,
    expected: float,
    *,
    abs_tol: float,
    rel_tol: float,
) -> None:
    assert actual is not None
    assert actual == pytest.approx(expected, abs=abs_tol, rel=rel_tol)

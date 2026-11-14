"""Manual live-solver benchmark for the nine-stream OpenHENS case.

This is run by hand and has no CI use. It solves the Linnhoff and Ahmad (1999)
nine-stream case with the full OpenHENS workflow and eight workers, which can
take hours (it did not finish in 120 minutes in one attempt), and needs the
synthesis extra plus the Couenne and Ipopt binaries. It checks the result
against the values the former pytest regression asserted, and exits with
status 1 listing each mismatch.

    uv run python scripts/benchmark_nine_stream.py

The heat-balance, temperature and cost checks reuse the helpers in
``tests/analysis/heat_exchanger_networks/test_solver_regressions.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

CASE_ID = "Nine-stream-Linnhoff-and-Ahmad-1999-1"
MAX_PARALLEL = 8
TAC_ABS_TOL = 1.0
TAC_REL_TOL = 1e-4
ESM_ATTEMPT_WEIGHT = 11

EXPECTED = {
    "best_solution": 2905508.6335796523,
    "best_dt_cont_multiplier": 1.8,
    "total_cases_attempted": 1133,
    "total_cases_solved_min": 840,
    "solved_esm_count_min": 67,
    "best_stages": 4,
    "best_recovery_units": 11,
    "best_cu_units": 3,
    "best_hu_units": 2,
}


def _weighted_job_count(outcomes: Iterable[Any]) -> int:
    return sum(
        ESM_ATTEMPT_WEIGHT if outcome.task.method == "network_evolution_method" else 1
        for outcome in outcomes
    )


def compare_to_expected(
    *,
    total_annual_cost: float,
    stage_count: int,
    summary_metrics: dict[str, Any],
    outcomes: Iterable[Any],
) -> list[str]:
    """Return one message per value that differs from ``EXPECTED``."""
    outcomes = list(outcomes)
    successes = [outcome for outcome in outcomes if outcome.status == "success"]
    failures = []

    tolerance = max(TAC_ABS_TOL, TAC_REL_TOL * abs(EXPECTED["best_solution"]))
    if abs(total_annual_cost - EXPECTED["best_solution"]) > tolerance:
        failures.append(
            f"total_annual_cost {total_annual_cost} != {EXPECTED['best_solution']}"
        )
    attempted = _weighted_job_count(outcomes)
    if attempted != EXPECTED["total_cases_attempted"]:
        failures.append(
            f"total_cases_attempted {attempted} != {EXPECTED['total_cases_attempted']}"
        )
    solved = _weighted_job_count(successes)
    if solved < EXPECTED["total_cases_solved_min"]:
        failures.append(
            f"total_cases_solved {solved} < {EXPECTED['total_cases_solved_min']}"
        )
    solved_esm = sum(
        outcome.task.method == "network_evolution_method" for outcome in successes
    )
    if solved_esm < EXPECTED["solved_esm_count_min"]:
        failures.append(
            f"solved_esm_count {solved_esm} < {EXPECTED['solved_esm_count_min']}"
        )
    counts = {
        "stage_count": (stage_count, EXPECTED["best_stages"]),
        "recovery_units": (
            summary_metrics["recovery_units"],
            EXPECTED["best_recovery_units"],
        ),
        "hot_utility_units": (
            summary_metrics["hot_utility_units"],
            EXPECTED["best_hu_units"],
        ),
        "cold_utility_units": (
            summary_metrics["cold_utility_units"],
            EXPECTED["best_cu_units"],
        ),
    }
    for name, (actual, expected) in counts.items():
        if actual != expected:
            failures.append(f"{name} {actual} != {expected}")
    return failures


def main() -> int:
    sys.path.insert(0, str(_REPO_ROOT))
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method import (
        execute_open_hens_method,
    )
    from OpenPinch.application.problem import PinchProblem
    from tests.analysis.heat_exchanger_networks import test_solver_regressions as reg

    problem = PinchProblem(source=reg.FIXTURE_ROOT / f"{CASE_ID}.json")
    problem.target.all_heat_integration()
    settings = replace(
        workflow_settings_from_problem(problem), max_parallel=MAX_PARALLEL
    )
    workflow_result = execute_open_hens_method(
        problem, settings, executor=LocalSynthesisExecutor(print_output=False)
    )
    design = workflow_result.accepted_result
    network = design.network

    failures = compare_to_expected(
        total_annual_cost=design.objective_values["total_annual_cost"],
        stage_count=network.stage_count,
        summary_metrics=network.summary_metrics,
        outcomes=workflow_result.outcomes,
    )
    try:
        reg._assert_live_design_satisfies_heat_and_cost_contract(
            CASE_ID,
            network,
            expected_total=EXPECTED["best_solution"],
            d_tmin=reg._openhens_d_tmin(EXPECTED["best_dt_cont_multiplier"]),
        )
    except AssertionError as exc:
        failures.append(f"heat and cost contract: {exc}")

    for failure in failures:
        print(f"MISMATCH {failure}", file=sys.stderr)
    print("nine-stream benchmark " + ("FAILED" if failures else "matched"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

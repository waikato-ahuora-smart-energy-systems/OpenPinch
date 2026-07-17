"""Behavioural contract tests for reusable scalar optimisation."""

from __future__ import annotations

import ast
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, seed, settings

import OpenPinch.optimisation
import OpenPinch.optimisation.service as optimisation_service
from OpenPinch.optimisation.errors import (
    InvalidObjectiveValueError,
    InvalidOptimisationProblemError,
    NoOptimisationCandidatesError,
)
from OpenPinch.optimisation.models import (
    OptimisationMethod,
    OptimisationOptions,
    OptimisationProblem,
)
from OpenPinch.optimisation.service import run_multistart_minimisation
from tests.strategies.optimisation import finite_candidate_clouds


def _convex_objective(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float((x[0] - 0.4) ** 2 + (x[1] + 0.25) ** 2)


@pytest.mark.parametrize(
    ("method", "method_options"),
    [
        (OptimisationMethod.DUAL_ANNEALING, {}),
        (OptimisationMethod.CMA_ES, {"popsize": 6}),
        (
            OptimisationMethod.BAYESIAN,
            {"n_init": 4, "acq_candidates": 128},
        ),
        (
            OptimisationMethod.RBF,
            {"n_init": 4, "n_candidates": 128},
        ),
    ],
)
def test_each_backend_minimises_a_non_hpr_scalar_problem(
    method: OptimisationMethod,
    method_options: dict[str, object],
) -> None:
    problem = OptimisationProblem(
        objective=_convex_objective,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
    )

    result = run_multistart_minimisation(
        problem,
        method=method,
        options={
            "n_runs": 1,
            "maxiter": 6,
            "maxfun": 5_000,
            "max_minima": 2,
            "seed": 4,
            **method_options,
        },
    )

    assert result.method is method
    assert result.candidates
    assert result.best.objective < 1e-3
    assert result.best.point == pytest.approx((0.4, -0.25), abs=0.05)
    assert tuple(result.candidates) == tuple(sorted(result.candidates))


def test_method_run_is_reproducible_for_a_fixed_seed() -> None:
    problem = OptimisationProblem(
        objective=_convex_objective,
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
    )
    options = OptimisationOptions.from_mapping(
        {
            "n_runs": 1,
            "maxiter": 5,
            "maxfun": 1_000,
            "max_minima": 2,
            "seed": 20260715,
            "popsize": 6,
        }
    )

    first = run_multistart_minimisation(
        problem,
        method=OptimisationMethod.CMA_ES,
        options=options,
    )
    second = run_multistart_minimisation(
        problem,
        method=OptimisationMethod.CMA_ES,
        options=options,
    )

    assert first == second


def test_problem_and_results_are_immutable_values() -> None:
    problem = OptimisationProblem(
        objective=_convex_objective,
        bounds=((-1, 1), (-2, 2)),
        initial_points=((0, 0),),
    )

    assert problem.bounds == ((-1.0, 1.0), (-2.0, 2.0))
    assert problem.initial_points == ((0.0, 0.0),)
    with pytest.raises(FrozenInstanceError):
        problem.bounds = ()


@seed(20260715)
@given(candidate_cloud=finite_candidate_clouds())
@settings(max_examples=40)
def test_candidate_result_preserves_and_orders_finite_backend_values(
    candidate_cloud,
) -> None:
    points, objectives = candidate_cloud

    def backend(**_kwargs):
        return points.copy(), objectives.copy()

    problem = OptimisationProblem(
        objective=_convex_objective,
        bounds=((-100.0, 100.0), (-100.0, 100.0)),
    )

    with patch.object(optimisation_service, "_resolve_backend", return_value=backend):
        result = run_multistart_minimisation(problem, options={"n_runs": 1})

    assert len(result.candidates) == len(points)
    assert [candidate.objective for candidate in result.candidates] == sorted(
        objectives
    )
    assert all(len(candidate.point) == 2 for candidate in result.candidates)
    assert all(
        math.isfinite(value)
        for candidate in result.candidates
        for value in (candidate.objective, *candidate.point)
    )


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        (
            OptimisationProblem(objective=_convex_objective, bounds=()),
            "at least one",
        ),
        (
            OptimisationProblem(
                objective=_convex_objective,
                bounds=((1.0, -1.0),),
            ),
            "lower bound",
        ),
        (
            OptimisationProblem(
                objective=_convex_objective,
                bounds=((-1.0, 1.0),),
                initial_points=((2.0,),),
            ),
            "within",
        ),
    ],
)
def test_invalid_problem_data_is_rejected(
    problem: OptimisationProblem,
    message: str,
) -> None:
    with pytest.raises(InvalidOptimisationProblemError, match=message):
        run_multistart_minimisation(problem)


def test_unknown_method_and_backend_option_are_rejected() -> None:
    problem = OptimisationProblem(
        objective=_convex_objective,
        bounds=((-1.0, 1.0),),
    )
    with pytest.raises(
        InvalidOptimisationProblemError, match="Unsupported optimisation"
    ):
        run_multistart_minimisation(problem, method="unknown")
    with pytest.raises(InvalidOptimisationProblemError, match="unsupported_option"):
        run_multistart_minimisation(
            problem,
            options={"unsupported_option": True},
        )


def test_unexpected_objective_errors_are_not_converted_to_penalties() -> None:
    def broken_objective(_x):
        raise RuntimeError("programming defect")

    problem = OptimisationProblem(
        objective=broken_objective,
        bounds=((-1.0, 1.0),),
    )

    with pytest.raises(RuntimeError, match="programming defect"):
        run_multistart_minimisation(
            problem,
            method=OptimisationMethod.CMA_ES,
            options={"n_runs": 1, "maxiter": 1, "popsize": 4},
        )


@pytest.mark.parametrize(
    ("points", "objectives", "error"),
    [
        (np.empty((0, 1)), np.empty(0), NoOptimisationCandidatesError),
        (np.asarray([[0.0]]), np.asarray([np.nan]), InvalidObjectiveValueError),
        (np.asarray([[np.inf]]), np.asarray([0.0]), InvalidObjectiveValueError),
    ],
)
def test_invalid_backend_results_are_explicit(
    points,
    objectives,
    error,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        optimisation_service,
        "_resolve_backend",
        lambda _method: lambda **_kwargs: (points, objectives),
    )
    problem = OptimisationProblem(
        objective=_convex_objective,
        bounds=((-1.0, 1.0),),
    )

    with pytest.raises(error):
        run_multistart_minimisation(problem)


def test_optimisation_package_has_no_outward_openpinch_dependencies() -> None:
    package_dir = Path(OpenPinch.optimisation.__file__).parent
    forbidden_roots = {
        "adapters",
        "analysis",
        "application",
        "classes",
        "contracts",
        "domain",
        "lib",
        "presentation",
        "services",
        "utils",
    }

    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            if node.level < 2 or not node.module:
                continue
            assert node.module.split(".", 1)[0] not in forbidden_roots, path

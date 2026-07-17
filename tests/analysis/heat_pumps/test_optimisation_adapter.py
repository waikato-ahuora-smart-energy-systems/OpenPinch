"""Behavioural tests for the HPR-to-optimisation boundary."""

from __future__ import annotations

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter
from OpenPinch.contracts.hpr import HPRBackendResult, HPRThermoArtifacts
from OpenPinch.domain.enums import BB_Minimiser
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.optimisation.errors import NoOptimisationCandidatesError
from OpenPinch.optimisation.models import (
    OptimisationCandidate,
    OptimisationMethod,
    OptimisationResult,
)

from .helpers import _base_args


def _result(
    objective: float,
    *,
    success: bool = True,
) -> HPRBackendResult:
    return HPRBackendResult(
        obj=objective,
        utility_tot=objective,
        w_net=objective,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        success=success,
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
    )


def _quadratic_hpr_objective(point, _args, debug=False) -> HPRBackendResult:
    objective = float((point[0] - 0.25) ** 2)
    return _result(objective)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (None, ()),
        ([], ()),
        (1.5, ((1.5,),)),
        ([1.0, 2.0], ((1.0, 2.0),)),
        (np.array([[1.0], [2.0]]), ((1.0,), (2.0,))),
    ],
)
def test_initial_points_are_normalised_without_mutable_shape_state(values, expected):
    assert adapter.normalise_initial_points(values) == expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, OptimisationMethod.DUAL_ANNEALING),
        ("dual_annealing", OptimisationMethod.DUAL_ANNEALING),
        ("cmaes", OptimisationMethod.CMA_ES),
        ("bo", OptimisationMethod.BAYESIAN),
        ("rbf_surrogate", OptimisationMethod.RBF),
        (OptimisationMethod.RBF, OptimisationMethod.RBF),
        (BB_Minimiser.DA, OptimisationMethod.DUAL_ANNEALING),
        (BB_Minimiser.CMAES, OptimisationMethod.CMA_ES),
        (BB_Minimiser.BO, OptimisationMethod.BAYESIAN),
        (BB_Minimiser.RBF, OptimisationMethod.RBF),
    ],
)
def test_exact_configured_backend_identifiers_resolve_to_package_methods(
    configured, expected
):
    assert adapter._resolve_hpr_optimisation_method(configured) is expected


@pytest.mark.parametrize(
    "configured",
    ["rbf", "dual-annealing", "DUAL_ANNEALING", " dual_annealing", "bo "],
)
def test_noncanonical_backend_identifiers_are_rejected(configured):
    with pytest.raises(ValueError, match="Unsupported optimiser identifier"):
        adapter._resolve_hpr_optimisation_method(configured)


@pytest.mark.parametrize("configured", [1, 1.5, object()])
def test_non_string_backend_identifiers_are_rejected(configured):
    with pytest.raises(TypeError, match="must be a string or enum value"):
        adapter._resolve_hpr_optimisation_method(configured)


def test_legacy_hpr_method_normaliser_is_absent():
    assert not hasattr(adapter, "normalise_hpr_method")
    assert "normalise_hpr_method" not in adapter.__all__


def test_adapter_executes_reusable_optimiser_without_legacy_wrapper():
    candidates = adapter.run_hpr_candidate_search(
        objective=_quadratic_hpr_objective,
        initial_points=np.array([0.75]),
        bounds=[(0.0, 1.0)],
        args=_base_args(max_multi_start=1, bb_minimiser="dual_annealing"),
    )

    assert candidates
    assert candidates[0].objective < 1e-8
    assert candidates[0].point[0] == pytest.approx(0.25, abs=1e-4)


def test_explicit_failed_backend_result_becomes_finite_search_penalty():
    observed = {}

    def fake_run(problem, *, method, options):
        observed["objective"] = problem.objective(
            np.array([0.5]),
            *problem.args,
        )
        return OptimisationResult(
            method=method,
            candidates=(
                OptimisationCandidate(
                    objective=observed["objective"],
                    point=(0.5,),
                ),
            ),
        )

    candidates = adapter.run_hpr_candidate_search(
        objective=lambda _x, _args, debug=False: HPRBackendResult.failure(),
        initial_points=None,
        bounds=[(0.0, 1.0)],
        args=_base_args(),
        optimiser=fake_run,
    )

    assert observed["objective"] == pytest.approx(1e30)
    assert candidates[0].objective == pytest.approx(1e30)


def test_warm_start_is_ranked_with_backend_candidates():
    def optimiser(*_args, **_kwargs):
        return OptimisationResult(
            method=OptimisationMethod.RBF,
            candidates=(OptimisationCandidate(objective=0.8, point=(0.8,)),),
        )

    candidates = adapter.run_hpr_candidate_search(
        objective=lambda x, _args, debug=False: _result(float(x[0])),
        initial_points=np.array([0.2]),
        bounds=[(0.0, 1.0)],
        args=_base_args(bb_minimiser="rbf_surrogate"),
        optimiser=optimiser,
    )

    assert [candidate.point for candidate in candidates] == [(0.2,), (0.8,)]
    assert [candidate.objective for candidate in candidates] == pytest.approx(
        [0.2, 0.8]
    )


def test_no_backend_candidate_uses_valid_warm_start():
    def no_candidates(*_args, **_kwargs):
        raise NoOptimisationCandidatesError("none")

    candidates = adapter.run_hpr_candidate_search(
        objective=lambda x, _args, debug=False: _result(float(x[0])),
        initial_points=np.array([0.3]),
        bounds=[(0.0, 1.0)],
        args=_base_args(),
        optimiser=no_candidates,
    )

    assert candidates == (OptimisationCandidate(objective=0.3, point=(0.3,)),)


def test_unexpected_optimiser_error_propagates_even_with_warm_start():
    def broken_optimizer(*_args, **_kwargs):
        raise RuntimeError("optimizer exploded")

    with pytest.raises(RuntimeError, match="optimizer exploded"):
        adapter.run_hpr_candidate_search(
            objective=lambda x, _args, debug=False: _result(float(x[0])),
            initial_points=np.array([0.3]),
            bounds=[(0.0, 1.0)],
            args=_base_args(),
            optimiser=broken_optimizer,
        )


def test_unexpected_objective_error_propagates_through_adapter():
    def evaluate_once(problem, *, method, options):
        problem.objective(np.array([0.5]), *problem.args)
        raise AssertionError("unreachable")

    with pytest.raises(RuntimeError, match="thermodynamic failure"):
        adapter.run_hpr_candidate_search(
            objective=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("thermodynamic failure")
            ),
            initial_points=None,
            bounds=[(0.0, 1.0)],
            args=_base_args(),
            optimiser=evaluate_once,
        )


def test_fallback_shared_objective_is_weighted_when_costs_are_absent():
    weighted, objective = adapter.aggregate_hpr_period_results(
        {
            "p0": _result(10.0),
            "p1": _result(20.0),
        },
        [1.0, 3.0],
    )

    assert objective == pytest.approx(17.5)
    assert weighted.obj == pytest.approx(17.5)


def test_accounting_applies_refrigeration_penalty_and_scalar_objective():
    external_heat, external_cold, penalty, objective = adapter.build_hpr_accounting(
        work=10.0,
        Q_ext_heat=0.0,
        Q_ext_cold=5.0,
        args=_base_args(
            is_heat_pumping=False,
            heat_to_power_ratio=0.0,
            cold_to_power_ratio=0.0,
            eta_penalty=0.0,
            rho_penalty=2.0,
        ),
        penalise_external_cold_when_refrigerating=True,
    )

    assert external_heat == pytest.approx(0.0)
    assert external_cold == pytest.approx(5.0)
    assert penalty == pytest.approx(50.0)
    assert objective == pytest.approx((10.0 + 50.0) / 200.0)
    assert adapter.calc_hpr_obj(
        10.0,
        5.0,
        0.0,
        100.0,
        heat_to_power_ratio=2.0,
        penalty=1.0,
    ) == pytest.approx(0.21)


def test_result_translation_attaches_ambient_streams_and_validates_output():
    result = adapter.solve_hpr_placement(
        f_obj=lambda x, _args, debug=False: _result(float(x[0])),
        x0_ls=None,
        bnds=[(0.0, 1.0)],
        args=_base_args(),
        candidate_search=lambda **_kwargs: (
            OptimisationCandidate(objective=0.2, point=(0.2,)),
        ),
    )
    output = adapter.translate_hpr_output(result)

    assert result.success is True
    assert isinstance(result.amb_streams, StreamCollection)
    assert output.obj == pytest.approx(0.2)
    assert output.success is True


def test_solver_resolves_only_the_best_ranked_candidate():
    def candidate_search(**_kwargs):
        return (
            OptimisationCandidate(objective=0.2, point=(0.2,)),
            OptimisationCandidate(objective=0.8, point=(0.8,)),
        )

    calls = []

    def objective(point, _args, debug=False):
        calls.append(float(point[0]))
        return _result(float(point[0]))

    result = adapter.solve_hpr_placement(
        f_obj=objective,
        x0_ls=None,
        bnds=[(0.0, 1.0)],
        args=_base_args(),
        candidate_search=candidate_search,
    )

    assert result.obj == pytest.approx(0.2)
    assert calls == [0.2]

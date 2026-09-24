"""Bounded search and CoolProp preflight reliability regressions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

import OpenPinch.analysis.heat_pumps.targeting.parallel_vapour_compression as parallel_vc
from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    run_hpr_candidate_search,
    solve_hpr_placement,
)
from OpenPinch.analysis.heat_pumps.performance_maps.coolprop_preflight import (
    preflight_coolprop_hpr_targeting,
)
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPRParsedState,
    HPRSearchBudget,
    HPRTargetingError,
    HPRTopologyIdentifier,
)
from OpenPinch.optimisation.models import (
    OptimisationCandidate,
    OptimisationResult,
)

from .helpers import _base_args


def _success(value: float = 1.0) -> HPRBackendResult:
    return HPRBackendResult(
        obj=value,
        utility_tot=value,
        w_net=value,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )


@seed(20260924)
@given(
    maximum_iterations=st.integers(min_value=1, max_value=10_000),
    maximum_evaluations=st.integers(min_value=1, max_value=1_000_000),
)
def test_search_budget_maps_exactly_to_generic_options(
    maximum_iterations: int,
    maximum_evaluations: int,
) -> None:
    captured = {}

    def optimiser(problem, *, method, options):
        captured["options"] = options
        return OptimisationResult(method=method, candidates=())

    args = _base_args(
        search_budget=HPRSearchBudget(
            maximum_iterations=maximum_iterations,
            maximum_evaluations=maximum_evaluations,
        )
    )
    run_hpr_candidate_search(
        objective=lambda *_args, **_kwargs: _success(),
        initial_points=[[0.5]],
        bounds=[(0.0, 1.0)],
        args=args,
        optimiser=optimiser,
    )

    assert captured["options"].maxiter == maximum_iterations
    assert captured["options"].maxfun == maximum_evaluations


def test_warm_start_is_evaluated_first_and_exact_repeats_are_cached() -> None:
    events = []
    calls = {"objective": 0}

    def objective(point, _args, **_kwargs):
        calls["objective"] += 1
        events.append(("objective", tuple(point)))
        return _success(float(point[0]))

    def optimiser(problem, *, method, options):
        events.append(("optimiser", None))
        assert problem.objective((0.5)) == pytest.approx(0.5)
        assert problem.objective((0.5)) == pytest.approx(0.5)
        return OptimisationResult(
            method=method,
            candidates=(),
        )

    candidates = run_hpr_candidate_search(
        objective=objective,
        initial_points=[[0.5]],
        bounds=[(0.0, 1.0)],
        args=_base_args(search_budget=HPRSearchBudget()),
        optimiser=optimiser,
    )

    assert events[0] == ("objective", (0.5,))
    assert calls["objective"] == 1
    assert candidates[0].point == (0.5,)


def test_backend_exhaustion_retains_viable_warm_start() -> None:
    def exhausted(*_args, **_kwargs):
        from OpenPinch.optimisation.errors import NoOptimisationCandidatesError

        raise NoOptimisationCandidatesError("budget exhausted")

    candidates = run_hpr_candidate_search(
        objective=lambda *_args, **_kwargs: _success(0.25),
        initial_points=[[0.5]],
        bounds=[(0.0, 1.0)],
        args=_base_args(search_budget=HPRSearchBudget()),
        optimiser=exhausted,
    )

    assert candidates == (OptimisationCandidate(objective=0.25, point=(0.5,)),)


def test_no_viable_candidate_raises_bounded_typed_diagnostics() -> None:
    def objective(*_args, **_kwargs):
        return HPRBackendResult.failure(reason="state outside CoolProp envelope")

    with pytest.raises(HPRTargetingError) as captured:
        solve_hpr_placement(
            f_obj=objective,
            x0_ls=[[0.5]],
            bnds=[(0.0, 1.0)],
            args=_base_args(search_budget=HPRSearchBudget()),
            candidate_search=lambda **_kwargs: (
                OptimisationCandidate(objective=1e30, point=(0.5,)),
            ),
        )

    assert captured.value.diagnostics.evaluated_count == 1
    assert len(captured.value.diagnostics.representative_failures) == 1


def test_coolprop_preflight_covers_vc_and_mvr_stage_fluids(monkeypatch) -> None:
    calls = []

    def resolve(fluid, evaporating_temperature, condensing_temperature):
        calls.append((fluid, evaporating_temperature, condensing_temperature))
        return SimpleNamespace(source_spec=fluid)

    monkeypatch.setattr(
        "OpenPinch.analysis.heat_pumps.performance_maps.coolprop_preflight.resolve_hpr_working_fluid",
        resolve,
    )
    args = _base_args(
        n_cond=1,
        n_evap=1,
        n_mvr=2,
        refrigerant_ls=["R134A"],
        mvr_fluid_ls=["Water", "Water"],
    )
    state = HPRParsedState(
        T_evap=np.array([70.0, 90.0, 40.0]),
        T_cond=np.array([90.0, 110.0, 70.0]),
    )

    prepared = preflight_coolprop_hpr_targeting(
        args=args,
        state=state,
        topology_id=HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
    )

    assert len(prepared.stages) == 3
    assert [stage.role for stage in prepared.stages] == ["mvr", "mvr", "vc"]
    assert [call[0] for call in calls] == ["Water", "Water", "R134A"]


def test_invalid_fluid_preflight_never_enters_generic_optimizer(monkeypatch) -> None:
    optimiser_calls = []
    args = _base_args(
        n_cond=1,
        n_evap=1,
        initialise_simulated_cycle=False,
        refrigerant_ls=["not-a-fluid"],
    )
    monkeypatch.setattr(
        parallel_vc,
        "validate_vapour_hp_refrigerant_ls",
        lambda _count, _args: ["not-a-fluid"],
    )
    monkeypatch.setattr(
        parallel_vc,
        "solve_hpr_placement",
        lambda **kwargs: optimiser_calls.append(kwargs),
    )

    with pytest.raises(HPRTargetingError) as captured:
        parallel_vc.optimise_parallel_heat_pump_placement(args)

    assert optimiser_calls == []
    diagnostic = captured.value.diagnostics.representative_failures[0]
    assert diagnostic.fluid == "not-a-fluid"
    assert diagnostic.stage_index == 0


def test_named_public_budget_overrides_runtime_options(monkeypatch) -> None:
    monkeypatch.setattr(
        _TargetAccessor,
        "_execute",
        lambda _self, **kwargs: kwargs,
    )
    accessor = _TargetAccessor(SimpleNamespace())

    captured = accessor.carnot_heat_pump(
        options={"maximum_iterations": 9, "maximum_evaluations": 90},
        maximum_iterations=3,
        maximum_evaluations=30,
    )

    assert captured["options"]["maximum_iterations"] == 3
    assert captured["options"]["maximum_evaluations"] == 30


@pytest.mark.parametrize(
    ("field", "value"),
    [("maximum_iterations", 0), ("maximum_evaluations", True)],
)
def test_invalid_public_budget_fails_before_target_execution(
    monkeypatch,
    field: str,
    value: object,
) -> None:
    calls = []
    monkeypatch.setattr(
        _TargetAccessor,
        "_execute",
        lambda _self, **kwargs: calls.append(kwargs),
    )
    accessor = _TargetAccessor(SimpleNamespace())

    with pytest.raises(ValueError):
        accessor.carnot_heat_pump(**{field: value})

    assert calls == []

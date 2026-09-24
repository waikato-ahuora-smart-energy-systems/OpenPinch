"""Independent regressions for the post-construction reliability audit."""

import pickle
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.direct_mvr.models import DirectGasMVRStageError
from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    normalise_hpr_penalty_terms,
    run_hpr_candidate_search,
)
from OpenPinch.analysis.heat_pumps.performance_maps.coolprop_preflight import (
    preflight_coolprop_hpr_targeting,
)
from OpenPinch.analysis.heat_pumps.performance_maps.target_records import (
    build_coolprop_target_simulation_record,
)
from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPRFailureCategory,
    HPRParsedState,
    HPRSearchBudget,
    HPRTargetingError,
    HPRTopologyIdentifier,
)
from OpenPinch.optimisation.models import OptimisationResult

from .helpers import _base_args


def success(value):
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
    terms=st.lists(
        st.lists(
            st.floats(
                min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            max_size=5,
        ),
        min_size=1,
        max_size=5,
    )
)
def test_network_penalties_sum_unequal_stage_vectors(terms):
    from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
        CascadeVapourCompressionCycle,
    )
    from OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles import (
        ParallelVapourCompressionCycles,
    )

    for cycle_type in (CascadeVapourCompressionCycle, ParallelVapourCompressionCycles):
        cycle = cycle_type()
        cycle._solved = True
        cycle._subcycles = [
            SimpleNamespace(solved=True, penalty=np.asarray(row)) for row in terms
        ]
        assert cycle.penalty == pytest.approx(sum(sum(row) for row in terms))


@seed(20260924)
@given(limit=st.integers(min_value=1, max_value=20))
def test_budget_is_hard_across_warm_starts_restarts_and_backend_overruns(limit):
    calls = []

    def objective(point, args, **kwargs):
        calls.append(tuple(point))
        return success(float(point[0]))

    def optimiser(problem, *, method, options):
        for value in np.linspace(0.01, 0.99, 100):
            problem.objective([value])
        return OptimisationResult(method=method, candidates=())

    candidates = run_hpr_candidate_search(
        objective=objective,
        initial_points=[[0.5]],
        bounds=[(0.0, 1.0)],
        args=_base_args(
            max_multi_start=3, search_budget=HPRSearchBudget(maximum_evaluations=limit)
        ),
        optimiser=optimiser,
    )
    assert len(calls) == limit
    assert candidates[0].objective == min(point[0] for point in calls)
    assert candidates.diagnostics.evaluated_count == limit
    assert (
        candidates.diagnostics.category_counts[HPRFailureCategory.BUDGET_EXHAUSTION]
        == 1
    )


@pytest.mark.parametrize("value", ["1", ["1", "2"], 1 + 2j, np.array([1 + 0j])])
def test_penalties_reject_non_real_numeric_types(value):
    with pytest.raises(TypeError):
        normalise_hpr_penalty_terms(value)


def test_invalid_seed_is_not_a_global_capability_rejection():
    prepared = preflight_coolprop_hpr_targeting(
        args=_base_args(n_cond=1, n_evap=1, refrigerant_ls=["R134a"]),
        state=HPRParsedState(T_evap=np.array([20.0]), T_cond=np.array([150.0])),
        topology_id=HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
    )
    assert not prepared.stages[0].representative_state_supported
    viable = preflight_coolprop_hpr_targeting(
        args=_base_args(n_cond=1, n_evap=1, refrigerant_ls=["R134a"]),
        state=HPRParsedState(T_evap=np.array([20.0]), T_cond=np.array([60.0])),
        topology_id=HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
    )
    assert viable.stages[0].representative_state_supported


def test_invalid_fluid_rejected_before_default_carnot_initialization(monkeypatch):
    import OpenPinch.analysis.heat_pumps.targeting.parallel_vapour_compression as target

    def forbidden(*args, **kwargs):
        pytest.fail("invalid fluid entered Carnot initialization")

    monkeypatch.setattr(
        target, "optimise_parallel_carnot_heat_pump_placement", forbidden
    )
    with pytest.raises(HPRTargetingError) as caught:
        target.optimise_parallel_heat_pump_placement(
            _base_args(initialise_simulated_cycle=True, refrigerant_ls=["not-a-fluid"])
        )
    for error in (deepcopy(caught.value), pickle.loads(pickle.dumps(caught.value))):
        assert error.diagnostics == caught.value.diagnostics


def test_mvr_error_survives_copy_and_process_transport():
    error = DirectGasMVRStageError(
        reason_code="coolprop.required_state_unavailable",
        source_stream="steam",
        period_index=0,
        stage_index=2,
        fluid="Water",
    )
    assert deepcopy(error).stage_index == 2
    assert pickle.loads(pickle.dumps(error)).fluid == "Water"


def test_asymmetric_vc_mvr_record_preserves_zero_duty_stage_and_roles():
    args = _base_args(
        n_cond=1,
        n_evap=2,
        n_mvr=1,
        refrigerant_ls=["R134a", "R123"],
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
        is_heat_pumping=True,
    )
    cycle = SimpleNamespace(
        Q_heat_arr=np.array([5.0, 0.0, 10.0]),
        Q_cool_arr=np.array([4.0, 0.0, 0.0]),
        work_arr=np.array([1.0, 0.0, 2.0]),
        T_evap=np.array([20.0, 30.0, 80.0]),
        T_cond=np.array([80.0, 90.0, 100.0]),
    )
    record = build_coolprop_target_simulation_record(
        args=args,
        state=HPRParsedState(T_evap=cycle.T_evap, T_cond=cycle.T_cond),
        cycle=cycle,
        topology_id=HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
    )
    assert [loop.fluid_spec for loop in record.loops] == ["R134a", "R123", "Water"]
    assert record.loops[1].nominal_duty == 0.0
    assert record.nominal_useful_duty == 15.0
    assert (record.evaporator_count, record.condenser_count) == (2, 1)


def test_fluid_resolution_does_not_hide_internal_errors(monkeypatch):
    import OpenPinch.analysis.heat_pumps.performance_maps.fluids as fluids

    def broken(*args):
        raise RuntimeError("implementation defect")

    monkeypatch.setattr(fluids, "build_coolprop_abstract_state", broken)
    with pytest.raises(RuntimeError, match="implementation defect"):
        fluids.resolve_hpr_working_fluid("Water", 30.0, 100.0)


def test_fatal_error_during_local_polishing_propagates(monkeypatch):
    import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter
    from OpenPinch.optimisation.models import OptimisationCandidate

    def objective(point, args, **kwargs):
        if point[0] != 0.5:
            raise RuntimeError("local defect")
        return success(1.0)

    def optimiser(problem, *, method, options):
        assert options.local_method is None
        return OptimisationResult(
            method=method,
            candidates=(OptimisationCandidate(objective=1.0, point=(0.5,)),),
        )

    def polish(function, *args, **kwargs):
        return function([0.6])

    monkeypatch.setattr(adapter, "minimize", polish)
    with pytest.raises(RuntimeError, match="local defect"):
        run_hpr_candidate_search(
            objective=objective,
            initial_points=[[0.5]],
            bounds=[(0.0, 1.0)],
            args=_base_args(),
            optimiser=optimiser,
        )


def test_final_record_validation_is_not_candidate_infeasibility(monkeypatch):
    import OpenPinch.analysis.heat_pumps.targeting.vapour_compression_mvr as target

    def broken_record(**kwargs):
        raise ValueError("record contract defect")

    monkeypatch.setattr(
        target, "build_coolprop_target_simulation_record", broken_record
    )
    args = _base_args(
        n_cond=1,
        n_evap=1,
        n_mvr=1,
        refrigerant_ls=["R134A"],
        mvr_fluid_ls=["Water"],
        eta_mvr_comp=0.7,
        eta_motor=0.95,
        initialise_simulated_cycle=False,
    )
    point = np.array([0.0, 0.6, 0.1, 0.1, 0.1, 0.5, 0.5, 0.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="record contract defect"):
        target._compute_vc_mvr_system_obj(point, args)


def test_direct_mvr_second_stage_failure_leaves_problem_unchanged(monkeypatch):
    import OpenPinch.analysis.heat_pumps.direct_mvr.execution as execution
    from OpenPinch import PinchProblem

    problem = PinchProblem("process_mvr.json")
    before = problem.to_problem_json()
    solve_stage = execution._solve_compression_stage
    stages = []

    def second_stage_failure(**kwargs):
        stages.append(kwargs["stage_index"])
        if kwargs["stage_index"] == 2:
            raise ValueError("unavailable second-stage state")
        return solve_stage(**kwargs)

    monkeypatch.setattr(execution, "_solve_compression_stage", second_stage_failure)
    with pytest.raises(DirectGasMVRStageError) as caught:
        problem.components.add_process_mvr(
            "Evaporator vapour", n_stages=2, mvr_stage_t_lift=10.0
        )
    assert stages == [1, 2]
    assert caught.value.stage_index == 2
    assert problem.to_problem_json() == before


def test_failure_summary_preserves_search_counts():
    from OpenPinch.analysis.heat_pumps.optimisation_adapter import solve_hpr_placement

    def objective(*args, **kwargs):
        return HPRBackendResult.failure(reason="physical state outside envelope")

    with pytest.raises(HPRTargetingError) as caught:
        solve_hpr_placement(
            objective,
            [[0.5]],
            [(0.0, 1.0)],
            _base_args(search_budget=HPRSearchBudget(maximum_evaluations=2)),
        )
    summary = caught.value.diagnostics
    assert summary.evaluated_count == 3  # two search + one final warm-start check
    assert summary.category_counts[HPRFailureCategory.BUDGET_EXHAUSTION] == 1
    assert (
        summary.category_counts[HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY]
        == 3
    )


@pytest.mark.parametrize("point", [[float("nan")], [2.0], [0.5, 0.5]])
def test_invalid_warm_start_never_evaluates_objective(point):
    def objective(*args, **kwargs):
        pytest.fail("invalid point reached expensive objective")

    with pytest.raises(ValueError, match="initial points"):
        run_hpr_candidate_search(
            objective=objective,
            initial_points=[point],
            bounds=[(0.0, 1.0)],
            args=_base_args(),
        )


def test_serial_restarts_rotate_prepared_starts_and_seeds():
    observed = []

    def optimiser(problem, *, method, options):
        observed.append((problem.initial_points[0], options.seed, options.n_runs))
        return OptimisationResult(method=method, candidates=())

    run_hpr_candidate_search(
        objective=lambda *args, **kwargs: success(1.0),
        initial_points=[[0.2], [0.8]],
        bounds=[(0.0, 1.0)],
        args=_base_args(max_multi_start=3),
        optimiser=optimiser,
    )
    assert observed == [((0.2,), 0, 1), ((0.8,), 1, 1), ((0.2,), 2, 1)]


def test_optional_seed_failure_does_not_abort_physical_search():
    from OpenPinch.analysis.heat_pumps.optimisation_adapter import initialise_hpr_seed
    from OpenPinch.contracts.hpr import HPRFailureSummary

    diagnostic = HPRFailureSummary(
        simulation_backend="coolprop",
        cycle="screening",
        evaluated_count=1,
        category_counts={},
        budget=HPRSearchBudget(),
        warm_start_evaluated=True,
        warm_start_viable=False,
    )

    def unavailable(args):
        raise HPRTargetingError("no screening result", diagnostics=diagnostic)

    assert initialise_hpr_seed(unavailable, _base_args()) is None

    def defect(args):
        raise RuntimeError("screening implementation defect")

    with pytest.raises(RuntimeError, match="implementation defect"):
        initialise_hpr_seed(defect, _base_args())


@seed(20260924)
@given(
    points=st.lists(st.integers(min_value=0, max_value=8), max_size=40),
    budget=st.integers(min_value=1, max_value=8),
)
def test_cache_command_sequences_match_bounded_reference_model(points, budget):
    from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
        _CachedHPRScalarObjective,
        _HPRSearchBudgetExhausted,
    )

    calls = []

    def objective(point, args, **kwargs):
        value = point[0]
        calls.append(value)
        return success(value) if value % 2 else HPRBackendResult.failure(reason="state")

    cached = _CachedHPRScalarObjective(
        objective, _base_args(search_budget=HPRSearchBudget(maximum_evaluations=budget))
    )
    model = {}
    for point in points:
        key = (float(point),)
        if key not in model and len(model) == budget:
            with pytest.raises(_HPRSearchBudgetExhausted):
                cached(key)
        else:
            model.setdefault(key, float(point) if point % 2 else 1e30)
            assert cached(key) == model[key]
        assert cached.cache == model
        assert len(calls) == len(model)
        assert set(cached.viable) == {key for key in model if key[0] % 2}


@pytest.mark.parametrize("field", ["weighted_output", "hpr_operating_cost"])
def test_copyable_arbitrary_objects_cannot_cross_public_boundary(field):
    from OpenPinch.analysis.heat_pumps.optimisation_adapter import translate_hpr_output
    from OpenPinch.contracts.hpr import HPRThermoArtifacts
    from OpenPinch.domain.stream_collection import StreamCollection

    result = success(1.0).with_updates(
        amb_streams=StreamCollection(),
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
        **{field: object()},
    )
    with pytest.raises(TypeError, match="unsupported object"):
        translate_hpr_output(result)

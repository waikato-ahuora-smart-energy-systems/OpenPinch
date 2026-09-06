"""Real and classified-failure tests for TESPy nominal target evaluation."""

# ruff: noqa: E402 - optional dependency gate precedes concrete imports.

from __future__ import annotations

import gc
import math
import time
import weakref

import numpy as np
import pytest

pytest.importorskip("tespy")
pytestmark = pytest.mark.tespy

import OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy as adapter
import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as cascade
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.optimisation_adapter import translate_hpr_result
from OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy import (
    TespyHprTargetEvaluator,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting import (
    HprTargetEvaluatorCoordinator,
    get_hpr_target_evaluator,
    preflight_tespy_hpr_targeting,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting_models import (
    HprTargetEvaluationFailure,
    HprTargetEvaluatorError,
    HprTargetThermodynamicResult,
)
from OpenPinch.contracts.hpr_performance_map import HprPerformanceMapRequest
from tests.analysis.heat_pumps.test_hpr_target_evaluator_contracts import _request
from tests.analysis.heat_pumps.test_hpr_targeting_preflight import _target_args
from tests.analysis.heat_pumps.test_hpr_tespy_targeting_integration import _candidate


def _coordinator(*, mode="heat_pump", refrigerant="R134a"):
    prepared = preflight_tespy_hpr_targeting(
        _target_args(
            is_heat_pumping=mode == "heat_pump",
            refrigerant_ls=[refrigerant],
        )
    )
    evaluator = TespyHprTargetEvaluator(prepared)
    coordinator = HprTargetEvaluatorCoordinator(evaluator)
    metadata = coordinator.open()
    request = _request(
        mode=mode,
        model_id=prepared.model_id,
        working_fluid=prepared.working_fluid,
        evaporating_temperature=5.0,
        condensing_temperature=55.0,
    )
    return evaluator, coordinator, metadata, request


@pytest.mark.parametrize("mode", ["heat_pump", "refrigeration"])
@pytest.mark.parametrize(
    "refrigerant",
    [
        "R134a",
        "R407C",
        "HEOS::R32[0.5]&R125[0.5]",
        "HEOS::R32[0.3]&R125[0.3]&R143a[0.4]",
    ],
)
def test_real_tespy_target_design_supports_modes_and_fluid_categories(
    mode: str,
    refrigerant: str,
) -> None:
    evaluator, coordinator, metadata, request = _coordinator(
        mode=mode,
        refrigerant=refrigerant,
    )

    result = coordinator.evaluate(request)
    coordinator.close()

    assert isinstance(result, HprTargetThermodynamicResult)
    assert metadata.backend == result.backend == "tespy"
    assert metadata.model_id == result.model_id == request.model_id
    assert result.working_fluid == request.working_fluid
    assert all(
        math.isfinite(value) and value > 0.0
        for value in (result.q_source, result.q_sink, result.compressor_power)
    )
    assert result.q_sink == pytest.approx(
        result.q_source + result.compressor_power,
        abs=1e-5,
    )
    useful = result.q_sink if mode == "heat_pump" else result.q_source
    assert useful == pytest.approx(request.useful_duty, abs=1e-5)
    assert result.cop == pytest.approx(useful / result.compressor_power)
    assert result.source_profile[-1].enthalpy == pytest.approx(result.q_source)
    assert result.sink_profile[-1].enthalpy == pytest.approx(result.q_sink)
    assert evaluator.design_solve_count == 1
    assert evaluator.state == "closed"


def test_every_target_candidate_uses_a_fresh_design_solve_not_offdesign(
    monkeypatch,
) -> None:
    solve_modes: list[str] = []
    original = adapter.TespyHprPointSimulator._solve

    def capture(self, mode, **paths):
        solve_modes.append(mode)
        return original(self, mode, **paths)

    monkeypatch.setattr(adapter.TespyHprPointSimulator, "_solve", capture)
    evaluator, coordinator, _metadata, request = _coordinator()

    first = coordinator.evaluate(request)
    second = coordinator.evaluate(
        _request(
            model_id=request.model_id,
            working_fluid=request.working_fluid,
            useful_duty=120.0,
            candidate_id="second",
        )
    )
    coordinator.close()

    assert isinstance(first, HprTargetThermodynamicResult)
    assert isinstance(second, HprTargetThermodynamicResult)
    assert solve_modes == ["design", "design"]
    assert evaluator.design_solve_count == 2


def test_candidate_design_failure_is_local_and_later_candidate_runs(
    monkeypatch,
) -> None:
    original = adapter.TespyHprPointSimulator._solve
    calls = 0

    def fail_once(self, mode, **paths):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("candidate failed")
        return original(self, mode, **paths)

    monkeypatch.setattr(adapter.TespyHprPointSimulator, "_solve", fail_once)
    _evaluator, coordinator, _metadata, request = _coordinator()

    first = coordinator.evaluate(request)
    second = coordinator.evaluate(
        _request(
            model_id=request.model_id,
            working_fluid=request.working_fluid,
            useful_duty=120.0,
            candidate_id="second",
        )
    )
    coordinator.close()

    assert isinstance(first, HprTargetEvaluationFailure)
    assert first.session_fatal is False
    assert isinstance(second, HprTargetThermodynamicResult)


def test_candidate_cleanup_failure_is_session_fatal(monkeypatch) -> None:
    monkeypatch.setattr(
        adapter.TespyHprPointSimulator,
        "close",
        lambda self: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    _evaluator, coordinator, _metadata, request = _coordinator()

    with pytest.raises(HprTargetEvaluatorError) as captured:
        coordinator.evaluate(request)
    coordinator.close()

    assert captured.value.code == "cleanup_failure"
    assert captured.value.session_fatal is True


def test_real_result_contains_no_tespy_object() -> None:
    _evaluator, coordinator, _metadata, request = _coordinator()
    result = coordinator.evaluate(request)
    coordinator.close()

    assert isinstance(result, HprTargetThermodynamicResult)
    for field_name in result.__dataclass_fields__:
        value = getattr(result, field_name)
        assert not type(value).__module__.startswith("tespy")


def test_lazy_factory_returns_a_fresh_concrete_target_evaluator() -> None:
    prepared = preflight_tespy_hpr_targeting(_target_args())

    first = get_hpr_target_evaluator(prepared)
    second = get_hpr_target_evaluator(prepared)

    assert isinstance(first, TespyHprTargetEvaluator)
    assert isinstance(second, TespyHprTargetEvaluator)
    assert first is not second


def test_real_candidate_results_are_independent_of_evaluation_order() -> None:
    _first_evaluator, first, _metadata, base = _coordinator()
    request_a = base
    request_b = _request(
        model_id=base.model_id,
        working_fluid=base.working_fluid,
        useful_duty=120.0,
        candidate_id="b",
    )
    forward = {
        request.useful_duty: first.evaluate(request)
        for request in (request_a, request_b)
    }
    first.close()

    _second_evaluator, second, _metadata, _base = _coordinator()
    reverse = {
        request.useful_duty: second.evaluate(request)
        for request in (request_b, request_a)
    }
    second.close()

    for duty in forward:
        assert isinstance(forward[duty], HprTargetThermodynamicResult)
        assert isinstance(reverse[duty], HprTargetThermodynamicResult)
        assert reverse[duty].q_source == pytest.approx(forward[duty].q_source)
        assert reverse[duty].q_sink == pytest.approx(forward[duty].q_sink)
        assert reverse[duty].compressor_power == pytest.approx(
            forward[duty].compressor_power
        )


def test_real_tespy_candidate_enters_existing_hpr_accounting_pipeline() -> None:
    args = _target_args(
        initialise_simulated_cycle=False,
        max_multi_start=1,
        T_cold=np.array([100.0, 55.0, 30.0]),
        H_cold=np.array([200.0, 100.0, 0.0]),
    )
    prepared = preflight_tespy_hpr_targeting(args)
    evaluator = TespyHprTargetEvaluator(prepared)
    coordinator = HprTargetEvaluatorCoordinator(evaluator)
    coordinator.open()

    point = _candidate(args)
    point[1] = 45.0 / 70.0
    point[2] = 1.0
    point[3] = 2.0 / 45.0
    result = cascade._compute_tespy_cascade_hp_system_obj(
        point,
        args,
        coordinator=coordinator,
        prepared=prepared,
    )
    coordinator.close()

    assert result.success is True
    assert result.simulation_backend == "tespy"
    assert result.model is None
    assert result.w_net > 0.0
    assert result.Q_heat[0] == pytest.approx(result.Q_cool[0] + result.w_net)
    assert result.cop_h == pytest.approx(result.Q_heat[0] / result.w_net)
    assert result.hpr_hot_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        result.Q_heat[0]
    )
    assert result.hpr_cold_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        result.Q_cool[0]
    )
    assert result.hpr_total_annualized_cost is not None
    assert np.isfinite(result.obj)


def test_three_real_evaluator_calls_release_engine_objects_and_private_state(
    monkeypatch,
) -> None:
    simulator_refs = []
    network_refs = []
    closed_states = []
    concrete_simulator = adapter.TespyHprPointSimulator

    class TrackingSimulator(concrete_simulator):
        def __init__(self) -> None:
            super().__init__()
            simulator_refs.append(weakref.ref(self))

        def _build_network(self, context) -> None:
            super()._build_network(context)
            network_refs.append(weakref.ref(self._network))

        def close(self) -> None:
            super().close()
            closed_states.append(
                (
                    self._state,
                    self._temporary_directory,
                    self._design_state_path,
                    self._network,
                )
            )

    monkeypatch.setattr(adapter, "TespyHprPointSimulator", TrackingSimulator)

    for ordinal in range(3):
        evaluator, coordinator, _metadata, request = _coordinator()
        result = coordinator.evaluate(
            _request(
                model_id=request.model_id,
                working_fluid=request.working_fluid,
                candidate_id=f"real-lifecycle-{ordinal}",
            )
        )
        coordinator.close()
        assert isinstance(result, HprTargetThermodynamicResult)
        assert evaluator.state == coordinator.state == "closed"
        assert coordinator.cache_size == 0
        del result, request, coordinator, evaluator

    gc.collect()
    assert closed_states == [("closed", None, None, None)] * 3
    assert all(reference() is None for reference in simulator_refs)
    assert all(reference() is None for reference in network_refs)


def test_real_public_tespy_target_and_minimal_map_stay_within_smoke_budget(
    monkeypatch,
    record_property,
) -> None:
    def bounded_candidate_search(*, f_obj, x0_ls, bnds, args):
        point = np.array([(lower + upper) / 2.0 for lower, upper in bnds])
        point[0] = 0.0
        point[1] = 0.35
        point[2] = 0.20
        point[3] = 0.02
        point[4] = 0.50
        point[5] = 1.0
        point[-1] = 0.0
        result = f_obj(point, args)
        assert result.success is True
        return translate_hpr_result(result, ambient_args=args)

    monkeypatch.setattr(cascade, "solve_hpr_placement", bounded_candidate_search)
    problem = PinchProblem("pulp_mill.json")
    started = time.perf_counter()

    target_started = time.perf_counter()
    target = problem.target.vapour_compression_heat_pump(
        simulation_backend="tespy",
        condensers=1,
        evaporators=1,
        refrigerants=["ammonia"],
        initialize_from_carnot=False,
        allow_integrated_expander=False,
        load_fraction=0.25,
        maximum_restarts=1,
    )
    target_elapsed = time.perf_counter() - target_started
    record = target.hpr_details.target_simulation_record
    assert record is not None
    assert record.simulation_backend == "tespy"

    map_started = time.perf_counter()
    performance_map = problem.target.hpr_performance_map(
        target=target,
        request=HprPerformanceMapRequest(
            map_id="real-public-tespy-smoke",
            source_temperatures=[
                record.nominal_evaporating_temperature
                + record.source_approach_temperature
            ],
            sink_temperatures=[
                record.nominal_condensing_temperature - record.sink_approach_temperature
            ],
            load_fractions=[0.75, 1.0],
        ),
    )
    map_elapsed = time.perf_counter() - map_started
    total_elapsed = time.perf_counter() - started
    record_property("tespy_target_elapsed_seconds", target_elapsed)
    record_property("tespy_map_elapsed_seconds", map_elapsed)
    record_property("tespy_total_elapsed_seconds", total_elapsed)

    assert target.hpr_success is True
    assert performance_map.thermodynamic_backend == "tespy"
    assert len(performance_map.points) == 2
    assert [point.load_fraction for point in performance_map.points] == [0.75, 1.0]
    assert performance_map.points[0].q_sink < performance_map.points[1].q_sink
    assert (
        performance_map.points[0].electric_power
        < performance_map.points[1].electric_power
    )
    assert all(point.cop > 0.0 for point in performance_map.points)
    assert total_elapsed < 300.0

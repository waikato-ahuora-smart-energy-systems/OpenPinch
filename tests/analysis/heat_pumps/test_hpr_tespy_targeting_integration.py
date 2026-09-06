"""TESPy integration contracts at the current scalar HPR optimizer boundary."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.service as hp_service
import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as cascade
from OpenPinch.analysis.heat_pumps.performance_maps.targeting import (
    HprTargetEvaluatorCoordinator,
    preflight_tespy_hpr_targeting,
)
from OpenPinch.analysis.heat_pumps.performance_maps.targeting_models import (
    HprTargetEvaluatorError,
)
from OpenPinch.contracts.hpr import HPRBackendResult, HPRThermoArtifacts
from OpenPinch.domain.stream_collection import StreamCollection
from tests.analysis.heat_pumps.helpers import _patch_output_model_validate
from tests.analysis.heat_pumps.hpr_targeting_fakes import FakeHprTargetEvaluator
from tests.analysis.heat_pumps.test_hpr_targeting_preflight import _target_args


def _candidate(args) -> np.ndarray:
    _starts, bounds = cascade._get_cascade_hp_opt_setup(None, args)
    point = np.array([(lower + upper) / 2.0 for lower, upper in bounds])
    point[0] = 0.0
    point[-1] = 0.0
    return point


def _prepared_args(**overrides):
    args = _target_args(
        initialise_simulated_cycle=False,
        max_multi_start=1,
        **overrides,
    )
    return args, preflight_tespy_hpr_targeting(args)


def _coordinator(fake: FakeHprTargetEvaluator) -> HprTargetEvaluatorCoordinator:
    coordinator = HprTargetEvaluatorCoordinator(fake)
    coordinator.open()
    return coordinator


def test_tespy_objective_uses_evaluator_power_cop_profiles_and_accounting(
    monkeypatch,
) -> None:
    args, prepared = _prepared_args()
    fake = FakeHprTargetEvaluator(cop=5.0)
    coordinator = _coordinator(fake)
    monkeypatch.setattr(
        cascade,
        "CascadeVapourCompressionCycle",
        lambda: pytest.fail("CoolProp target cycle must not be constructed"),
    )

    result = cascade._compute_tespy_cascade_hp_system_obj(
        _candidate(args),
        args,
        coordinator=coordinator,
        prepared=prepared,
    )
    coordinator.close()

    request = fake.requests[0]
    assert result.success is True
    assert result.simulation_backend == "tespy"
    assert result.model is None
    assert result.w_net == pytest.approx(request.useful_duty / 5.0)
    assert result.cop_h == pytest.approx(5.0)
    assert result.Q_heat.tolist() == pytest.approx([request.useful_duty])
    assert result.Q_cool.tolist() == pytest.approx([request.useful_duty - result.w_net])
    assert result.hpr_hot_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        result.Q_heat[0]
    )
    assert result.hpr_cold_streams.sum_stream_attribute("heat_flow") == pytest.approx(
        result.Q_cool[0]
    )
    assert result.hpr_total_annualized_cost is not None
    assert result.obj == pytest.approx(
        result.hpr_total_annualized_cost.to("$/y").value + result.feasibility_penalty
    )
    record = result.target_simulation_record
    assert record is not None
    assert record.simulation_backend == "tespy"
    assert record.model_id == request.model_id
    assert record.refrigerant_spec == request.working_fluid.source_spec
    assert record.nominal_useful_duty == pytest.approx(request.useful_duty)
    assert record.engine_version == "fake-1.0"


@pytest.mark.parametrize("is_heat_pumping", [True, False])
def test_tespy_request_uses_current_mode_and_allocated_primary_duty(
    is_heat_pumping: bool,
) -> None:
    args, prepared = _prepared_args(is_heat_pumping=is_heat_pumping)
    fake = FakeHprTargetEvaluator()
    coordinator = _coordinator(fake)

    result = cascade._compute_tespy_cascade_hp_system_obj(
        _candidate(args),
        args,
        coordinator=coordinator,
        prepared=prepared,
    )
    coordinator.close()

    request = fake.requests[0]
    assert result.success is True
    assert request.mode == ("heat_pump" if is_heat_pumping else "refrigeration")
    expected = result.Q_heat[0] if is_heat_pumping else result.Q_cool[0]
    assert request.useful_duty == pytest.approx(expected)
    assert result.cop_h == pytest.approx(4.0)


def test_optimizer_owns_one_open_cache_and_close_for_complete_call(
    monkeypatch,
) -> None:
    args, prepared = _prepared_args()
    fake = FakeHprTargetEvaluator()
    observed_states: list[str] = []

    def solve(*, f_obj: Callable, x0_ls, bnds, args):
        observed_states.append(fake.state)
        point = _candidate(args)
        first = f_obj(point, args)
        second = f_obj(point.copy(), args)
        assert first.obj == second.obj
        return first

    monkeypatch.setattr(cascade, "get_hpr_target_evaluator", lambda _prepared: fake)
    monkeypatch.setattr(cascade, "solve_hpr_placement", solve)

    result = cascade.optimise_cascade_heat_pump_placement(
        args,
        prepared_tespy=prepared,
    )

    assert result.simulation_backend == "tespy"
    assert observed_states == ["ready"]
    assert fake.open_calls == fake.close_calls == 1
    assert fake.evaluate_calls == 1
    assert fake.state == "closed"


def test_tespy_objective_values_determine_candidate_ranking_and_final_target(
    monkeypatch,
) -> None:
    args, prepared = _prepared_args()
    fake = FakeHprTargetEvaluator(
        cop=lambda request: 8.0 if request.subcooling > 0.0 else 2.0
    )

    def solve(*, f_obj: Callable, x0_ls, bnds, args):
        low_cop = _candidate(args)
        low_cop[3] = 0.0
        high_cop = low_cop.copy()
        high_cop[3] = 0.5
        results = [f_obj(point, args) for point in (low_cop, high_cop)]
        assert results[1].w_net < results[0].w_net
        assert results[1].obj < results[0].obj
        return min(results, key=lambda result: result.obj)

    monkeypatch.setattr(cascade, "get_hpr_target_evaluator", lambda _prepared: fake)
    monkeypatch.setattr(cascade, "solve_hpr_placement", solve)
    monkeypatch.setattr(
        cascade,
        "CascadeVapourCompressionCycle",
        lambda: pytest.fail("CoolProp target cycle must not be constructed"),
    )

    result = cascade.optimise_cascade_heat_pump_placement(
        args,
        prepared_tespy=prepared,
    )

    assert result.cop_h == pytest.approx(8.0)
    assert result.w_net == pytest.approx(fake.requests[-1].useful_duty / 8.0)
    assert fake.evaluate_calls == 2
    assert fake.close_calls == 1


@pytest.mark.parametrize(
    ("behavior", "expected_exception"),
    [
        ("local_failure", ValueError),
        ("fatal_failure", HprTargetEvaluatorError),
    ],
)
def test_all_local_failure_or_fatal_session_closes_once(
    monkeypatch,
    behavior: str,
    expected_exception: type[Exception],
) -> None:
    args, prepared = _prepared_args()
    fake = FakeHprTargetEvaluator((behavior,))

    def solve(*, f_obj: Callable, x0_ls, bnds, args):
        result = f_obj(_candidate(args), args)
        if not result.success:
            raise ValueError("all candidates failed")
        return result

    monkeypatch.setattr(cascade, "get_hpr_target_evaluator", lambda _prepared: fake)
    monkeypatch.setattr(cascade, "solve_hpr_placement", solve)

    with pytest.raises(expected_exception):
        cascade.optimise_cascade_heat_pump_placement(
            args,
            prepared_tespy=prepared,
        )

    assert fake.close_calls == 1
    assert fake.state == "closed"


def test_optimizer_error_still_closes_and_cleanup_error_is_not_hidden(
    monkeypatch,
) -> None:
    args, prepared = _prepared_args()
    fake = FakeHprTargetEvaluator(cleanup_failure=True)
    monkeypatch.setattr(cascade, "get_hpr_target_evaluator", lambda _prepared: fake)
    monkeypatch.setattr(
        cascade,
        "solve_hpr_placement",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("optimizer failed")),
    )

    with pytest.raises(HprTargetEvaluatorError) as captured:
        cascade.optimise_cascade_heat_pump_placement(
            args,
            prepared_tespy=prepared,
        )

    assert captured.value.code == "cleanup_failure"
    assert fake.close_calls == 1


def test_service_preflights_once_and_passes_prepared_value_to_optimizer(
    monkeypatch,
) -> None:
    args, prepared = _prepared_args()
    calls: list[object] = []
    result = HPRBackendResult(
        obj=1.0,
        utility_tot=1.0,
        w_net=1.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        simulation_backend="tespy",
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
        amb_streams=StreamCollection(),
        success=True,
    )
    monkeypatch.setattr(hp_service, "construct_HPRTargetInputs", lambda **_kw: args)
    monkeypatch.setattr(
        hp_service,
        "preflight_tespy_hpr_targeting",
        lambda received: calls.append(received) or prepared,
    )
    monkeypatch.setitem(
        hp_service._HP_PLACEMENT_HANDLERS,
        args.hpr_type,
        lambda received, *, prepared_tespy: (
            calls.append((received, prepared_tespy)) or result
        ),
    )
    _patch_output_model_validate(monkeypatch)

    output = hp_service._get_hpr_targets(
        Q_hpr_target=10.0,
        T_vals=np.array([100.0, 50.0]),
        H_hot=np.array([0.0, -10.0]),
        H_cold=np.array([10.0, 0.0]),
        config=object(),
        is_heat_pumping=True,
        simulation_backend="tespy",
    )

    assert output["simulation_backend"] == "tespy"
    assert calls == [args, (args, prepared)]

"""Numerical inverse targeting for a global heat-recovery dt_min."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.heat_recovery_dt_min import (
    evaluate_process_heat_recovery,
    solve_heat_recovery_dt_min,
)
from OpenPinch.contracts.heat_recovery_dt_min import HeatRecoveryDtMinStatus
from OpenPinch.domain.stream_collection import StreamCollection


def _two_stream_problem() -> PinchProblem:
    return PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site/Process",
                    "name": "Hot",
                    "t_supply": 200.0,
                    "t_target": 100.0,
                    "heat_flow": 100.0,
                    "dt_cont": 17.0,
                    "htc": 1.0,
                },
                {
                    "zone": "Site/Process",
                    "name": "Cold",
                    "t_supply": 50.0,
                    "t_target": 150.0,
                    "heat_flow": 100.0,
                    "dt_cont": 3.0,
                    "htc": 1.0,
                },
            ],
            "utilities": [],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Process", "type": "Process Zone"}],
            },
        },
        project_name="Site",
    )


def test_analytical_two_stream_interior_and_zero_boundary() -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    assert evaluate_process_heat_recovery(
        zone.hot_streams,
        zone.cold_streams,
        dt_min=0.0,
        period_idx=0,
    ) == pytest.approx(100.0)
    assert evaluate_process_heat_recovery(
        zone.hot_streams,
        zone.cold_streams,
        dt_min=100.0,
        period_idx=0,
    ) == pytest.approx(50.0)

    interior = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=50.0,
        period_idx=0,
    )
    assert interior.status is HeatRecoveryDtMinStatus.SOLVED
    assert interior.dt_min == pytest.approx(100.0, abs=1e-6)
    assert interior.achieved_heat_recovery >= 50.0 - 1e-6

    zero = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=0.0,
        period_idx=0,
    )
    assert zero.status is HeatRecoveryDtMinStatus.ZERO_RECOVERY_BOUNDARY
    assert zero.dt_min == pytest.approx(150.0, abs=1e-6)
    assert zero.achieved_heat_recovery == pytest.approx(0.0, abs=1e-6)


def test_threshold_problem_limit_returns_greatest_feasible_dt_min() -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    result = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=100.0,
        period_idx=0,
    )

    assert result.status is HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT
    assert result.dt_min == pytest.approx(50.0, abs=1e-6)
    assert result.iterations > 0
    assert (
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=result.dt_min + 2e-6,
            period_idx=0,
        )
        < result.thermodynamic_limit
    )


def test_packaged_threshold_problem_returns_positive_limit_dt_min() -> None:
    problem = PinchProblem("pulp_mill.json", project_name="Site")
    forward = problem.target.direct_heat_integration(
        zone="Bleaching",
        period_id="0",
    )
    result = problem.target.heat_recovery_dt_min(
        heat_recovery=float(forward.heat_recovery_target),
        zone="Bleaching",
        period_id="0",
    )

    assert result.status is HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT
    assert result.thermodynamic_limit.value == pytest.approx(
        float(forward.heat_recovery_target)
    )
    assert result.dt_min.value == pytest.approx(
        58.34505012947355,
        abs=1e-6,
    )


def test_packaged_forward_target_round_trips_to_known_global_dt_min() -> None:
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    forward = problem.target.direct_heat_integration()
    zone = problem._master_zone

    result = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=float(forward.heat_recovery_target),
        period_idx=0,
    )

    assert result.status is HeatRecoveryDtMinStatus.SOLVED
    assert result.dt_min == pytest.approx(10.0, abs=2e-6)
    assert result.achieved_heat_recovery >= float(forward.heat_recovery_target)
    assert result.heat_recovery_residual == pytest.approx(
        result.achieved_heat_recovery - float(forward.heat_recovery_target)
    )


def test_empty_or_nonoverlapping_sides_have_zero_limit() -> None:
    empty = solve_heat_recovery_dt_min(
        StreamCollection(),
        StreamCollection(),
        requested_heat_recovery=0.0,
        period_idx=0,
    )
    assert empty.status is HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT
    assert empty.dt_min == 0.0
    assert empty.thermodynamic_limit == 0.0

    problem = PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site/Process",
                    "name": "Hot",
                    "t_supply": 80.0,
                    "t_target": 40.0,
                    "heat_flow": 40.0,
                    "dt_cont": 0.0,
                    "htc": 1.0,
                },
                {
                    "zone": "Site/Process",
                    "name": "Cold",
                    "t_supply": 100.0,
                    "t_target": 140.0,
                    "heat_flow": 40.0,
                    "dt_cont": 0.0,
                    "htc": 1.0,
                },
            ],
            "utilities": [],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Process", "type": "Process Zone"}],
            },
        },
        project_name="Site",
    )
    zone = problem._master_zone
    no_overlap = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=0.0,
        period_idx=0,
    )
    assert no_overlap.thermodynamic_limit == 0.0
    assert no_overlap.dt_min == 0.0


@pytest.mark.parametrize(
    "requested",
    [-1.0, float("nan"), float("inf"), True, np.bool_(True), "50", [50.0]],
)
def test_solver_rejects_invalid_requested_recovery(requested) -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    with pytest.raises((TypeError, ValueError)):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=requested,
            period_idx=0,
        )


@pytest.mark.parametrize(
    ("dt_min", "period_idx"),
    [
        (True, 0),
        (np.bool_(True), 0),
        ("1", 0),
        ([1.0], 0),
        (1.0, True),
        (1.0, np.bool_(True)),
        (1.0, 0.9),
        (1.0, "0"),
    ],
)
def test_evaluator_rejects_implicit_numerical_argument_coercion(
    dt_min,
    period_idx,
) -> None:
    zone = _two_stream_problem()._master_zone

    with pytest.raises((TypeError, ValueError)):
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=dt_min,
            period_idx=period_idx,
        )


def test_precise_evaluator_is_monotone_at_coincident_boundary() -> None:
    zone = _two_stream_problem()._master_zone
    dt_mins = [49.99998, 49.999989, 49.99999, 49.999999, 50.0, 50.000001]
    recoveries = [
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=dt_min,
            period_idx=0,
        )
        for dt_min in dt_mins
    ]

    assert recoveries == sorted(recoveries, reverse=True)


@pytest.mark.parametrize("requested", [1e-12, 5e-7, 1e-6])
def test_positive_micro_recovery_is_solved_and_achieved(requested) -> None:
    zone = _two_stream_problem()._master_zone

    result = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=requested,
        period_idx=0,
    )

    assert result.status is HeatRecoveryDtMinStatus.SOLVED
    assert result.achieved_heat_recovery >= requested
    assert result.dt_min < 150.0


def test_micro_duty_thermodynamic_limit_keeps_threshold_plateau() -> None:
    payload = _two_stream_problem().to_problem_json()
    for stream in payload["streams"]:
        stream["heat_flow"] = 5e-7
    zone = PinchProblem(payload, project_name="Site")._master_zone

    result = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=5e-7,
        period_idx=0,
    )

    assert result.status is HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT
    assert result.dt_min == pytest.approx(50.0, abs=1e-6)
    assert result.achieved_heat_recovery == 5e-7


def test_solver_rejects_recovery_above_thermodynamic_limit() -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    with pytest.raises(ValueError, match="exceeds the thermodynamic limit"):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=100.1,
            period_idx=0,
        )

    with pytest.raises(ValueError, match="exceeds the thermodynamic limit"):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=100.0 + 5e-7,
            period_idx=0,
        )

    with pytest.raises(ValueError, match="exceeds the thermodynamic limit"):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=np.nextafter(100.0, np.inf),
            period_idx=0,
        )


def test_solver_clamps_only_tolerance_sized_recovery_excursions(monkeypatch) -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone
    module = __import__(
        "OpenPinch.analysis.targeting.heat_recovery_dt_min",
        fromlist=["_evaluate_detached_process_heat_recovery"],
    )
    original = module._evaluate_detached_process_heat_recovery

    def within_tolerance(hot, cold, *, dt_min, period_idx):
        recovery = original(
            hot,
            cold,
            dt_min=dt_min,
            period_idx=period_idx,
        )
        return 100.0 + 5e-7 if 0.0 < dt_min < 150.0 else recovery

    monkeypatch.setattr(
        module,
        "_evaluate_detached_process_heat_recovery",
        within_tolerance,
    )
    result = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=50.0,
        period_idx=0,
    )
    assert result.achieved_heat_recovery <= result.thermodynamic_limit

    def outside_tolerance(hot, cold, *, dt_min, period_idx):
        recovery = original(
            hot,
            cold,
            dt_min=dt_min,
            period_idx=period_idx,
        )
        return 100.001 if 0.0 < dt_min < 150.0 else recovery

    monkeypatch.setattr(
        module,
        "_evaluate_detached_process_heat_recovery",
        outside_tolerance,
    )
    with pytest.raises(RuntimeError, match="thermodynamic bounds"):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=99.9995,
            period_idx=0,
        )


def test_solver_rejects_an_unstable_final_boundary(monkeypatch) -> None:
    zone = _two_stream_problem()._master_zone
    module = __import__(
        "OpenPinch.analysis.targeting.heat_recovery_dt_min",
        fromlist=["_evaluate_detached_process_heat_recovery"],
    )
    original = module._evaluate_detached_process_heat_recovery
    evaluations: dict[float, int] = {}

    def unstable(hot, cold, *, dt_min, period_idx):
        recovery = original(
            hot,
            cold,
            dt_min=dt_min,
            period_idx=period_idx,
        )
        evaluations[dt_min] = evaluations.get(dt_min, 0) + 1
        if 0.0 < dt_min < 150.0 and evaluations[dt_min] > 1:
            return 0.0
        return recovery

    monkeypatch.setattr(
        module,
        "_evaluate_detached_process_heat_recovery",
        unstable,
    )

    with pytest.raises(RuntimeError, match="boundary verification"):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=50.0,
            period_idx=0,
        )


def test_zero_solver_rejects_an_unstable_final_boundary(monkeypatch) -> None:
    zone = _two_stream_problem()._master_zone
    module = __import__(
        "OpenPinch.analysis.targeting.heat_recovery_dt_min",
        fromlist=["_evaluate_detached_process_heat_recovery"],
    )
    original = module._evaluate_detached_process_heat_recovery
    evaluations: dict[float, int] = {}

    def unstable(hot, cold, *, dt_min, period_idx):
        recovery = original(
            hot,
            cold,
            dt_min=dt_min,
            period_idx=period_idx,
        )
        evaluations[dt_min] = evaluations.get(dt_min, 0) + 1
        if recovery == 0.0 and evaluations[dt_min] > 1:
            return 1.0
        return recovery

    monkeypatch.setattr(
        module,
        "_evaluate_detached_process_heat_recovery",
        unstable,
    )

    with pytest.raises(RuntimeError, match="boundary verification"):
        solve_heat_recovery_dt_min(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=0.0,
            period_idx=0,
        )


def test_interior_plateau_returns_its_greatest_feasible_dt_min() -> None:
    streams = [
        ("H0", 182.0, 143.0, 121.0),
        ("H1", 266.0, 240.0, 38.0),
        ("C0", 137.0, 169.0, 113.0),
        ("C1", 149.0, 176.0, 149.0),
    ]
    problem = PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site/P",
                    "name": name,
                    "t_supply": supply,
                    "t_target": target,
                    "heat_flow": duty,
                    "dt_cont": 0.0,
                    "htc": 1.0,
                }
                for name, supply, target, duty in streams
            ],
            "utilities": [],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "P", "type": "Process Zone"}],
            },
        },
        project_name="Site",
    )
    zone = problem._master_zone
    result = solve_heat_recovery_dt_min(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=38.0,
        period_idx=0,
    )

    assert result.dt_min == pytest.approx(103.0, abs=2e-6)
    assert (
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=result.dt_min + 2e-6,
            period_idx=0,
        )
        < 38.0
    )

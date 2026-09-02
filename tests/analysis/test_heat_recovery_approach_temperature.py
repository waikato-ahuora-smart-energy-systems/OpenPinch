"""Numerical inverse targeting for a global heat-recovery approach."""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.approach_temperature import (
    evaluate_process_heat_recovery,
    solve_heat_recovery_approach,
)
from OpenPinch.contracts.heat_recovery import HeatRecoveryApproachStatus
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
        approach_temperature=0.0,
        period_idx=0,
    ) == pytest.approx(100.0)
    assert evaluate_process_heat_recovery(
        zone.hot_streams,
        zone.cold_streams,
        approach_temperature=100.0,
        period_idx=0,
    ) == pytest.approx(50.0)

    interior = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=50.0,
        period_idx=0,
    )
    assert interior.status is HeatRecoveryApproachStatus.SOLVED
    assert interior.approach_temperature == pytest.approx(100.0, abs=1e-6)
    assert interior.achieved_heat_recovery >= 50.0 - 1e-6

    zero = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=0.0,
        period_idx=0,
    )
    assert zero.status is HeatRecoveryApproachStatus.ZERO_RECOVERY_BOUNDARY
    # The existing cascade canonicalises temperature intervals within its own
    # 1e-5 thermal epsilon; bisection then locates that numerical zero boundary
    # to the stricter service tolerance.
    assert zero.approach_temperature == pytest.approx(150.0, abs=2e-5)
    assert zero.achieved_heat_recovery == pytest.approx(0.0, abs=1e-6)


def test_thermodynamic_limit_has_exact_zero_approach() -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    result = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=100.0,
        period_idx=0,
    )

    assert result.status is HeatRecoveryApproachStatus.AT_THERMODYNAMIC_LIMIT
    assert result.approach_temperature == 0.0
    assert result.iterations == 0


def test_packaged_forward_target_round_trips_to_known_global_approach() -> None:
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    forward = problem.target.direct_heat_integration()
    zone = problem._master_zone

    result = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=float(forward.heat_recovery_target),
        period_idx=0,
    )

    assert result.status is HeatRecoveryApproachStatus.SOLVED
    assert result.approach_temperature == pytest.approx(10.0, abs=2e-6)
    assert result.achieved_heat_recovery == pytest.approx(
        float(forward.heat_recovery_target),
        abs=1e-6,
    )


def test_empty_or_nonoverlapping_sides_have_zero_limit() -> None:
    empty = solve_heat_recovery_approach(
        StreamCollection(),
        StreamCollection(),
        requested_heat_recovery=0.0,
        period_idx=0,
    )
    assert empty.status is HeatRecoveryApproachStatus.AT_THERMODYNAMIC_LIMIT
    assert empty.approach_temperature == 0.0
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
    no_overlap = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=0.0,
        period_idx=0,
    )
    assert no_overlap.thermodynamic_limit == 0.0
    assert no_overlap.approach_temperature == 0.0


@pytest.mark.parametrize(
    "requested",
    [-1.0, float("nan"), float("inf"), True],
)
def test_solver_rejects_invalid_requested_recovery(requested) -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    with pytest.raises((TypeError, ValueError)):
        solve_heat_recovery_approach(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=requested,
            period_idx=0,
        )


def test_solver_rejects_recovery_above_thermodynamic_limit() -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone

    with pytest.raises(ValueError, match="exceeds the thermodynamic limit"):
        solve_heat_recovery_approach(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=100.1,
            period_idx=0,
        )

    with pytest.raises(ValueError, match="exceeds the thermodynamic limit"):
        solve_heat_recovery_approach(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=100.0 + 5e-7,
            period_idx=0,
        )


def test_solver_clamps_only_tolerance_sized_recovery_excursions(monkeypatch) -> None:
    problem = _two_stream_problem()
    zone = problem._master_zone
    module = __import__(
        "OpenPinch.analysis.targeting.approach_temperature",
        fromlist=["_evaluate_detached_process_heat_recovery"],
    )
    original = module._evaluate_detached_process_heat_recovery

    def within_tolerance(hot, cold, *, approach_temperature, period_idx):
        recovery = original(
            hot,
            cold,
            approach_temperature=approach_temperature,
            period_idx=period_idx,
        )
        return 100.0 + 5e-7 if 0.0 < approach_temperature < 150.0 else recovery

    monkeypatch.setattr(
        module,
        "_evaluate_detached_process_heat_recovery",
        within_tolerance,
    )
    result = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=50.0,
        period_idx=0,
    )
    assert result.achieved_heat_recovery <= result.thermodynamic_limit

    def outside_tolerance(hot, cold, *, approach_temperature, period_idx):
        recovery = original(
            hot,
            cold,
            approach_temperature=approach_temperature,
            period_idx=period_idx,
        )
        return 100.001 if 0.0 < approach_temperature < 150.0 else recovery

    monkeypatch.setattr(
        module,
        "_evaluate_detached_process_heat_recovery",
        outside_tolerance,
    )
    with pytest.raises(RuntimeError, match="thermodynamic bounds"):
        solve_heat_recovery_approach(
            zone.hot_streams,
            zone.cold_streams,
            requested_heat_recovery=99.9995,
            period_idx=0,
        )


def test_interior_plateau_returns_its_greatest_feasible_approach() -> None:
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
    result = solve_heat_recovery_approach(
        zone.hot_streams,
        zone.cold_streams,
        requested_heat_recovery=38.0,
        period_idx=0,
    )

    assert result.approach_temperature == pytest.approx(103.0, abs=2e-6)
    assert (
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            approach_temperature=result.approach_temperature + 2e-5,
            period_idx=0,
        )
        < 38.0
    )

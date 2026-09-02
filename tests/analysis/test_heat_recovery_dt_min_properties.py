"""Property-based invariants for global dt_min inversion."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.heat_recovery_dt_min import (
    evaluate_process_heat_recovery,
)
from OpenPinch.contracts.heat_recovery_dt_min import HeatRecoveryDtMinResult
from tests.strategies.heat_recovery import (
    heat_recovery_problem_payloads,
    threshold_problem_payloads,
)


@seed(20260902)
@settings(max_examples=25, deadline=None)
@given(
    payload=heat_recovery_problem_payloads(),
    fractions=st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
)
def test_recovery_is_non_increasing_and_stream_order_invariant(
    payload,
    fractions,
) -> None:
    problem = PinchProblem(payload, project_name="Site")
    zone = problem._master_zone
    bound = max(
        0.0,
        max(stream.maximum_temperature.value for stream in zone.hot_streams)
        - min(stream.minimum_temperature.value for stream in zone.cold_streams),
    )
    dt_mins = sorted(fraction * bound for fraction in fractions)
    recoveries = [
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=dt_min,
            period_idx=0,
        )
        for dt_min in dt_mins
    ]
    assert recoveries[1] <= recoveries[0] + 1e-6

    reversed_payload = deepcopy(payload)
    reversed_payload["streams"].reverse()
    reversed_zone = PinchProblem(
        reversed_payload,
        project_name="Site",
    )._master_zone
    reversed_recovery = evaluate_process_heat_recovery(
        reversed_zone.hot_streams,
        reversed_zone.cold_streams,
        dt_min=dt_mins[0],
        period_idx=0,
    )
    assert reversed_recovery == pytest.approx(recoveries[0], abs=1e-6)


@seed(20260902)
@settings(max_examples=20, deadline=None)
@given(
    payload=heat_recovery_problem_payloads(),
    fraction=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
)
def test_inverse_is_bounded_idempotent_unit_invariant_and_json_safe(
    payload,
    fraction,
) -> None:
    problem = PinchProblem(payload, project_name="Site")
    zone = problem._master_zone
    limit = evaluate_process_heat_recovery(
        zone.hot_streams,
        zone.cold_streams,
        dt_min=0.0,
        period_idx=0,
    )
    requested = limit * fraction
    result = problem.target.heat_recovery_dt_min(
        heat_recovery=requested,
        zone="Site/Process",
    )
    repeated = problem.target.heat_recovery_dt_min(
        heat_recovery={"value": requested / 1_000.0, "unit": "MW"},
        zone="Site/Process",
    )

    bound = max(
        0.0,
        max(stream.maximum_temperature.value for stream in zone.hot_streams)
        - min(stream.minimum_temperature.value for stream in zone.cold_streams),
    )
    assert 0.0 <= result.dt_min.value <= bound + 1e-6
    assert result.achieved_heat_recovery.value >= requested - max(
        1e-6,
        requested * 1e-9,
    )
    assert repeated == result
    assert (
        HeatRecoveryDtMinResult.model_validate_json(result.model_dump_json()) == result
    )

    larger = min(bound, result.dt_min.value + 2e-5)
    if larger > result.dt_min.value + 1e-6:
        larger_recovery = evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=larger,
            period_idx=0,
        )
        assert larger_recovery < requested + max(1e-6, requested * 1e-9)


@seed(20260902)
@settings(max_examples=20, deadline=None)
@given(
    dt_min=st.floats(
        min_value=1.0,
        max_value=220.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_generated_uniform_forward_targets_round_trip_through_inverse(
    dt_min,
) -> None:
    source_problem = PinchProblem("basic_pinch.json", project_name="Site")
    payload = source_problem.to_problem_json()
    for stream in payload["streams"]:
        stream["dt_cont"] = {
            "value": dt_min / 2.0,
            "unit": "delta_degC",
        }
    forward_problem = PinchProblem(payload, project_name="Site")
    forward = forward_problem.target.direct_heat_integration()

    inverse = source_problem.target.heat_recovery_dt_min(
        heat_recovery=float(forward.heat_recovery_target),
    )
    recovered = evaluate_process_heat_recovery(
        source_problem._master_zone.hot_streams,
        source_problem._master_zone.cold_streams,
        dt_min=inverse.dt_min.value,
        period_idx=0,
    )

    assert recovered == pytest.approx(float(forward.heat_recovery_target), abs=2e-5)
    assert inverse.dt_min.value == pytest.approx(dt_min, abs=2e-5)


@seed(20260903)
@settings(max_examples=20, deadline=None)
@given(payload=threshold_problem_payloads())
def test_threshold_limit_returns_maximal_order_invariant_json_safe_dt_min(
    payload,
) -> None:
    problem = PinchProblem(payload, project_name="Site")
    zone = problem._master_zone
    limit = evaluate_process_heat_recovery(
        zone.hot_streams,
        zone.cold_streams,
        dt_min=0.0,
        period_idx=0,
    )
    expected_dt_min = (
        zone.hot_streams[0].minimum_temperature.value
        - zone.cold_streams[0].minimum_temperature.value
    )
    result = problem.target.heat_recovery_dt_min(
        heat_recovery=limit,
        zone="Site/Process",
    )
    repeated = problem.target.heat_recovery_dt_min(
        heat_recovery=limit,
        zone="Site/Process",
    )
    equivalent_unit_result = problem.target.heat_recovery_dt_min(
        heat_recovery={"value": limit / 1_000.0, "unit": "MW"},
        zone="Site/Process",
    )

    reversed_payload = deepcopy(payload)
    reversed_payload["streams"].reverse()
    reversed_result = PinchProblem(
        reversed_payload,
        project_name="Site",
    ).target.heat_recovery_dt_min(
        heat_recovery=limit,
        zone="Site/Process",
    )

    assert result.status.value == "at_thermodynamic_limit"
    assert result.dt_min.value == pytest.approx(
        expected_dt_min,
        abs=2e-5,
    )
    assert repeated == result
    assert reversed_result == result
    assert equivalent_unit_result.status == result.status
    assert equivalent_unit_result.dt_min.value == pytest.approx(
        result.dt_min.value,
        abs=1e-6,
    )
    assert equivalent_unit_result.thermodynamic_limit.value == pytest.approx(limit)
    assert (
        HeatRecoveryDtMinResult.model_validate_json(result.model_dump_json()) == result
    )
    assert (
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            dt_min=result.dt_min.value + 2e-5,
            period_idx=0,
        )
        < limit
    )

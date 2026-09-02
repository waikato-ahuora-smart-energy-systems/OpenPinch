"""Property-based invariants for global HRAT inversion."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.approach_temperature import (
    evaluate_process_heat_recovery,
)
from OpenPinch.contracts.heat_recovery import HeatRecoveryApproachResult
from tests.strategies.heat_recovery import heat_recovery_problem_payloads


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
    approaches = sorted(fraction * bound for fraction in fractions)
    recoveries = [
        evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            approach_temperature=approach,
            period_idx=0,
        )
        for approach in approaches
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
        approach_temperature=approaches[0],
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
        approach_temperature=0.0,
        period_idx=0,
    )
    requested = limit * fraction
    result = problem.target.heat_recovery_approach_temperature(
        heat_recovery=requested,
        zone="Site/Process",
    )
    repeated = problem.target.heat_recovery_approach_temperature(
        heat_recovery={"value": requested / 1_000.0, "unit": "MW"},
        zone="Site/Process",
    )

    bound = max(
        0.0,
        max(stream.maximum_temperature.value for stream in zone.hot_streams)
        - min(stream.minimum_temperature.value for stream in zone.cold_streams),
    )
    assert 0.0 <= result.approach_temperature.value <= bound + 1e-6
    assert result.achieved_heat_recovery.value >= requested - max(
        1e-6,
        requested * 1e-9,
    )
    assert repeated == result
    assert (
        HeatRecoveryApproachResult.model_validate_json(result.model_dump_json())
        == result
    )

    larger = min(bound, result.approach_temperature.value + 2e-5)
    if larger > result.approach_temperature.value + 1e-6:
        larger_recovery = evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            approach_temperature=larger,
            period_idx=0,
        )
        assert larger_recovery < requested + max(1e-6, requested * 1e-9)


@seed(20260902)
@settings(max_examples=20, deadline=None)
@given(
    approach=st.floats(
        min_value=1.0,
        max_value=220.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_generated_uniform_forward_targets_round_trip_through_inverse(
    approach,
) -> None:
    source_problem = PinchProblem("basic_pinch.json", project_name="Site")
    payload = source_problem.to_problem_json()
    for stream in payload["streams"]:
        stream["dt_cont"] = {
            "value": approach / 2.0,
            "unit": "delta_degC",
        }
    forward_problem = PinchProblem(payload, project_name="Site")
    forward = forward_problem.target.direct_heat_integration()

    inverse = source_problem.target.heat_recovery_approach_temperature(
        heat_recovery=float(forward.heat_recovery_target),
    )
    recovered = evaluate_process_heat_recovery(
        source_problem._master_zone.hot_streams,
        source_problem._master_zone.cold_streams,
        approach_temperature=inverse.approach_temperature.value,
        period_idx=0,
    )

    assert recovered == pytest.approx(float(forward.heat_recovery_target), abs=2e-5)
    assert inverse.approach_temperature.value == pytest.approx(approach, abs=2e-5)

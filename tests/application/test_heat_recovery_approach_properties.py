"""Property checks for multiperiod inverse-target orchestration."""

from __future__ import annotations

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.approach_temperature import (
    evaluate_process_heat_recovery,
)
from tests.strategies.heat_recovery import (
    multiperiod_heat_recovery_problem_payloads,
)


@seed(20260902)
@settings(max_examples=12, deadline=None)
@given(
    payload=multiperiod_heat_recovery_problem_payloads(),
    fraction=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
)
def test_generated_all_period_results_have_parallel_equivalence_and_no_state(
    payload,
    fraction,
) -> None:
    problem = PinchProblem(payload, project_name="Site")
    before = problem.to_problem_json()
    zone = problem._master_zone
    requests = {
        period_id: evaluate_process_heat_recovery(
            zone.hot_streams,
            zone.cold_streams,
            approach_temperature=0.0,
            period_idx=period_idx,
        )
        * fraction
        for period_id, period_idx in problem.period_ids.items()
    }

    sequential = problem.target.all_periods.heat_recovery_approach_temperature(
        heat_recovery=requests,
        zone="Site/Process",
        workers=1,
    )
    parallel = problem.target.all_periods.heat_recovery_approach_temperature(
        heat_recovery=requests,
        zone="Site/Process",
        workers=2,
    )

    assert list(sequential) == list(problem.period_ids)
    assert sequential == parallel
    assert problem.to_problem_json() == before
    assert problem.period_results == {}
    for period_id, result in sequential.items():
        assert result.period_id == period_id
        assert result.achieved_heat_recovery.value >= requests[period_id] - 1e-6

"""Property checks for multiperiod inverse-target orchestration."""

from __future__ import annotations

from copy import deepcopy

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem
from OpenPinch.analysis.targeting.heat_recovery_dt_min import (
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
            dt_min=0.0,
            period_idx=period_idx,
        )
        * fraction
        for period_id, period_idx in problem.period_ids.items()
    }

    sequential = problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery=requests,
        zone="Site/Process",
        workers=1,
    )
    parallel = problem.target.all_periods.heat_recovery_dt_min(
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


@seed(20260903)
@settings(max_examples=12, deadline=None)
@given(
    payload=multiperiod_heat_recovery_problem_payloads(),
    fraction=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
)
def test_generated_foreign_zone_objects_resolve_against_the_local_problem(
    payload,
    fraction,
) -> None:
    local = PinchProblem(payload, project_name="Site")
    foreign_payload = deepcopy(payload)
    for stream in foreign_payload["streams"]:
        heat_flow = stream["heat_flow"]
        heat_flow["values"] = [value * 1.5 for value in heat_flow["values"]]
    foreign = PinchProblem(foreign_payload, project_name="Site")
    foreign_zone = foreign._master_zone.get_subzone("Site/Process")
    local_zone = local._master_zone.get_subzone("Site/Process")
    limit = evaluate_process_heat_recovery(
        local_zone.hot_streams,
        local_zone.cold_streams,
        dt_min=0.0,
        period_idx=0,
    )
    request = limit * fraction

    by_object = local.target.heat_recovery_dt_min(
        heat_recovery=request,
        zone=foreign_zone,
        period_id=next(iter(local.period_ids)),
    )
    by_address = local.target.heat_recovery_dt_min(
        heat_recovery=request,
        zone="Site/Process",
        period_id=next(iter(local.period_ids)),
    )

    assert by_object == by_address

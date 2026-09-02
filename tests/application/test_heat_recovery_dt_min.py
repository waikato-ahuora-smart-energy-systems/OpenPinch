"""Application orchestration for inverse heat-recovery targeting."""

from __future__ import annotations

import pickle
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.contracts.heat_recovery_dt_min import (
    HeatRecoveryDtMinResult,
    HeatRecoveryDtMinStatus,
)
from OpenPinch.domain.value import Q_, Value


def _multiperiod_payload() -> dict:
    return {
        "streams": [
            {
                "zone": "Site/Process",
                "name": "Hot",
                "t_supply": {"values": [200.0, 220.0], "unit": "degC"},
                "t_target": {"values": [100.0, 120.0], "unit": "degC"},
                "heat_flow": {"values": [100.0, 200.0], "unit": "kW"},
                "dt_cont": 13.0,
                "htc": 1.0,
            },
            {
                "zone": "Site/Process",
                "name": "Cold",
                "t_supply": {"values": [50.0, 60.0], "unit": "degC"},
                "t_target": {"values": [150.0, 170.0], "unit": "degC"},
                "heat_flow": {"values": [100.0, 200.0], "unit": "kW"},
                "dt_cont": 7.0,
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [{"name": "Process", "type": "Process Zone"}],
        },
        "options": {"PROBLEM_PERIOD_IDS": ["0", "peak"]},
    }


def _no_overlap_payload() -> dict:
    payload = _multiperiod_payload()
    payload["streams"][0].update(t_supply=80.0, t_target=40.0, heat_flow=40.0)
    payload["streams"][1].update(t_supply=100.0, t_target=140.0, heat_flow=40.0)
    payload["options"] = {}
    return payload


def _stream_contributions(
    problem: PinchProblem,
) -> list[tuple[str, object, float, bool]]:
    return [
        (
            stream.name,
            stream.delta_t_contribution.to_dict(),
            stream.delta_t_contribution_multiplier,
            stream.delta_t_contribution_multiplier_locked,
        )
        for stream in problem._master_zone.process_streams
    ]


def test_selected_period_service_returns_units_scope_and_known_dt_min() -> None:
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    result = problem.target.heat_recovery_dt_min(
        heat_recovery={"value": 5.150000005, "unit": "MW"},
        zone="Site/Plant",
        period_id="0",
    )

    assert isinstance(result, HeatRecoveryDtMinResult)
    assert result.scope == "Site/Plant"
    assert result.period_id == "0"
    assert result.status is HeatRecoveryDtMinStatus.SOLVED
    assert result.dt_min.value == pytest.approx(10.0, abs=2e-6)
    assert result.dt_min.unit == "delta_degC"
    assert result.requested_heat_recovery.value == pytest.approx(5_150.000005)
    assert result.requested_heat_recovery.unit == "kW"


def test_output_unit_overrides_apply_to_every_thermal_quantity() -> None:
    payload = _multiperiod_payload()
    payload["options"].update(
        INPUT_UNIT_HEAT_FLOW="MW",
        OUTPUT_UNIT_HEAT_FLOW="MW",
        OUTPUT_UNIT_TEMPERATURE="K",
    )
    result = PinchProblem(payload, project_name="Site").target.heat_recovery_dt_min(
        heat_recovery=0.05,
        zone="Site/Process",
        period_id="0",
    )

    assert result.dt_min.unit == "K"
    assert result.requested_heat_recovery.value == pytest.approx(0.05)
    assert result.requested_heat_recovery.unit == "MW"
    assert result.achieved_heat_recovery.unit == "MW"
    assert result.thermodynamic_limit.unit == "MW"
    assert result.heat_recovery_residual.unit == "MW"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_recovery_values_are_rejected(value) -> None:
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    with pytest.raises((TypeError, ValueError)):
        problem.target.heat_recovery_dt_min(heat_recovery=value)


@pytest.mark.parametrize(
    "value",
    [
        "50",
        b"50",
        [50.0],
        (50.0,),
        np.array(50.0),
        np.array([50.0]),
        {"0": 50.0},
        {"value": 50.0},
        {"value": True, "unit": "kW"},
        {"value": np.bool_(True), "unit": "kW"},
    ],
)
def test_selected_period_rejects_implicitly_coercible_non_scalars(value) -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")

    with pytest.raises((TypeError, ValueError), match="finite non-negative scalar"):
        problem.target.heat_recovery_dt_min(
            heat_recovery=value,
            zone="Site/Process",
            period_id="0",
        )


@pytest.mark.parametrize(
    "value",
    [
        50,
        50.0,
        np.int64(50),
        np.float64(50.0),
        Decimal("50"),
        Fraction(100, 2),
        Value(50.0, unit="kW"),
        Q_(0.05, "MW"),
        {"value": 0.05, "unit": "MW"},
    ],
)
def test_selected_period_accepts_only_supported_scalar_forms(value) -> None:
    result = PinchProblem(
        _multiperiod_payload(), project_name="Site"
    ).target.heat_recovery_dt_min(
        heat_recovery=value,
        zone="Site/Process",
        period_id="0",
    )

    assert result.requested_heat_recovery.value == pytest.approx(50.0)


def test_above_limit_error_reports_complete_context() -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")

    with pytest.raises(ValueError) as error:
        problem.target.heat_recovery_dt_min(
            heat_recovery=101.0,
            zone="Site/Process",
            period_id="0",
        )

    message = str(error.value)
    assert "101" in message
    assert "100" in message
    assert "Site/Process" in message
    assert "period '0'" in message
    assert "kW" in message


def test_community_and_region_scopes_are_rejected_with_guidance() -> None:
    payload = _multiperiod_payload()
    payload["zone_tree"] = {
        "name": "Region",
        "type": "Region",
        "children": [
            {
                "name": "Community",
                "type": "Community",
                "children": [
                    {
                        "name": "Site",
                        "type": "Site",
                        "children": [{"name": "Process", "type": "Process Zone"}],
                    }
                ],
            }
        ],
    }
    for stream in payload["streams"]:
        stream["zone"] = "Region/Community/Site/Process"
    problem = PinchProblem(payload, project_name="Region")

    with pytest.raises(ValueError, match="Site, Process Zone, or Unit Operation"):
        problem.target.heat_recovery_dt_min(
            heat_recovery=50.0,
            zone="Region/Community",
            period_id="0",
        )


def test_all_period_scalar_broadcast_mapping_order_and_parallel_parity() -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")

    sequential = problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone="Site/Process",
    )
    parallel = problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery={"0": 50.0, "peak": 50.0},
        zone="Site/Process",
        workers=2,
    )

    assert list(sequential) == ["0", "peak"]
    assert list(parallel) == ["0", "peak"]
    assert [result.period_id for result in sequential.values()] == ["0", "peak"]
    assert sequential == parallel

    explicit_scalar = problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery={"value": 0.05, "unit": "MW"},
        zone="Site/Process",
    )
    assert list(explicit_scalar) == ["0", "peak"]
    assert all(
        result.requested_heat_recovery.value == pytest.approx(50.0)
        for result in explicit_scalar.values()
    )


def test_exact_period_ids_take_precedence_over_scalar_mapping_keys() -> None:
    payload = _multiperiod_payload()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["value", "unit"]
    problem = PinchProblem(payload, project_name="Site")

    results = problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery={"value": 40.0, "unit": 60.0},
        zone="Site/Process",
    )

    assert list(results) == ["value", "unit"]
    assert results["value"].requested_heat_recovery.value == pytest.approx(40.0)
    assert results["unit"].requested_heat_recovery.value == pytest.approx(60.0)


@pytest.mark.parametrize(
    "mapping",
    [{}, {"0": 50.0}, {"0": 50.0, "peak": 50.0, "x": 1.0}],
)
def test_all_period_mapping_requires_exact_canonical_ids(mapping) -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")
    with pytest.raises(ValueError, match="exactly the canonical period IDs"):
        problem.target.all_periods.heat_recovery_dt_min(heat_recovery=mapping)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_all_period_workers_must_be_a_positive_integer(workers) -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        problem.target.all_periods.heat_recovery_dt_min(
            heat_recovery=50.0,
            workers=workers,
        )


def test_selected_and_all_period_calls_are_non_mutating() -> None:
    problem = PinchProblem(_multiperiod_payload(), project_name="Site")
    problem.target.direct_heat_integration(period_id="peak")
    problem.target.all_periods.direct_heat_integration()
    before = {
        "input": problem.to_problem_json(),
        "results": problem.results.model_dump(mode="json"),
        "period_results": {
            key: value.model_dump(mode="json")
            for key, value in problem.period_results.items()
        },
        "run_spec": deepcopy(problem._last_target_run_spec),
        "contributions": _stream_contributions(problem),
        "targets": pickle.dumps(problem._master_zone.targets),
        "target_ids": {
            key: id(value) for key, value in problem._master_zone.targets.items()
        },
    }

    problem.target.heat_recovery_dt_min(
        heat_recovery=50.0,
        period_id="0",
    )
    problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery=50.0,
        workers=2,
    )

    assert problem.to_problem_json() == before["input"]
    assert problem.results.model_dump(mode="json") == before["results"]
    assert {
        key: value.model_dump(mode="json")
        for key, value in problem.period_results.items()
    } == before["period_results"]
    assert problem._last_target_run_spec == before["run_spec"]
    assert _stream_contributions(problem) == before["contributions"]
    assert pickle.dumps(problem._master_zone.targets) == before["targets"]
    assert {
        key: id(value) for key, value in problem._master_zone.targets.items()
    } == before["target_ids"]


def test_workspace_active_case_and_batch_failure_isolation() -> None:
    workspace = PinchWorkspace(_multiperiod_payload(), project_name="Site")
    workspace.load(
        _no_overlap_payload(),
        case_name="no-overlap",
        activate=False,
    )

    active = workspace.target.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone="Site/Process",
        period_id="0",
    )
    batch = workspace.cases().target.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone="Site/Process",
        period_id="0",
    )
    batch_periods = workspace.cases().target.all_periods.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone="Site/Process",
    )

    assert active.period_id == "0"
    assert list(batch.results) == [workspace.baseline_name]
    assert list(batch.errors) == ["no-overlap"]
    assert list(batch_periods.results) == [workspace.baseline_name]
    assert list(batch_periods.errors) == ["no-overlap"]


def test_zone_objects_are_resolved_locally_by_address() -> None:
    local = PinchProblem(_multiperiod_payload(), project_name="Site")
    foreign_payload = _multiperiod_payload()
    for stream in foreign_payload["streams"]:
        stream["heat_flow"] = {"values": [250.0, 300.0], "unit": "kW"}
    foreign = PinchProblem(foreign_payload, project_name="Site")
    foreign_zone = foreign._master_zone.get_subzone("Site/Process")

    by_object = local.target.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone=foreign_zone,
        period_id="0",
    )
    by_address = local.target.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone="Site/Process",
        period_id="0",
    )

    assert by_object == by_address


def test_foreign_zone_with_missing_local_address_is_rejected() -> None:
    local = PinchProblem(_multiperiod_payload(), project_name="Site")
    foreign = PinchProblem("basic_pinch.json", project_name="Site")
    foreign_zone = foreign._master_zone.get_subzone("Site/Plant")

    with pytest.raises((KeyError, ValueError), match="Site/Plant"):
        local.target.heat_recovery_dt_min(
            heat_recovery=50.0,
            zone=foreign_zone,
            period_id="0",
        )


def test_batch_resolves_zone_object_independently_for_each_case() -> None:
    workspace = PinchWorkspace(_multiperiod_payload(), project_name="Site")
    workspace.load(_no_overlap_payload(), case_name="no-overlap", activate=False)
    zone = workspace.case()._master_zone.get_subzone("Site/Process")

    batch = workspace.cases().target.heat_recovery_dt_min(
        heat_recovery=50.0,
        zone=zone,
        period_id="0",
    )

    assert list(batch.results) == [workspace.baseline_name]
    assert list(batch.errors) == ["no-overlap"]

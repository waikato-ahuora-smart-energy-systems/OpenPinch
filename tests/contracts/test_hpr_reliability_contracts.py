"""Reliability contracts for detached HPR targeting results."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.hpr import (
    HPRFailureCategory,
    HPRFailureDiagnostic,
    HPRFailureSummary,
    HPRSearchBudget,
    HPRSimulationLoopRecord,
    HPRSimulationStageRecord,
    HPRTargetingError,
    HprTargetSimulationRecord,
    HPRTopologyIdentifier,
)


def _stage(*, ordinal: int = 0) -> HPRSimulationStageRecord:
    return HPRSimulationStageRecord(
        stage_id=f"vc-{ordinal}",
        ordinal=ordinal,
        role="vapour_compression",
        fluid_spec="Water",
        evaporating_or_suction_temperature=50.0,
        condensing_or_discharge_temperature=70.0,
        compressor_isentropic_efficiency=0.7,
        useful_duty=100.0,
        compressor_work=20.0,
        assumptions={"power_boundary": "compressor_only"},
    )


def _general_record() -> HprTargetSimulationRecord:
    return HprTargetSimulationRecord(
        simulation_backend="coolprop",
        mode="heat_pump",
        cycle_id="vapour_compression_mvr",
        topology_id=HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
        schema_version="1.1",
        model_id="openpinch-vc-mvr-v1",
        refrigerant_spec="Water",
        nominal_evaporating_temperature=50.0,
        nominal_condensing_temperature=90.0,
        nominal_useful_duty=100.0,
        source_approach_temperature=3.0,
        sink_approach_temperature=3.0,
        compressor_isentropic_efficiency=0.7,
        superheat=0.0,
        subcooling=0.0,
        internal_hx_gas_temperature_change=0.0,
        evaporator_count=1,
        condenser_count=1,
        engine_version="8.0.0",
        assumptions={"power_boundary": "compressor_only"},
        loops=(
            HPRSimulationLoopRecord(
                loop_id="vc-loop",
                ordinal=0,
                loop_role="vapour_compression",
                fluid_spec="Water",
                nominal_duty=100.0,
                nominal_work=20.0,
                stages=(_stage(),),
            ),
        ),
    )


@pytest.mark.parametrize(
    "values",
    [
        {"maximum_iterations": True, "maximum_evaluations": 10},
        {"maximum_iterations": 1.0, "maximum_evaluations": 10},
        {"maximum_iterations": 0, "maximum_evaluations": 10},
        {"maximum_iterations": 1, "maximum_evaluations": -1},
    ],
)
def test_search_budget_requires_exact_positive_integers(values) -> None:
    with pytest.raises(ValidationError):
        HPRSearchBudget(**values)


def test_search_budget_defaults_match_reusable_optimizer() -> None:
    budget = HPRSearchBudget()

    assert budget.maximum_iterations == 300
    assert budget.maximum_evaluations == 1_000_000
    assert HPRSearchBudget.model_validate_json(budget.model_dump_json()) == budget


def test_failure_contract_is_bounded_detached_and_value_error_compatible() -> None:
    diagnostic = HPRFailureDiagnostic(
        category=HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY,
        reason_code="coolprop.flash_infeasible",
        summary="candidate state is outside the supported envelope",
        fluid="Water",
        candidate_index=2,
    )
    summary = HPRFailureSummary(
        simulation_backend="coolprop",
        cycle="cascade_vapour_compression",
        evaluated_count=3,
        category_counts={
            HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY: 3,
        },
        representative_failures=(diagnostic,),
        budget=HPRSearchBudget(maximum_iterations=5, maximum_evaluations=50),
        warm_start_evaluated=True,
        warm_start_viable=False,
    )

    error = HPRTargetingError("no viable candidate", diagnostics=summary)

    assert isinstance(error, ValueError)
    assert error.diagnostics == summary
    assert HPRFailureSummary.model_validate_json(summary.model_dump_json()) == summary


def test_failure_summary_rejects_incoherent_warm_start_flags() -> None:
    with pytest.raises(ValidationError, match="warm_start"):
        HPRFailureSummary(
            simulation_backend="coolprop",
            cycle="cascade_vapour_compression",
            evaluated_count=0,
            category_counts={},
            representative_failures=(),
            budget=HPRSearchBudget(),
            warm_start_evaluated=False,
            warm_start_viable=True,
        )


def test_generalized_simulation_record_round_trips_ordered_loops() -> None:
    record = _general_record()

    assert record.cycle_id == "vapour_compression_mvr"
    assert record.topology_id is HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR
    assert record.loops[0].stages[0].stage_id == "vc-0"
    assert (
        HprTargetSimulationRecord.model_validate_json(record.model_dump_json())
        == record
    )


def test_generalized_simulation_record_rejects_noncontiguous_stage_order() -> None:
    with pytest.raises(ValidationError, match="ordinal"):
        HPRSimulationLoopRecord(
            loop_id="vc-loop",
            ordinal=0,
            loop_role="vapour_compression",
            fluid_spec="Water",
            nominal_duty=100.0,
            nominal_work=20.0,
            stages=(_stage(ordinal=1),),
        )


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_generalized_stage_rejects_nonfinite_state(value: float) -> None:
    with pytest.raises(ValidationError):
        HPRSimulationStageRecord.model_validate(
            _stage().model_dump() | {"evaporating_or_suction_temperature": value}
        )

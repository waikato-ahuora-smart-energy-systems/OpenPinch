"""Transform runtime target state into external reporting contracts."""

from __future__ import annotations

from collections.abc import Mapping

from ...contracts.reporting import HeatUtility, PinchTemp, TargetResults
from ...contracts.units import coerce_output_value
from ...domain.enums import SummaryRowType
from ...domain.stream_collection import StreamCollection
from ...domain.targets import (
    BaseTargetModel,
    DirectIntegrationTarget,
    DirectOrcTarget,
    HeatPumpTargetBase,
    IndirectIntegrationTarget,
    UtilitySummaryTarget,
)
from ...domain.value import Value


def direct_result(target, *, is_total=False):
    result = _utility_summary_result(target, is_total=is_total)
    return result.model_copy(
        update={
            "work_target": _metric(target, "work_target"),
            "turbine_efficiency_target": _metric(
                target,
                "turbine_efficiency_target",
            ),
            "area": _metric(target, "area"),
            "num_units": target.num_units,
            "capital_cost": _metric(target, "capital_cost"),
            "total_cost": _metric(target, "total_cost"),
        }
    )


def indirect_result(target, *, is_total=False):
    result = _utility_summary_result(target, is_total=is_total)
    return result.model_copy(
        update={
            "work_target": _metric(target, "work_target"),
            "turbine_efficiency_target": _metric(
                target,
                "turbine_efficiency_target",
            ),
        }
    )


def hpr_result(target, *, is_total=False):
    result = _utility_summary_result(target, is_total=is_total)
    return result.model_copy(
        update={
            "work_target": _metric(target, "work_target"),
            "turbine_efficiency_target": _metric(
                target,
                "turbine_efficiency_target",
            ),
            "hpr_cycle": target.hpr_cycle,
            "hpr_simulation_backend": target.hpr_simulation_backend,
            "hpr_target_simulation_record": _target_simulation_record(target),
            "hpr_utility_total": _metric(target, "hpr_utility_total"),
            "hpr_work": _metric(target, "hpr_work"),
            "hpr_external_utility": _metric(target, "hpr_external_utility"),
            "hpr_ambient_hot": _metric(target, "hpr_ambient_hot"),
            "hpr_ambient_cold": _metric(target, "hpr_ambient_cold"),
            **_hpr_summary_values(target),
            "hpr_cop": _metric(target, "hpr_cop"),
            "hpr_eta_he": _metric(target, "hpr_eta_he"),
            "hpr_operating_cost": _metric(target, "hpr_operating_cost"),
            "hpr_capital_cost": _metric(target, "hpr_capital_cost"),
            "hpr_annualized_capital_cost": _metric(
                target,
                "hpr_annualized_capital_cost",
            ),
            "hpr_hot_utility_annualized_capital_cost": _metric(
                target,
                "hpr_hot_utility_annualized_capital_cost",
            ),
            "hpr_refrigeration_annualized_capital_cost": _metric(
                target,
                "hpr_refrigeration_annualized_capital_cost",
            ),
            "hpr_utility_annualized_capital_cost": _metric(
                target,
                "hpr_utility_annualized_capital_cost",
            ),
            "hpr_total_annualized_cost": _metric(
                target,
                "hpr_total_annualized_cost",
            ),
            "hpr_machine_capital_costs": getattr(
                target, "hpr_machine_capital_costs", None
            ),
            "hpr_machine_annualized_capital_costs": getattr(
                target, "hpr_machine_annualized_capital_costs", None
            ),
            "hpr_success": target.hpr_success,
            "hpr_shared_design": getattr(target, "hpr_shared_design", None),
            "hpr_hot_streams": target.hpr_hot_streams,
            "hpr_cold_streams": target.hpr_cold_streams,
        }
    )


def _hpr_summary_values(target: HeatPumpTargetBase) -> dict:
    load = target.hpr_load
    details = target.hpr_details
    objective = _detail(details, "obj")
    return {
        "hpr_selected_load": None
        if load is None
        else coerce_output_value(
            {"value": load.selected, "unit": load.heat_flow_unit},
            metric_name="hpr_selected_load",
            config=target.config,
        ),
        "hpr_achieved_load": None
        if load is None
        else coerce_output_value(
            {"value": load.achieved, "unit": load.heat_flow_unit},
            metric_name="hpr_achieved_load",
            config=target.config,
        ),
        "hpr_objective": None if objective is None else float(objective),
        **_hpr_shared_design_values(details),
    }


def _hpr_shared_design_values(details) -> dict:
    vector = _detail(details, "design_vector")
    period_ids = _detail(details, "period_ids")
    weights = _detail(details, "period_weights")
    period_outputs = _detail(details, "period_outputs")
    weighted = _detail(details, "weighted_output")
    weighted_objective = _detail(weighted, "obj") if weighted is not None else None
    return {
        "hpr_design_vector": None
        if vector is None
        else tuple(float(value) for value in vector),
        "hpr_period_ids": None if period_ids is None else tuple(period_ids),
        "hpr_period_weights": None
        if weights is None
        else tuple(float(weight) for weight in weights),
        "hpr_period_success": None
        if not isinstance(period_outputs, Mapping)
        else {
            str(period_id): bool(_detail(output, "success"))
            for period_id, output in period_outputs.items()
        },
        "hpr_weighted_objective": None
        if weighted_objective is None
        else float(weighted_objective),
    }


def _detail(details, field: str):
    if isinstance(details, Mapping):
        return details.get(field)
    return getattr(details, field, None)


def _target_simulation_record(target: HeatPumpTargetBase):
    return _detail(target.hpr_details, "target_simulation_record")


def _utility_summary_result(
    target: UtilitySummaryTarget,
    *,
    is_total: bool,
) -> TargetResults:
    degree_of_integration = None
    if target.degree_of_int is not None:
        degree_of_integration = _metric(
            target,
            "degree_of_int",
            metric_name="degree_of_integration",
        )
    return TargetResults(
        scope=target.scope,
        provenance=target.provenance,
        zone_type=target.zone_type,
        integration_type=target.integration_type,
        target_method=target.target_method,
        period_idx=target.period_idx,
        period_id=target.period_id,
        degree_of_integration=degree_of_integration,
        Qh=_metric(target, "hot_utility_target", metric_name="Qh"),
        Qc=_metric(target, "cold_utility_target", metric_name="Qc"),
        Qr=_metric(target, "heat_recovery_target", metric_name="Qr"),
        utility_cost=_metric(target, "utility_cost"),
        process_component_work_target=_metric(
            target,
            "process_component_work_target",
            metric_name="work_target",
        ),
        row_type=(
            SummaryRowType.FOOTER.value if is_total else SummaryRowType.CONTENT.value
        ),
        hot_utilities=_heat_utilities(target.hot_utilities, target=target),
        cold_utilities=_heat_utilities(target.cold_utilities, target=target),
        pinch_temp=PinchTemp(
            cold_temp=_metric(target, "cold_pinch", metric_name="cold_temp"),
            hot_temp=_metric(target, "hot_pinch", metric_name="hot_temp"),
        ),
        exergy_sources=_metric(target, "exergy_sources"),
        exergy_sinks=_metric(target, "exergy_sinks"),
        ETE=_metric(target, "ETE"),
        exergy_req_min=_metric(target, "exergy_req_min"),
        exergy_des_min=_metric(target, "exergy_des_min"),
    )


def _metric(
    target: BaseTargetModel,
    field: str,
    *,
    metric_name: str | None = None,
):
    return coerce_output_value(
        getattr(target, field),
        metric_name=metric_name or field,
        config=target.config,
    )


def _heat_utilities(
    utilities: StreamCollection,
    *,
    target: BaseTargetModel,
) -> list[HeatUtility]:
    return [
        HeatUtility(
            name=utility.name,
            heat_flow=_period_duty(
                coerce_output_value(
                    utility.heat_flow,
                    metric_name="utility_heat_flow",
                    config=target.config,
                ),
                period_idx=target.period_idx,
            ),
        )
        for utility in utilities
    ]


def _period_duty(heat_flow: Value | None, *, period_idx: int | None):
    """Return the duty for the target's own period.

    Utility streams carry one duty per period, but a period's target only sets
    its own entry; the others are left from the copied input.
    """
    if heat_flow is None or period_idx is None or heat_flow.num_periods <= 1:
        return heat_flow
    return Value(float(heat_flow.period_values[period_idx]), heat_flow.unit)


def orc_result(target, *, is_total=False):
    result = _utility_summary_result(target, is_total=is_total)
    return result.model_copy(
        update={
            "orc_model": target.orc_model,
            "orc_fluid": target.orc_fluid,
            "orc_net_power": _metric(target, "orc_net_power"),
            "orc_heat_in": _metric(target, "orc_heat_in"),
            "orc_condenser_duty": _metric(target, "orc_condenser_duty"),
            "orc_thermal_efficiency": _metric(target, "orc_thermal_efficiency"),
            "orc_capital_cost": _metric(target, "orc_capital_cost"),
            "orc_annualized_capital_cost": _metric(
                target, "orc_annualized_capital_cost"
            ),
            "orc_power_value": _metric(target, "orc_power_value"),
            "orc_cooling_cost_change": _metric(target, "orc_cooling_cost_change"),
            "orc_total_annualized_cost_change": _metric(
                target, "orc_total_annualized_cost_change"
            ),
            "orc_evaporating_temperatures": target.orc_evaporating_temperatures,
            "orc_condensing_temperature": target.orc_condensing_temperature,
            "orc_stage_net_power": target.orc_stage_net_power,
        }
    )


THERMAL_RESULT_ADAPTERS = {
    UtilitySummaryTarget: _utility_summary_result,
    DirectIntegrationTarget: direct_result,
    IndirectIntegrationTarget: indirect_result,
    HeatPumpTargetBase: hpr_result,
    DirectOrcTarget: orc_result,
}

"""Transform runtime target state into external reporting contracts."""

from __future__ import annotations

from typing import Any

from ...contracts.reporting import HeatUtility, PinchTemp, TargetResults
from ...contracts.units import coerce_output_value
from ...domain.enums import SummaryRowType
from ...domain.stream_collection import StreamCollection
from ...domain.targets import (
    BaseTargetModel,
    DirectIntegrationTarget,
    HeatPumpTargetBase,
    IndirectIntegrationTarget,
    UtilitySummaryTarget,
)
from ...domain.value import Value


def target_to_result(
    target: BaseTargetModel,
    isTotal: bool = False,
) -> TargetResults:
    """Return the external reporting model for one runtime target."""
    if not target.reportable or not isinstance(target, UtilitySummaryTarget):
        raise NotImplementedError(
            f"Reporting is not defined for {type(target).__name__}."
        )

    result = _utility_summary_result(target, is_total=isTotal)
    if isinstance(target, DirectIntegrationTarget):
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
    if isinstance(target, IndirectIntegrationTarget):
        return result.model_copy(
            update={
                "work_target": _metric(target, "work_target"),
                "turbine_efficiency_target": _metric(
                    target,
                    "turbine_efficiency_target",
                ),
            }
        )
    if isinstance(target, HeatPumpTargetBase):
        return result.model_copy(
            update={
                "work_target": _metric(target, "work_target"),
                "turbine_efficiency_target": _metric(
                    target,
                    "turbine_efficiency_target",
                ),
                "hpr_cycle": target.hpr_cycle,
                "hpr_utility_total": _metric(target, "hpr_utility_total"),
                "hpr_work": _metric(target, "hpr_work"),
                "hpr_external_utility": _metric(target, "hpr_external_utility"),
                "hpr_ambient_hot": _metric(target, "hpr_ambient_hot"),
                "hpr_ambient_cold": _metric(target, "hpr_ambient_cold"),
                "hpr_cop": _metric(target, "hpr_cop"),
                "hpr_eta_he": _metric(target, "hpr_eta_he"),
                "hpr_operating_cost": _metric(target, "hpr_operating_cost"),
                "hpr_capital_cost": _metric(target, "hpr_capital_cost"),
                "hpr_annualized_capital_cost": _metric(
                    target,
                    "hpr_annualized_capital_cost",
                ),
                "hpr_total_annualized_cost": _metric(
                    target,
                    "hpr_total_annualized_cost",
                ),
                "hpr_compressor_capital_cost": _metric(
                    target,
                    "hpr_compressor_capital_cost",
                ),
                "hpr_heat_exchanger_capital_cost": _metric(
                    target,
                    "hpr_heat_exchanger_capital_cost",
                ),
                "hpr_success": target.hpr_success,
                "hpr_hot_streams": target.hpr_hot_streams,
                "hpr_cold_streams": target.hpr_cold_streams,
            }
        )
    return result


def serialize_target(
    target: BaseTargetModel,
    isTotal: bool = False,
) -> dict[str, Any]:
    """Return one target report as JSON-compatible Python data."""
    return _serialise_report_data(
        target_to_result(target, isTotal=isTotal).model_dump(mode="python")
    )


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
            heat_flow=coerce_output_value(
                utility.heat_flow,
                metric_name="utility_heat_flow",
                config=target.config,
            ),
        )
        for utility in utilities
    ]


def _serialise_report_data(value: Any) -> Any:
    if isinstance(value, Value):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialise_report_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise_report_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serialise_report_data(value.model_dump(mode="python"))
    return value


__all__ = ["serialize_target", "target_to_result"]

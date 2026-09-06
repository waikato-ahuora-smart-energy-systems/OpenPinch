"""Schemas for serialized summaries and report-facing data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..domain.analysis import AnalysisProvenance
from ..domain.stream_collection import StreamCollection
from ..domain.value import Value
from .hpr import HprTargetSimulationRecord
from .report_metrics import report_field
from .units import coerce_output_value
from .workspace import ValidationReport

_REPORT_MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
)


class _ReportModel(BaseModel):
    """Serialize report-owned domain values through explicit Pydantic v2 hooks."""

    model_config = _REPORT_MODEL_CONFIG

    @field_serializer("*", mode="wrap", when_used="json", check_fields=False)
    def _serialize_report_field(self, value, handler):
        if isinstance(value, Value):
            return value.to_dict()
        if isinstance(value, StreamCollection):
            return value.to_dict()
        return handler(value)


def _report_value_data(value, *, metric_name: str) -> Value | None:
    if value is None:
        return None
    if isinstance(value, Value):
        return Value(value)
    if hasattr(value, "model_dump") and not isinstance(value, Mapping):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping) and value.get("unit") is not None:
        return Value(value)
    if hasattr(value, "unit") and getattr(value, "unit", None) is not None:
        return Value(value)
    return coerce_output_value(value, metric_name=metric_name)


class HeatUtility(_ReportModel):
    """Report-friendly representation of a utility contribution."""

    name: str
    heat_flow: Value

    @field_validator("heat_flow", mode="before")
    @classmethod
    def _coerce_heat_flow(cls, value):
        return _report_value_data(value, metric_name="utility_heat_flow")


class PinchTemp(_ReportModel):
    """Hot and cold pinch temperatures attached to a targeting record."""

    cold_temp: Value | None = None
    hot_temp: Value | None = None

    @field_validator("cold_temp", mode="before")
    @classmethod
    def _coerce_cold_temp(cls, value):
        return _report_value_data(value, metric_name="cold_temp")

    @field_validator("hot_temp", mode="before")
    @classmethod
    def _coerce_hot_temp(cls, value):
        return _report_value_data(value, metric_name="hot_temp")


class TargetResults(_ReportModel):
    """Summary metrics for a single zone/target returned by the analysis."""

    scope: str = report_field("scope", "consensus", representation="structured")
    provenance: AnalysisProvenance | None = report_field(
        "provenance", "exclude", None, representation="structured"
    )
    zone_type: str = report_field("zone_type", "consensus", representation="structured")
    integration_type: str = report_field(
        "integration_type", "consensus", representation="structured"
    )
    target_method: str = report_field(
        "target_method", "consensus", representation="structured"
    )
    period_idx: Optional[int] = report_field(
        "period_idx", "exclude", None, representation="structured"
    )
    period_id: Optional[str] = report_field(
        "period_id", "derived", None, representation="structured"
    )
    degree_of_integration: Value | None = report_field(
        "degree_of_integration", "weighted_mean", None, representation="quantity"
    )
    Qh: Value = report_field("Qh", "weighted_mean", representation="quantity")
    Qc: Value = report_field("Qc", "weighted_mean", representation="quantity")
    Qr: Value = report_field("Qr", "weighted_mean", representation="quantity")
    utility_cost: Value | None = report_field(
        "utility_cost", "weighted_mean", None, representation="quantity"
    )
    row_type: Optional[str] = report_field(
        "row_type", "consensus", None, representation="structured"
    )
    hot_utilities: List[HeatUtility] = report_field(
        "hot_utilities", "derived", default_factory=list, representation="structured"
    )
    cold_utilities: List[HeatUtility] = report_field(
        "cold_utilities", "derived", default_factory=list, representation="structured"
    )
    pinch_temp: PinchTemp = report_field(
        "pinch_temp", "derived", representation="structured"
    )
    work_target: Value | None = report_field(
        "work_target", "weighted_mean", None, representation="quantity"
    )
    process_component_work_target: Value | None = report_field(
        "process_component_work_target",
        "weighted_mean",
        None,
        representation="quantity",
    )
    turbine_efficiency_target: Value | None = report_field(
        "turbine_efficiency_target", "weighted_mean", None, representation="quantity"
    )
    area: Value | None = report_field(
        "area", "weighted_mean", None, representation="quantity"
    )
    num_units: Optional[float] = report_field(
        "num_units", "weighted_mean", None, representation="scalar"
    )
    capital_cost: Value | None = report_field(
        "capital_cost", "weighted_mean", None, representation="quantity"
    )
    total_cost: Value | None = report_field(
        "total_cost", "weighted_mean", None, representation="quantity"
    )
    exergy_sources: Value | None = report_field(
        "exergy_sources", "weighted_mean", None, representation="quantity"
    )
    exergy_sinks: Value | None = report_field(
        "exergy_sinks", "weighted_mean", None, representation="quantity"
    )
    ETE: Value | None = report_field(
        "ETE", "weighted_mean", None, representation="quantity"
    )
    exergy_req_min: Value | None = report_field(
        "exergy_req_min", "weighted_mean", None, representation="quantity"
    )
    exergy_des_min: Value | None = report_field(
        "exergy_des_min", "weighted_mean", None, representation="quantity"
    )
    hpr_cycle: Optional[str] = report_field(
        "hpr_cycle", "consensus", None, representation="structured"
    )
    hpr_simulation_backend: Literal["coolprop", "tespy"] | None = report_field(
        "hpr_simulation_backend", "consensus", None, representation="structured"
    )
    hpr_target_simulation_record: HprTargetSimulationRecord | None = report_field(
        "hpr_target_simulation_record", "exclude", None, representation="structured"
    )
    hpr_utility_total: Value | None = report_field(
        "hpr_utility_total", "weighted_mean", None, representation="quantity"
    )
    hpr_work: Value | None = report_field(
        "hpr_work", "weighted_mean", None, representation="quantity"
    )
    hpr_external_utility: Value | None = report_field(
        "hpr_external_utility", "weighted_mean", None, representation="quantity"
    )
    hpr_ambient_hot: Value | None = report_field(
        "hpr_ambient_hot", "weighted_mean", None, representation="quantity"
    )
    hpr_ambient_cold: Value | None = report_field(
        "hpr_ambient_cold", "weighted_mean", None, representation="quantity"
    )
    hpr_cop: Value | None = report_field(
        "hpr_cop", "weighted_mean", None, representation="quantity"
    )
    hpr_eta_he: Value | None = report_field(
        "hpr_eta_he", "weighted_mean", None, representation="quantity"
    )
    hpr_operating_cost: Value | None = report_field(
        "hpr_operating_cost", "weighted_mean", None, representation="quantity"
    )
    hpr_capital_cost: Value | None = report_field(
        "hpr_capital_cost", "maximum", None, representation="quantity"
    )
    hpr_annualized_capital_cost: Value | None = report_field(
        "hpr_annualized_capital_cost", "maximum", None, representation="quantity"
    )
    hpr_total_annualized_cost: Value | None = report_field(
        "hpr_total_annualized_cost", "derived", None, representation="structured"
    )
    hpr_compressor_capital_cost: Value | None = report_field(
        "hpr_compressor_capital_cost", "maximum", None, representation="quantity"
    )
    hpr_heat_exchanger_capital_cost: Value | None = report_field(
        "hpr_heat_exchanger_capital_cost", "maximum", None, representation="quantity"
    )
    hpr_success: Optional[bool] = report_field(
        "hpr_success", "consensus", None, representation="structured"
    )
    hpr_hot_streams: Optional[StreamCollection] = report_field(
        "hpr_hot_streams", "consensus", None, representation="structured"
    )
    hpr_cold_streams: Optional[StreamCollection] = report_field(
        "hpr_cold_streams", "consensus", None, representation="structured"
    )

    @field_validator(
        "degree_of_integration",
        "Qh",
        "Qc",
        "Qr",
        "utility_cost",
        "work_target",
        "process_component_work_target",
        "turbine_efficiency_target",
        "area",
        "capital_cost",
        "total_cost",
        "exergy_sources",
        "exergy_sinks",
        "ETE",
        "exergy_req_min",
        "exergy_des_min",
        "hpr_utility_total",
        "hpr_work",
        "hpr_external_utility",
        "hpr_ambient_hot",
        "hpr_ambient_cold",
        "hpr_cop",
        "hpr_eta_he",
        "hpr_operating_cost",
        "hpr_capital_cost",
        "hpr_annualized_capital_cost",
        "hpr_total_annualized_cost",
        "hpr_compressor_capital_cost",
        "hpr_heat_exchanger_capital_cost",
        mode="before",
    )
    @classmethod
    def _coerce_report_values(cls, value, info):
        return _report_value_data(value, metric_name=info.field_name)


class ReportMetric(BaseModel):
    """One numeric report metric resolved for a target and optional state."""

    scope: str
    zone_type: str
    integration_type: str
    target_method: str
    metric: str
    label: str
    value: Any = None
    unit: Optional[str] = None
    period_id: Optional[str] = None


class GraphAvailability(BaseModel):
    """One graph available from a solved report data."""

    graph_id: str
    graph_set_id: str
    target_name: str
    zone_name: Optional[str] = None
    zone_address: Optional[str] = None
    target_type: Optional[str] = None
    graph_type: Optional[str] = None
    graph_name: str
    index: int


class ProblemReport(BaseModel):
    """Typed report data for script, notebook, and export workflows."""

    project_name: str
    solved: bool
    validation: ValidationReport
    targets: List[TargetResults] = Field(default_factory=list)
    metrics: List[ReportMetric] = Field(default_factory=list)
    graph_catalog: List[GraphAvailability] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "GraphAvailability",
    "HeatUtility",
    "ProblemReport",
    "ReportMetric",
    "TargetResults",
    "PinchTemp",
]


TargetResults.model_rebuild()

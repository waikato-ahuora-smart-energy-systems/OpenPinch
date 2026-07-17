"""Schemas for serialized summaries and report-facing data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..domain.stream_collection import StreamCollection
from ..domain.value import Value
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

    name: str
    period_idx: Optional[int] = None
    period_id: Optional[str] = None
    degree_of_integration: Value | None = None
    Qh: Value
    Qc: Value
    Qr: Value
    utility_cost: Value | None = None
    row_type: Optional[str] = None
    hot_utilities: List[HeatUtility] = Field(default_factory=list)
    cold_utilities: List[HeatUtility] = Field(default_factory=list)
    pinch_temp: PinchTemp
    work_target: Value | None = None
    process_component_work_target: Value | None = None
    turbine_efficiency_target: Value | None = None
    area: Value | None = None
    num_units: Optional[float] = None
    capital_cost: Value | None = None
    total_cost: Value | None = None
    exergy_sources: Value | None = None
    exergy_sinks: Value | None = None
    ETE: Value | None = None
    exergy_req_min: Value | None = None
    exergy_des_min: Value | None = None
    hpr_cycle: Optional[str] = None
    hpr_utility_total: Value | None = None
    hpr_work: Value | None = None
    hpr_external_utility: Value | None = None
    hpr_ambient_hot: Value | None = None
    hpr_ambient_cold: Value | None = None
    hpr_cop: Value | None = None
    hpr_eta_he: Value | None = None
    hpr_operating_cost: Value | None = None
    hpr_capital_cost: Value | None = None
    hpr_annualized_capital_cost: Value | None = None
    hpr_total_annualized_cost: Value | None = None
    hpr_compressor_capital_cost: Value | None = None
    hpr_heat_exchanger_capital_cost: Value | None = None
    hpr_success: Optional[bool] = None
    hpr_hot_streams: Optional[StreamCollection] = None
    hpr_cold_streams: Optional[StreamCollection] = None

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

    target_name: str
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

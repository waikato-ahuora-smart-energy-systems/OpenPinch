"""Schemas for serialized summaries and report-facing data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...classes.stream_collection import StreamCollection
from ...classes.value import Value
from ..unit_system import coerce_output_value

_REPORT_MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    json_encoders={Value: lambda value: value.to_dict()},
)


def _report_value_payload(value, *, metric_name: str) -> Value | None:
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


class HeatUtility(BaseModel):
    """Report-friendly representation of a utility contribution."""

    model_config = _REPORT_MODEL_CONFIG

    name: str
    heat_flow: Value

    @field_validator("heat_flow", mode="before")
    @classmethod
    def _coerce_heat_flow(cls, value):
        return _report_value_payload(value, metric_name="utility_heat_flow")


class PinchTemp(BaseModel):
    """Hot and cold pinch temperatures attached to a targeting record."""

    model_config = _REPORT_MODEL_CONFIG

    cold_temp: Value | None = None
    hot_temp: Value | None = None

    @field_validator("cold_temp", mode="before")
    @classmethod
    def _coerce_cold_temp(cls, value):
        return _report_value_payload(value, metric_name="cold_temp")

    @field_validator("hot_temp", mode="before")
    @classmethod
    def _coerce_hot_temp(cls, value):
        return _report_value_payload(value, metric_name="hot_temp")


class TargetResults(BaseModel):
    """Summary metrics for a single zone/target returned by the analysis."""

    model_config = _REPORT_MODEL_CONFIG

    name: str
    idx: Optional[int] = None
    state_id: Optional[str] = None
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
        mode="before",
    )
    @classmethod
    def _coerce_report_values(cls, value, info):
        return _report_value_payload(value, metric_name=info.field_name)


__all__ = [
    "HeatUtility",
    "TargetResults",
    "PinchTemp",
]


TargetResults.model_rebuild()

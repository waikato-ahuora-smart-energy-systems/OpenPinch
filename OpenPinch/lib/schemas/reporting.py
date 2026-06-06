"""Schemas for serialized summaries and report-facing data."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...classes.stream_collection import StreamCollection
from .common import HPRMetric, MaybeVU, ScalarOrVU


class HeatUtility(BaseModel):
    """Report-friendly representation of a utility contribution."""

    name: str
    heat_flow: ScalarOrVU


class TempPinch(BaseModel):
    """Hot and cold pinch temperatures attached to a targeting record."""

    cold_temp: MaybeVU = None
    hot_temp: MaybeVU = None


class TargetResults(BaseModel):
    """Summary metrics for a single zone/target returned by the analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    idx: Optional[int] = None
    state_id: Optional[str] = None
    degree_of_integration: MaybeVU = None
    Qh: ScalarOrVU
    Qc: ScalarOrVU
    Qr: ScalarOrVU
    utility_cost: MaybeVU = None
    row_type: Optional[str] = None
    hot_utilities: List[HeatUtility] = Field(default_factory=list)
    cold_utilities: List[HeatUtility] = Field(default_factory=list)
    temp_pinch: TempPinch
    work_target: MaybeVU = None
    turbine_efficiency_target: MaybeVU = None
    area: MaybeVU = None
    num_units: Optional[float] = None
    capital_cost: Optional[float] = None
    total_cost: Optional[float] = None
    exergy_sources: MaybeVU = None
    exergy_sinks: MaybeVU = None
    ETE: Optional[float] = None
    exergy_req_min: MaybeVU = None
    exergy_des_min: MaybeVU = None
    hpr_cycle: Optional[str] = None
    hpr_utility_total: HPRMetric = None
    hpr_work: HPRMetric = None
    hpr_external_utility: HPRMetric = None
    hpr_ambient_hot: HPRMetric = None
    hpr_ambient_cold: HPRMetric = None
    hpr_cop: HPRMetric = None
    hpr_eta_he: HPRMetric = None
    hpr_success: Optional[bool] = None
    hpr_hot_streams: Optional[StreamCollection] = None
    hpr_cold_streams: Optional[StreamCollection] = None


__all__ = [
    "HeatUtility",
    "TargetResults",
    "TempPinch",
]


TargetResults.model_rebuild()

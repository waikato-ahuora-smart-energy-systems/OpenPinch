"""Rich runtime target models used by OpenPinch services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..classes.problem_table import ProblemTable
from ..classes.stream_collection import StreamCollection
from .config import Configuration, tol
from .enums import SummaryRowType

if TYPE_CHECKING:
    from ..classes.zone import Zone


def _normalise_target_name(
    *,
    zone_name: Optional[str],
    target_type: Optional[str],
    name: Optional[str],
) -> str:
    if not target_type:
        raise ValueError("type is required.")
    if zone_name == "":
        raise ValueError("zone_name is required.")
    if not name and not zone_name:
        raise ValueError("zone_name or name is required.")
    if name:
        return str(name)

    suffix = f"/{target_type}"
    assert zone_name is not None
    return zone_name if str(zone_name).endswith(suffix) else f"{zone_name}{suffix}"


def _serialise_temp_pinch(cold_pinch: Optional[float], hot_pinch: Optional[float]) -> dict[str, float | None]:
    if isinstance(cold_pinch, float) and isinstance(hot_pinch, float):
        if abs(cold_pinch - hot_pinch) < tol:
            return {"cold_temp": cold_pinch}
        return {"cold_temp": cold_pinch, "hot_temp": hot_pinch}
    if isinstance(cold_pinch, float):
        return {"cold_temp": cold_pinch}
    if isinstance(hot_pinch, float):
        return {"hot_temp": hot_pinch}
    return {"cold_temp": None, "hot_temp": None}


def _serialise_utilities(utilities: StreamCollection) -> list[dict[str, Any]]:
    return [{"name": utility.name, "heat_flow": utility.heat_flow} for utility in utilities]


class BaseTargetModel(BaseModel):
    """Shared metadata for all solved target objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    zone_name: Optional[str] = Field(default=None, exclude=True, repr=False)
    name: str
    type: str
    parent_zone: Any = None
    config: Configuration = Field(default_factory=Configuration)
    active: bool = True

    @model_validator(mode="before")
    @classmethod
    def _set_name(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data["name"] = _normalise_target_name(
            zone_name=data.get("zone_name"),
            target_type=data.get("type"),
            name=data.get("name"),
        )
        return data

    def serialize_json(self, isTotal: bool = False) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must implement serialize_json().")


class GraphBackedTarget(BaseTargetModel):
    """Target with graph payloads attached."""

    graphs: dict[str, Any] = Field(default_factory=dict)

    def add_graph(self, name: str, result: Any) -> None:
        self.graphs[name] = result


class UtilitySummaryTarget(BaseTargetModel):
    """Target that returns utility duties and recovered heat summaries."""

    hot_utilities: StreamCollection = Field(default_factory=StreamCollection)
    cold_utilities: StreamCollection = Field(default_factory=StreamCollection)

    hot_utility_target: float
    cold_utility_target: float
    heat_recovery_target: float
    heat_recovery_limit: Optional[float] = None
    degree_of_int: Optional[float] = None
    utility_cost: float = 0.0

    hot_pinch: Optional[float] = None
    cold_pinch: Optional[float] = None

    @property
    def utility_streams(self) -> StreamCollection:
        return self.hot_utilities + self.cold_utilities

    def calc_utility_cost(self) -> float:
        self.utility_cost = sum(u.ut_cost for u in self.utility_streams)
        return float(self.utility_cost)

    def serialize_json(self, isTotal: bool = False) -> dict[str, Any]:
        degree_of_integration = None
        if self.degree_of_int is not None:
            degree_of_integration = self.degree_of_int * 100

        return {
            "name": self.name,
            "degree_of_integration": degree_of_integration,
            "Qh": self.hot_utility_target,
            "Qc": self.cold_utility_target,
            "Qr": self.heat_recovery_target,
            "utility_cost": self.utility_cost,
            "row_type": SummaryRowType.FOOTER.value
            if isTotal
            else SummaryRowType.CONTENT.value,
            "hot_utilities": _serialise_utilities(self.hot_utilities),
            "cold_utilities": _serialise_utilities(self.cold_utilities),
            "temp_pinch": _serialise_temp_pinch(self.cold_pinch, self.hot_pinch),
        }


class DirectIntegrationTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Detailed direct-integration target returned by `compute_direct_integration_targets`."""

    pt: ProblemTable
    pt_real: ProblemTable

    utility_heat_recovery_target: Optional[float] = None

    area: Optional[float] = None
    num_units: Optional[float] = None
    capital_cost: Optional[float] = None
    total_cost: Optional[float] = None

    exergy_sinks: Optional[float] = None
    exergy_sources: Optional[float] = None
    exergy_des_min: Optional[float] = None
    exergy_req_min: Optional[float] = None
    ETE: Optional[float] = None

    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None

    def serialize_json(self, isTotal: bool = False) -> dict[str, Any]:
        data = super().serialize_json(isTotal=isTotal)
        if self.area is not None or self.num_units is not None:
            data["area"] = self.area
            data["num_units"] = self.num_units
            data["capital_cost"] = self.capital_cost
            data["total_cost"] = self.total_cost
        if self.exergy_sources is not None or self.exergy_sinks is not None:
            data["exergy_sources"] = self.exergy_sources
            data["exergy_sinks"] = self.exergy_sinks
            data["ETE"] = None if self.ETE is None else self.ETE * 100
            data["exergy_req_min"] = self.exergy_req_min
            data["exergy_des_min"] = self.exergy_des_min
        if self.work_target is not None or self.turbine_efficiency_target is not None:
            data["work_target"] = self.work_target
            data["turbine_efficiency_target"] = (
                None
                if self.turbine_efficiency_target is None
                else self.turbine_efficiency_target * 100
            )
        return data


class TotalProcessTarget(UtilitySummaryTarget):
    """Aggregated process-level utility summary built from solved subzones."""


class TotalSiteTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Total-site / indirect-integration target with site problem tables and graphs."""

    pt: ProblemTable
    pt_real: ProblemTable

    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None

    def serialize_json(self, isTotal: bool = False) -> dict[str, Any]:
        data = super().serialize_json(isTotal=isTotal)
        if self.work_target is not None or self.turbine_efficiency_target is not None:
            data["work_target"] = self.work_target
            data["turbine_efficiency_target"] = (
                None
                if self.turbine_efficiency_target is None
                else self.turbine_efficiency_target * 100
            )
        return data


class HeatPumpTargetBase(GraphBackedTarget):
    """Base contract for advanced HPR targets returned by explicit `target_*` methods."""

    pt: ProblemTable
    hpr_cycle: str
    hpr_utility_total: Any
    hpr_work: Any
    hpr_external_utility: Any
    hpr_ambient_hot: Any
    hpr_ambient_cold: Any
    hpr_cop: Any
    hpr_eta_he: Any
    hpr_success: bool
    hpr_hot_streams: StreamCollection
    hpr_cold_streams: StreamCollection
    hpr_details: Any

    def serialize_json(self, isTotal: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "degree_of_integration": None,
            "Qh": 0.0,
            "Qc": 0.0,
            "Qr": 0.0,
            "utility_cost": None,
            "row_type": SummaryRowType.FOOTER.value
            if isTotal
            else SummaryRowType.CONTENT.value,
            "hot_utilities": [],
            "cold_utilities": [],
            "temp_pinch": {"cold_temp": None, "hot_temp": None},
            "hpr_cycle": self.hpr_cycle,
            "hpr_utility_total": self.hpr_utility_total,
            "hpr_work": self.hpr_work,
            "hpr_external_utility": self.hpr_external_utility,
            "hpr_ambient_hot": self.hpr_ambient_hot,
            "hpr_ambient_cold": self.hpr_ambient_cold,
            "hpr_cop": self.hpr_cop,
            "hpr_eta_he": self.hpr_eta_he,
            "hpr_success": self.hpr_success,
            "hpr_hot_streams": self.hpr_hot_streams,
            "hpr_cold_streams": self.hpr_cold_streams,
        }


class DirectHeatPumpTarget(HeatPumpTargetBase):
    """Direct heat-pump targeting result."""


class IndirectHeatPumpTarget(HeatPumpTargetBase):
    """Indirect heat-pump targeting result."""


class DirectRefrigerationTarget(HeatPumpTargetBase):
    """Direct refrigeration targeting result."""


class IndirectRefrigerationTarget(HeatPumpTargetBase):
    """Indirect refrigeration targeting result."""


AnyTargetModel = (
    DirectIntegrationTarget
    | TotalProcessTarget
    | TotalSiteTarget
    | DirectHeatPumpTarget
    | IndirectHeatPumpTarget
    | DirectRefrigerationTarget
    | IndirectRefrigerationTarget
)


__all__ = [
    "AnyTargetModel",
    "BaseTargetModel",
    "DirectHeatPumpTarget",
    "DirectIntegrationTarget",
    "DirectRefrigerationTarget",
    "HeatPumpTargetBase",
    "IndirectHeatPumpTarget",
    "IndirectRefrigerationTarget",
    "TotalProcessTarget",
    "TotalSiteTarget",
    "UtilitySummaryTarget",
]

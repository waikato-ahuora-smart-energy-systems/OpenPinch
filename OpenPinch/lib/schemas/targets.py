"""Runtime target schemas used by OpenPinch analysis services."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...classes.problem_table import ProblemTable
from ...classes.stream_collection import StreamCollection
from ..config import Configuration
from ..enums import SummaryRowType
from .reporting import HeatUtility, TargetResults, TempPinch


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


def _build_temp_pinch(
    cold_pinch: Optional[float],
    hot_pinch: Optional[float],
) -> TempPinch:
    if isinstance(cold_pinch, float) and isinstance(hot_pinch, float):
        return TempPinch(cold_temp=cold_pinch, hot_temp=hot_pinch)
    if isinstance(cold_pinch, float):
        return TempPinch(cold_temp=cold_pinch)
    if isinstance(hot_pinch, float):
        return TempPinch(hot_temp=hot_pinch)
    return TempPinch()


def _build_heat_utilities(utilities: StreamCollection) -> list[HeatUtility]:
    return [
        HeatUtility(name=utility.name, heat_flow=utility.heat_flow)
        for utility in utilities
    ]


def _row_type(isTotal: bool) -> str:
    return SummaryRowType.FOOTER.value if isTotal else SummaryRowType.CONTENT.value


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

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Convert the runtime target into the exported reporting schema."""
        raise NotImplementedError

    def serialize_json(self, isTotal: bool = False) -> dict[str, Any]:
        """Serialise the reporting-schema view of this target to plain Python."""
        return self.to_target_results(isTotal=isTotal).model_dump(mode="python")


class GraphBackedTarget(BaseTargetModel):
    """Target with graph payloads attached."""

    graphs: dict[str, Any] = Field(default_factory=dict)

    def add_graph(self, name: str, result: Any) -> None:
        """Attach one graph payload under ``name`` for later export."""
        self.graphs[name] = result


class UtilitySummaryTarget(BaseTargetModel):
    """Target that returns utility duties and recovered-heat summaries."""

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
        """Return hot and cold utilities as one combined collection."""
        return self.hot_utilities + self.cold_utilities

    def calc_utility_cost(self) -> float:
        """Calculate and cache the total utility cost across attached utilities."""
        self.utility_cost = sum(u.ut_cost for u in self.utility_streams)
        return float(self.utility_cost)

    def _base_target_results(self, isTotal: bool = False) -> TargetResults:
        degree_of_integration = None
        if self.degree_of_int is not None:
            degree_of_integration = self.degree_of_int * 100

        return TargetResults(
            name=self.name,
            degree_of_integration=degree_of_integration,
            Qh=self.hot_utility_target,
            Qc=self.cold_utility_target,
            Qr=self.heat_recovery_target,
            utility_cost=self.utility_cost,
            row_type=_row_type(isTotal),
            hot_utilities=_build_heat_utilities(self.hot_utilities),
            cold_utilities=_build_heat_utilities(self.cold_utilities),
            temp_pinch=_build_temp_pinch(self.cold_pinch, self.hot_pinch),
        )

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Return the common reporting payload for utility-summary targets."""
        return self._base_target_results(isTotal=isTotal)


class DirectIntegrationTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Detailed direct-integration runtime target."""

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

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Return the reporting payload including DI-only cost and work fields."""
        base = self._base_target_results(isTotal=isTotal)
        return base.model_copy(
            update={
                "work_target": self.work_target,
                "turbine_efficiency_target": None
                if self.turbine_efficiency_target is None
                else self.turbine_efficiency_target * 100,
                "area": self.area,
                "num_units": self.num_units,
                "capital_cost": self.capital_cost,
                "total_cost": self.total_cost,
                "exergy_sources": self.exergy_sources,
                "exergy_sinks": self.exergy_sinks,
                "ETE": None if self.ETE is None else self.ETE * 100,
                "exergy_req_min": self.exergy_req_min,
                "exergy_des_min": self.exergy_des_min,
            }
        )


class TotalProcessTarget(UtilitySummaryTarget):
    """Aggregated process-level utility summary built from solved subzones."""


class TotalSiteTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Total Site / indirect integration target with site Problem Tables and graphs."""

    pt: ProblemTable
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Return the reporting payload including Total Site work fields."""
        base = self._base_target_results(isTotal=isTotal)
        return base.model_copy(
            update={
                "work_target": self.work_target,
                "turbine_efficiency_target": None
                if self.turbine_efficiency_target is None
                else self.turbine_efficiency_target * 100,
            }
        )


class HeatPumpTargetBase(GraphBackedTarget):
    """Base contract for advanced HPR targets from explicit ``target_*`` methods."""

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

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Return the reporting payload for explicit HPR target results."""
        return TargetResults(
            name=self.name,
            degree_of_integration=None,
            Qh=0.0,
            Qc=0.0,
            Qr=0.0,
            utility_cost=None,
            row_type=_row_type(isTotal),
            hot_utilities=[],
            cold_utilities=[],
            temp_pinch=TempPinch(),
            hpr_cycle=self.hpr_cycle,
            hpr_utility_total=self.hpr_utility_total,
            hpr_work=self.hpr_work,
            hpr_external_utility=self.hpr_external_utility,
            hpr_ambient_hot=self.hpr_ambient_hot,
            hpr_ambient_cold=self.hpr_ambient_cold,
            hpr_cop=self.hpr_cop,
            hpr_eta_he=self.hpr_eta_he,
            hpr_success=self.hpr_success,
            hpr_hot_streams=self.hpr_hot_streams,
            hpr_cold_streams=self.hpr_cold_streams,
        )


class DirectHeatPumpTarget(HeatPumpTargetBase):
    """Direct heat pump targeting result."""


class IndirectHeatPumpTarget(HeatPumpTargetBase):
    """Indirect heat pump targeting result."""


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

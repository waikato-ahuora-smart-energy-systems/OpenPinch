"""Runtime target schemas used by OpenPinch analysis services."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .configuration import Configuration
from .problem_table import ProblemTable
from .stream_collection import StreamCollection


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


class BaseTargetModel(BaseModel):
    """Shared metadata for all solved target objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    zone_name: Optional[str] = Field(default=None, exclude=True, repr=False)
    period_id: Optional[str] = None
    period_idx: Optional[int] = Field(default=None, exclude=True, repr=False)
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


class GraphBackedTarget(BaseTargetModel):
    """Target with graph data attached."""

    graphs: dict[str, Any] = Field(default_factory=dict)

    def add_graph(self, name: str, result: Any) -> None:
        """Attach one graph data under ``name`` for later export."""
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
    process_component_work_target: Optional[float] = None
    exergy_sinks: Optional[float] = None
    exergy_sources: Optional[float] = None
    exergy_des_min: Optional[float] = None
    exergy_req_min: Optional[float] = None
    ETE: Optional[float] = None

    @property
    def utility_streams(self) -> StreamCollection:
        """Return hot and cold utilities as one combined collection."""
        return self.hot_utilities + self.cold_utilities

    def calc_utility_cost(self) -> float:
        """Calculate and cache the total utility cost across attached utilities."""
        self.utility_cost = sum(u.ut_cost for u in self.utility_streams)
        return float(self.utility_cost)


class DirectIntegrationTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Detailed direct-integration runtime target."""

    pt: ProblemTable
    pt_real: ProblemTable
    utility_heat_recovery_target: Optional[float] = None
    area: Optional[float] = None
    num_units: Optional[float] = None
    capital_cost: Optional[float] = None
    total_cost: Optional[float] = None
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None


class TotalProcessTarget(UtilitySummaryTarget):
    """Aggregated process-level utility summary built from solved subzones."""


class TotalSiteTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Total Site / indirect integration target with site Problem Tables and graphs."""

    pt: ProblemTable
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None


class EnergyTransferTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Energy transfer diagram and heat-surplus/deficit table target."""

    pt: ProblemTable
    base_target_type: str
    base_target_name: str
    heat_surplus_deficit_table: list[dict[str, Any]] = Field(default_factory=list)
    energy_transfer_diagram: dict[str, Any] = Field(default_factory=dict)


class HeatPumpTargetBase(GraphBackedTarget, UtilitySummaryTarget):
    """Base contract for advanced HPR targets from explicit ``target_*`` methods."""

    pt: ProblemTable
    hot_utilities: StreamCollection = Field(default_factory=StreamCollection)
    cold_utilities: StreamCollection = Field(default_factory=StreamCollection)
    hot_utility_target: float = 0.0
    cold_utility_target: float = 0.0
    heat_recovery_target: float = 0.0
    heat_recovery_limit: Optional[float] = None
    degree_of_int: Optional[float] = None
    utility_cost: float = 0.0
    hot_pinch: Optional[float] = None
    cold_pinch: Optional[float] = None
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None
    hpr_cycle: str
    hpr_utility_total: Any
    hpr_work: Any
    hpr_external_utility: Any
    hpr_ambient_hot: Any
    hpr_ambient_cold: Any
    hpr_cop: Any
    hpr_eta_he: Any
    hpr_operating_cost: Any = None
    hpr_capital_cost: Any = None
    hpr_annualized_capital_cost: Any = None
    hpr_total_annualized_cost: Any = None
    hpr_compressor_capital_cost: Any = None
    hpr_heat_exchanger_capital_cost: Any = None
    hpr_success: bool
    hpr_hot_streams: StreamCollection
    hpr_cold_streams: StreamCollection
    hpr_details: Any


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
    | EnergyTransferTarget
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
    "EnergyTransferTarget",
    "HeatPumpTargetBase",
    "IndirectHeatPumpTarget",
    "IndirectRefrigerationTarget",
    "TotalProcessTarget",
    "TotalSiteTarget",
    "UtilitySummaryTarget",
]

"""Runtime target schemas used by OpenPinch analysis services."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import AnalysisProvenance
from .configuration import Configuration
from .enums import IntegrationType, TargetMethod, TargetType, ZoneType
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

    target_types: ClassVar[tuple[str, ...]] = ()
    integration_route: ClassVar[str | None] = None
    analysis_method: ClassVar[str | None] = None
    prerequisite_target_types: ClassVar[tuple[str, ...]] = ()

    def prerequisite_types(self) -> tuple[str, ...]:
        return self.prerequisite_target_types

    @classmethod
    def _classification(cls, data):
        if cls.integration_route is not None and cls.analysis_method is not None:
            return cls.integration_route, cls.analysis_method
        family = _TARGET_FAMILIES.get(str(data.get("type", "")))
        if family is None:
            raise ValueError(
                f"No reporting classification for target type {data.get('type')!r}."
            )
        return family._classification(data)

    zone_name: Optional[str] = Field(default=None, exclude=True, repr=False)
    provenance: AnalysisProvenance | None = None
    scope: str
    zone_type: str = ZoneType.P.value
    integration_type: str
    target_method: str
    period_id: Optional[str] = None
    period_idx: Optional[int] = Field(default=None, exclude=True, repr=False)
    name: str
    type: str
    parent_zone: Any = None
    config: Configuration = Field(default_factory=Configuration)
    active: bool = True
    reportable: bool = Field(default=True, exclude=True, repr=False)

    def provenance_settings(self) -> dict[str, Any]:
        """Effective calculation settings, including family-owned runtime choices."""
        return dict(self.config._values)

    @model_validator(mode="before")
    @classmethod
    def _set_name(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        zone_name = data.get("zone_name")
        parent_zone = data.get("parent_zone")
        scope = data.get("scope")
        if not scope:
            if (
                parent_zone is not None
                and hasattr(parent_zone, "address")
                and zone_name
            ):
                scope = f"{parent_zone.address}/{zone_name}"
            else:
                scope = zone_name
        if not scope:
            raise ValueError("scope or zone_name is required.")
        data["scope"] = str(scope)
        if not data.get("type"):
            raise ValueError("type is required.")
        integration_type, target_method = cls._classification(data)
        data["integration_type"] = integration_type
        data["target_method"] = target_method
        data["name"] = _normalise_target_name(
            zone_name=data["scope"],
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
        self.utility_cost = sum(u.utility_cost for u in self.utility_streams)
        return float(self.utility_cost)


class DirectIntegrationTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Detailed direct-integration runtime target."""

    target_types = (
        TargetType.DI.value,
        TargetType.TL.value,
    )
    integration_route = IntegrationType.Process.value
    analysis_method = TargetMethod.HeatExchange.value

    pt: ProblemTable
    pt_real: ProblemTable
    utility_heat_recovery_target: Optional[float] = None
    area: Optional[float] = None
    num_units: Optional[float] = None
    capital_cost: Optional[float] = None
    total_cost: Optional[float] = None
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None


class SubzoneAggregateTarget(UtilitySummaryTarget):
    """Internal utility summary built from immediate solved subzones."""

    target_types = (TargetType.SA.value,)
    integration_route = IntegrationType.Utility.value
    analysis_method = TargetMethod.HeatExchange.value

    reportable: bool = Field(default=False, exclude=True, repr=False)


class IndirectIntegrationTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Utility-mediated integration target for an aggregate Zone scope."""

    target_types = (TargetType.II.value,)
    prerequisite_target_types = (TargetType.DI.value, TargetType.SA.value)
    integration_route = IntegrationType.Utility.value
    analysis_method = TargetMethod.HeatExchange.value

    pt: ProblemTable
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None


class EnergyTransferTarget(GraphBackedTarget, UtilitySummaryTarget):
    """Energy transfer diagram and heat-surplus/deficit table target."""

    target_types = (TargetType.ET.value,)

    def prerequisite_types(self) -> tuple[str, ...]:
        return (self.base_target_type,)

    integration_route = IntegrationType.Process.value
    analysis_method = TargetMethod.EnergyTransfer.value

    @classmethod
    def _classification(cls, data):
        base = _TARGET_FAMILIES.get(data.get("base_target_type"))
        route = base.integration_route if base is not None else cls.integration_route
        return route, cls.analysis_method

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
    hpr_simulation_backend: Literal["coolprop", "tespy"] = "coolprop"
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

    def provenance_settings(self) -> dict[str, Any]:
        return {
            **super().provenance_settings(),
            "simulation_backend": self.hpr_simulation_backend,
        }


class DirectHeatPumpTarget(HeatPumpTargetBase):
    """Direct heat pump targeting result."""

    target_types = (TargetType.DHP.value,)
    prerequisite_target_types = (TargetType.DI.value,)
    integration_route = IntegrationType.Process.value
    analysis_method = TargetMethod.HeatPump.value


class IndirectHeatPumpTarget(HeatPumpTargetBase):
    """Indirect heat pump targeting result."""

    target_types = (TargetType.IHP.value,)
    prerequisite_target_types = (TargetType.II.value,)
    integration_route = IntegrationType.Utility.value
    analysis_method = TargetMethod.HeatPump.value


class DirectRefrigerationTarget(HeatPumpTargetBase):
    """Direct refrigeration targeting result."""

    target_types = (TargetType.DR.value,)
    prerequisite_target_types = (TargetType.DI.value,)
    integration_route = IntegrationType.Process.value
    analysis_method = TargetMethod.Refrigeration.value


class IndirectRefrigerationTarget(HeatPumpTargetBase):
    """Indirect refrigeration targeting result."""

    target_types = (TargetType.IR.value,)
    prerequisite_target_types = (TargetType.II.value,)
    integration_route = IntegrationType.Utility.value
    analysis_method = TargetMethod.Refrigeration.value


AnyTargetModel = (
    DirectIntegrationTarget
    | SubzoneAggregateTarget
    | IndirectIntegrationTarget
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
    "IndirectIntegrationTarget",
    "IndirectRefrigerationTarget",
    "SubzoneAggregateTarget",
    "UtilitySummaryTarget",
]


_TARGET_FAMILIES = MappingProxyType(
    {
        target_type: family
        for family in (
            DirectIntegrationTarget,
            SubzoneAggregateTarget,
            IndirectIntegrationTarget,
            EnergyTransferTarget,
            DirectHeatPumpTarget,
            IndirectHeatPumpTarget,
            DirectRefrigerationTarget,
            IndirectRefrigerationTarget,
        )
        for target_type in family.target_types
    }
)

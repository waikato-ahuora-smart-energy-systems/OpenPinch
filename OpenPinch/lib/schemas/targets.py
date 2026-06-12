"""Runtime target schemas used by OpenPinch analysis services."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...classes.problem_table import ProblemTable
from ...classes.stream_collection import StreamCollection
from ...classes.value import Value
from ..config import Configuration
from ..enums import SummaryRowType
from ..unit_system import coerce_output_value
from .reporting import HeatUtility, PinchTemp, TargetResults


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
    *,
    config: Configuration,
) -> PinchTemp:
    return PinchTemp(
        cold_temp=coerce_output_value(
            cold_pinch,
            metric_name="cold_temp",
            config=config,
        ),
        hot_temp=coerce_output_value(
            hot_pinch,
            metric_name="hot_temp",
            config=config,
        ),
    )


def _build_heat_utilities(
    utilities: StreamCollection,
    *,
    config: Configuration,
) -> list[HeatUtility]:
    return [
        HeatUtility(
            name=utility.name,
            heat_flow=coerce_output_value(
                utility.heat_flow,
                metric_name="utility_heat_flow",
                config=config,
            ),
        )
        for utility in utilities
    ]


def _row_type(isTotal: bool) -> str:
    return SummaryRowType.FOOTER.value if isTotal else SummaryRowType.CONTENT.value


def _serialise_report_payload(value: Any) -> Any:
    if isinstance(value, Value):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialise_report_payload(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialise_report_payload(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serialise_report_payload(value.model_dump(mode="python"))
    return value


class BaseTargetModel(BaseModel):
    """Shared metadata for all solved target objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    zone_name: Optional[str] = Field(default=None, exclude=True, repr=False)
    state_id: Optional[str] = None
    state_idx: Optional[int] = Field(default=None, exclude=True, repr=False)
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
        return _serialise_report_payload(
            self.to_target_results(isTotal=isTotal).model_dump(mode="python")
        )


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

    def _base_target_results(self, isTotal: bool = False) -> TargetResults:
        degree_of_integration = None
        if self.degree_of_int is not None:
            degree_of_integration = coerce_output_value(
                self.degree_of_int,
                metric_name="degree_of_integration",
                config=self.config,
            )

        return TargetResults(
            name=self.name,
            idx=self.state_idx,
            state_id=self.state_id,
            degree_of_integration=degree_of_integration,
            Qh=coerce_output_value(
                self.hot_utility_target,
                metric_name="Qh",
                config=self.config,
            ),
            Qc=coerce_output_value(
                self.cold_utility_target,
                metric_name="Qc",
                config=self.config,
            ),
            Qr=coerce_output_value(
                self.heat_recovery_target,
                metric_name="Qr",
                config=self.config,
            ),
            utility_cost=coerce_output_value(
                self.utility_cost,
                metric_name="utility_cost",
                config=self.config,
            ),
            process_component_work_target=coerce_output_value(
                self.process_component_work_target,
                metric_name="work_target",
                config=self.config,
            ),
            row_type=_row_type(isTotal),
            hot_utilities=_build_heat_utilities(self.hot_utilities, config=self.config),
            cold_utilities=_build_heat_utilities(
                self.cold_utilities,
                config=self.config,
            ),
            pinch_temp=_build_temp_pinch(
                self.cold_pinch,
                self.hot_pinch,
                config=self.config,
            ),
            exergy_sources=coerce_output_value(
                self.exergy_sources,
                metric_name="exergy_sources",
                config=self.config,
            ),
            exergy_sinks=coerce_output_value(
                self.exergy_sinks,
                metric_name="exergy_sinks",
                config=self.config,
            ),
            ETE=coerce_output_value(
                self.ETE,
                metric_name="ETE",
                config=self.config,
            ),
            exergy_req_min=coerce_output_value(
                self.exergy_req_min,
                metric_name="exergy_req_min",
                config=self.config,
            ),
            exergy_des_min=coerce_output_value(
                self.exergy_des_min,
                metric_name="exergy_des_min",
                config=self.config,
            ),
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
    work_target: Optional[float] = None
    turbine_efficiency_target: Optional[float] = None

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Return the reporting payload including DI-only cost and work fields."""
        base = self._base_target_results(isTotal=isTotal)
        return base.model_copy(
            update={
                "work_target": coerce_output_value(
                    self.work_target,
                    metric_name="work_target",
                    config=self.config,
                ),
                "turbine_efficiency_target": coerce_output_value(
                    self.turbine_efficiency_target,
                    metric_name="turbine_efficiency_target",
                    config=self.config,
                ),
                "area": coerce_output_value(
                    self.area,
                    metric_name="area",
                    config=self.config,
                ),
                "num_units": self.num_units,
                "capital_cost": coerce_output_value(
                    self.capital_cost,
                    metric_name="capital_cost",
                    config=self.config,
                ),
                "total_cost": coerce_output_value(
                    self.total_cost,
                    metric_name="total_cost",
                    config=self.config,
                ),
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
                "work_target": coerce_output_value(
                    self.work_target,
                    metric_name="work_target",
                    config=self.config,
                ),
                "turbine_efficiency_target": coerce_output_value(
                    self.turbine_efficiency_target,
                    metric_name="turbine_efficiency_target",
                    config=self.config,
                ),
            }
        )


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
    hpr_success: bool
    hpr_hot_streams: StreamCollection
    hpr_cold_streams: StreamCollection
    hpr_details: Any

    def to_target_results(self, isTotal: bool = False) -> TargetResults:
        """Return the reporting payload for explicit HPR target results."""
        base = self._base_target_results(isTotal=isTotal)
        return base.model_copy(
            update={
                "work_target": coerce_output_value(
                    self.work_target,
                    metric_name="work_target",
                    config=self.config,
                ),
                "turbine_efficiency_target": coerce_output_value(
                    self.turbine_efficiency_target,
                    metric_name="turbine_efficiency_target",
                    config=self.config,
                ),
                "hpr_cycle": self.hpr_cycle,
                "hpr_utility_total": coerce_output_value(
                    self.hpr_utility_total,
                    metric_name="hpr_utility_total",
                    config=self.config,
                ),
                "hpr_work": coerce_output_value(
                    self.hpr_work,
                    metric_name="hpr_work",
                    config=self.config,
                ),
                "hpr_external_utility": coerce_output_value(
                    self.hpr_external_utility,
                    metric_name="hpr_external_utility",
                    config=self.config,
                ),
                "hpr_ambient_hot": coerce_output_value(
                    self.hpr_ambient_hot,
                    metric_name="hpr_ambient_hot",
                    config=self.config,
                ),
                "hpr_ambient_cold": coerce_output_value(
                    self.hpr_ambient_cold,
                    metric_name="hpr_ambient_cold",
                    config=self.config,
                ),
                "hpr_cop": coerce_output_value(
                    self.hpr_cop,
                    metric_name="hpr_cop",
                    config=self.config,
                ),
                "hpr_eta_he": coerce_output_value(
                    self.hpr_eta_he,
                    metric_name="hpr_eta_he",
                    config=self.config,
                ),
                "hpr_success": self.hpr_success,
                "hpr_hot_streams": self.hpr_hot_streams,
                "hpr_cold_streams": self.hpr_cold_streams,
            }
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

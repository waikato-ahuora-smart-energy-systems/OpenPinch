"""Design-view data of a HEN synthesis result, for clients that draw it.

Duties and temperatures are in the units the ``HeatExchangerNetwork`` carries.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

GridMatchKind = Literal["recovery", "hot_utility", "cold_utility"]


class _ViewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeatExchangerNetworkGridMatch(_ViewModel):
    """One active exchanger placed on the grid, in one period."""

    kind: GridMatchKind
    exchanger_id: str
    source_stream: str
    sink_stream: str
    stage: int | None = None
    duty: float
    source_inlet_temperature: float | None = None
    source_outlet_temperature: float | None = None
    sink_inlet_temperature: float | None = None
    sink_outlet_temperature: float | None = None


class HeatExchangerNetworkGridBranchCount(_ViewModel):
    """The number of parallel branches of a stream in a stage."""

    stream: str
    stage: int
    count: int


class HeatExchangerNetworkGridLayout(_ViewModel):
    """The grid diagram's topology: stream rows, stage columns and matches.

    Matches list recovery exchangers first, then heaters, then coolers. Drawing
    positions are left to the client.
    """

    period_id: str
    hot_streams: tuple[str, ...]
    cold_streams: tuple[str, ...]
    stages: tuple[int, ...]
    matches: tuple[HeatExchangerNetworkGridMatch, ...]
    branch_counts: tuple[HeatExchangerNetworkGridBranchCount, ...] = ()


class HeatExchangerNetworkRankingRow(_ViewModel):
    """One network of the duplicate-free ranking, as ``design.top(n)`` lists it."""

    rank: int
    method: str
    status: str
    objective_value: float | None = None
    total_annual_cost: float | None = None
    unit_count: int
    total_heat_recovery: float
    total_hot_utility: float
    total_cold_utility: float


class HeatExchangerNetworkUtilityDuty(_ViewModel):
    """One utility's active duty in a network, as ``design.utility(name)``."""

    name: str
    side: Literal["hot", "cold"]
    duty: float


class HeatExchangerNetworkView(_ViewModel):
    """The design view of one ranked network: what a results screen shows.

    ``ranking`` is empty when the result has no ranked outcomes; the totals
    then describe the result's own network.
    """

    rank: int
    period_id: str | None = None
    ranking: tuple[HeatExchangerNetworkRankingRow, ...]
    total_heat_recovery: float
    total_hot_utility: float
    total_cold_utility: float
    unit_count: int
    utility_duties: tuple[HeatExchangerNetworkUtilityDuty, ...]
    grid: HeatExchangerNetworkGridLayout


__all__ = [
    "GridMatchKind",
    "HeatExchangerNetworkRankingRow",
    "HeatExchangerNetworkUtilityDuty",
    "HeatExchangerNetworkView",
    "HeatExchangerNetworkGridBranchCount",
    "HeatExchangerNetworkGridLayout",
    "HeatExchangerNetworkGridMatch",
]

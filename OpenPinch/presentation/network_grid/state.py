"""Data models for heat exchanger network grid diagrams."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain._heat_exchanger.period_state import HeatExchangerPeriodState
from ...domain.heat_exchanger import HeatExchanger
from ...domain.heat_exchanger_network import HeatExchangerNetwork


@dataclass(frozen=True)
class GridDiagramMatch:
    """One active exchanger match placed in the process-stream grid."""

    exchanger: HeatExchanger
    state: HeatExchangerPeriodState
    source_stream: str
    sink_stream: str
    stage: int | None
    duty: float


@dataclass(frozen=True)
class HeatExchangerNetworkGridModel:
    """Normalized topology for a heat exchanger network grid diagram."""

    network: HeatExchangerNetwork
    period_id: str
    hot_streams: tuple[str, ...]
    cold_streams: tuple[str, ...]
    stages: tuple[int, ...]
    recovery_matches: tuple[GridDiagramMatch, ...]
    hot_utility_matches: tuple[GridDiagramMatch, ...]
    cold_utility_matches: tuple[GridDiagramMatch, ...]
    branch_counts: dict[tuple[str, int], int]


__all__ = [
    "GridDiagramMatch",
    "HeatExchangerNetworkGridModel",
]

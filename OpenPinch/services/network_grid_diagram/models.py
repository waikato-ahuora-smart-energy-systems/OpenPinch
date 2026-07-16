"""Data models for heat exchanger network grid diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...classes.heat_exchanger import HeatExchanger, HeatExchangerPeriodState
from ...classes.heat_exchanger_network import HeatExchangerNetwork


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


@dataclass
class HeatExchangerNetworkGridDiagram:
    """Rendered heat exchanger network grid diagram."""

    fig: Any
    ax: Any
    network: HeatExchangerNetwork
    grid_model: HeatExchangerNetworkGridModel

    def show(self) -> None:
        """Display the Plotly figure."""
        self.fig.show()

    def save(self, path: str | Path = "grid_diagram.png") -> None:
        """Save the Plotly figure to ``path``."""
        path = Path(path)
        if path.suffix.lower() == ".html":
            self.fig.write_html(path)
            return
        self.fig.write_image(path)


__all__ = [
    "GridDiagramMatch",
    "HeatExchangerNetworkGridDiagram",
    "HeatExchangerNetworkGridModel",
]

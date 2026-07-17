"""Grid-diagram presentation for ranked synthesis results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...analysis.heat_exchanger_networks.results.selection import network_for_rank
from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .state import HeatExchangerNetworkGridModel


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
        destination = Path(path)
        if destination.suffix.lower() == ".html":
            self.fig.write_html(destination)
            return
        self.fig.write_image(destination)


def build_result_grid_diagram(
    result: HeatExchangerNetworkSynthesisResult,
    solution_rank: int = 1,
    *,
    period_id: str | None = None,
    temperature_scaled: bool = False,
):
    """Return a grid diagram for one ranked synthesis network."""
    from .service import build_grid_diagram

    network = network_for_rank(result, solution_rank)
    return build_grid_diagram(
        network,
        period_id=network.resolve_period_id(period_id),
        temperature_scaled=temperature_scaled,
    )


__all__ = ["HeatExchangerNetworkGridDiagram", "build_result_grid_diagram"]

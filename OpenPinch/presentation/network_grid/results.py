"""Grid-diagram presentation for ranked synthesis results."""

from __future__ import annotations

from ...analysis.heat_exchanger_networks.results import network_for_rank
from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...services.network_grid_diagram import service as _grid_service


def build_result_grid_diagram(
    result: HeatExchangerNetworkSynthesisResult,
    solution_rank: int = 1,
    *,
    period_id: str | None = None,
    stream_line_width: float = 5.0,
    temperature_scaled: bool = False,
):
    """Return a grid diagram for one ranked synthesis network."""
    network = network_for_rank(result, solution_rank)
    return _grid_service.build_grid_diagram(
        network,
        period_id=network.resolve_period_id(period_id),
        stream_line_width=stream_line_width,
        temperature_scaled=temperature_scaled,
    )


__all__ = ["build_result_grid_diagram"]

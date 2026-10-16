"""The grid diagram's topology as plain data, without a plotting backend."""

from __future__ import annotations

from ...contracts.synthesis.view import (
    HeatExchangerNetworkGridBranchCount,
    HeatExchangerNetworkGridLayout,
    HeatExchangerNetworkGridMatch,
)
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .geometry import build_grid_model
from .state import GridDiagramMatch


def build_grid_layout(
    network: HeatExchangerNetwork,
    *,
    period_id: str | None = None,
) -> HeatExchangerNetworkGridLayout:
    """Return the grid model of ``network`` in one period as a contract."""
    model = build_grid_model(network, period_id=period_id)
    return HeatExchangerNetworkGridLayout(
        period_id=model.period_id,
        hot_streams=model.hot_streams,
        cold_streams=model.cold_streams,
        stages=model.stages,
        matches=tuple(
            _match(kind, match)
            for kind, matches in (
                ("recovery", model.recovery_matches),
                ("hot_utility", model.hot_utility_matches),
                ("cold_utility", model.cold_utility_matches),
            )
            for match in matches
        ),
        branch_counts=tuple(
            HeatExchangerNetworkGridBranchCount(stream=stream, stage=stage, count=count)
            for (stream, stage), count in model.branch_counts.items()
        ),
    )


def _match(kind: str, match: GridDiagramMatch) -> HeatExchangerNetworkGridMatch:
    state = match.state
    return HeatExchangerNetworkGridMatch(
        kind=kind,
        exchanger_id=match.exchanger.exchanger_id,
        source_stream=match.source_stream,
        sink_stream=match.sink_stream,
        stage=match.stage,
        duty=float(match.duty),
        source_inlet_temperature=state.source_inlet_temperature,
        source_outlet_temperature=state.source_outlet_temperature,
        sink_inlet_temperature=state.sink_inlet_temperature,
        sink_outlet_temperature=state.sink_outlet_temperature,
    )


__all__ = ["build_grid_layout"]

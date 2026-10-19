"""The grid diagram's topology as plain data, without a plotting backend."""

from __future__ import annotations

from ...contracts.synthesis.view import (
    HeatExchangerNetworkGridBranchCount,
    HeatExchangerNetworkGridLayout,
    HeatExchangerNetworkGridMatch,
    HeatExchangerNetworkGridStream,
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
    matches = tuple(
        _match(kind, match)
        for kind, kind_matches in (
            ("recovery", model.recovery_matches),
            ("hot_utility", model.hot_utility_matches),
            ("cold_utility", model.cold_utility_matches),
        )
        for match in kind_matches
    )
    return HeatExchangerNetworkGridLayout(
        period_id=model.period_id,
        hot_streams=model.hot_streams,
        cold_streams=model.cold_streams,
        stages=model.stages,
        matches=matches,
        branch_counts=tuple(
            HeatExchangerNetworkGridBranchCount(stream=stream, stage=stage, count=count)
            for (stream, stage), count in model.branch_counts.items()
        ),
        stream_temperatures=(
            *(_stream(stream, "hot", matches) for stream in model.hot_streams),
            *(_stream(stream, "cold", matches) for stream in model.cold_streams),
        ),
    )


def _stream(
    stream: str,
    side: str,
    matches: tuple[HeatExchangerNetworkGridMatch, ...],
) -> HeatExchangerNetworkGridStream:
    """A stream's supply and target: the ends of its exchangers' temperatures."""
    end = "source" if side == "hot" else "sink"

    def temperatures(which: str) -> list[float]:
        values = (
            getattr(match, f"{end}_{which}_temperature")
            for match in matches
            if getattr(match, f"{end}_stream") == stream
        )
        return [value for value in values if value is not None]

    inlets, outlets = temperatures("inlet"), temperatures("outlet")
    if side == "hot":
        supply, target = max(inlets, default=None), min(outlets, default=None)
    else:
        supply, target = min(inlets, default=None), max(outlets, default=None)
    return HeatExchangerNetworkGridStream(
        stream=stream,
        side=side,
        supply_temperature=supply,
        target_temperature=target,
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
        area=match.area,
    )


__all__ = ["build_grid_layout"]

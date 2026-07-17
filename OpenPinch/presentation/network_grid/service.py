"""Service entrypoint for heat exchanger network grid diagrams."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .dependencies import require_network_grid_diagram_dependency
from .geometry import build_grid_model
from .rendering import _PlotlyGridRenderer
from .results import HeatExchangerNetworkGridDiagram


def build_grid_diagram(
    networks: HeatExchangerNetwork | Sequence[HeatExchangerNetwork],
    *,
    index: int | None = None,
    period_id: str | None = None,
    temperature_scaled: bool = False,
) -> HeatExchangerNetworkGridDiagram | tuple[HeatExchangerNetworkGridDiagram, ...]:
    """Return OpenHENS-style grid diagrams for one or more networks."""
    if isinstance(networks, HeatExchangerNetwork):
        if index is not None and index != 0:
            raise IndexError("index is 0-based and must be 0 for a single network")
        return _build_single_grid_diagram(
            networks,
            period_id=period_id,
            temperature_scaled=temperature_scaled,
        )
    if not isinstance(networks, Sequence) or isinstance(
        networks,
        (str, bytes, bytearray),
    ):
        raise TypeError(
            "networks must be a HeatExchangerNetwork or a sequence of "
            "HeatExchangerNetwork objects"
        )

    selected_networks = tuple(networks)
    if not selected_networks:
        raise ValueError("at least one HeatExchangerNetwork is required")
    _validate_network_sequence(selected_networks)
    if index is None:
        return tuple(
            _build_single_grid_diagram(
                network,
                period_id=period_id,
                temperature_scaled=temperature_scaled,
            )
            for network in selected_networks
        )
    if index < 0:
        raise IndexError("index is 0-based and must be at least 0")
    if index >= len(selected_networks):
        raise IndexError(
            f"index {index} is unavailable; only {len(selected_networks)} "
            "network(s) are available"
        )
    return _build_single_grid_diagram(
        selected_networks[index],
        period_id=period_id,
        temperature_scaled=temperature_scaled,
    )


def _validate_network_sequence(networks: Sequence[Any]) -> None:
    for position, network in enumerate(networks):
        if not isinstance(network, HeatExchangerNetwork):
            raise TypeError(
                "networks must contain only HeatExchangerNetwork objects; "
                f"item {position} is {type(network).__name__}"
            )


def _build_single_grid_diagram(
    network: HeatExchangerNetwork,
    *,
    period_id: str | None,
    temperature_scaled: bool,
) -> HeatExchangerNetworkGridDiagram:
    grid_model = build_grid_model(network, period_id=period_id)
    plotly_go = require_network_grid_diagram_dependency(
        "plotly.graph_objects",
        package="plotly",
        purpose="heat exchanger network grid diagrams",
    )
    renderer = _PlotlyGridRenderer(
        grid_model,
        graph_objects=plotly_go,
        temperature_scaled=temperature_scaled,
    )
    return HeatExchangerNetworkGridDiagram(
        fig=renderer.fig,
        ax=renderer.ax,
        network=network,
        grid_model=grid_model,
    )


__all__ = ["build_grid_diagram"]

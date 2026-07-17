"""Owner-level orchestration for deterministic graph-data construction."""

from __future__ import annotations

from collections.abc import Iterable

from ...domain.enums import GT, ArrowHead, StreamLoc
from ...domain.targets import BaseTargetModel
from ...domain.zone import Zone
from .composite import _make_composite_graph
from .grand_composite import _make_gcc_graph
from .primitives import _create_curve, _streamloc_colour
from .specifications import GRAPH_BUILD_SPECS, _GraphBuildSpec


def get_output_graph_data(
    zone: Zone,
    graph_sets: dict | None = None,
) -> dict:
    """Return serialized graph sets for one zone hierarchy."""
    graph_sets = {} if graph_sets is None else graph_sets
    for target in zone.targets.values():
        graph_sets[target.name] = _create_graph_set(target, zone=zone)
    for subzone in zone.subzones.values():
        get_output_graph_data(subzone, graph_sets)
    return graph_sets


def _create_graph_set(
    target: BaseTargetModel,
    zone: Zone | None = None,
) -> dict:
    """Build every available graph for one target and its zone context."""
    context = resolve_graph_context(target, zone)
    target_graphs = getattr(target, "graphs", {})
    graphs = build_available_graphs(context["graph_title"], target_graphs)
    return build_graph_set_payload(target, context, graphs)


def resolve_graph_context(
    target: BaseTargetModel,
    zone: Zone | None = None,
) -> dict:
    """Return metadata shared by every graph-set payload."""
    return {
        "graph_title": target.name,
        "zone_name": getattr(zone, "name", None) or getattr(target, "zone_name", None),
        "zone_address": getattr(zone, "address", None),
    }


def iter_available_graph_specs(
    target_graphs: dict,
) -> Iterable[_GraphBuildSpec]:
    """Yield available graph specifications in canonical order."""
    return (
        spec for spec in GRAPH_BUILD_SPECS if spec.graph_type.value in target_graphs
    )


def build_available_graphs(
    graph_title: str,
    target_graphs: dict,
) -> list[dict]:
    """Build every graph available on one target."""
    return [
        build_graph_from_spec(graph_title, target_graphs, spec)
        for spec in iter_available_graph_specs(target_graphs)
    ]


def build_graph_from_spec(
    graph_title: str,
    target_graphs: dict,
    spec: _GraphBuildSpec,
) -> dict:
    """Dispatch one declarative graph specification to its builder."""
    graph_key = spec.graph_type.value
    if spec.builder == "composite":
        return _make_composite_graph(
            graph_title=graph_title,
            key=graph_key,
            data=target_graphs[graph_key],
            label=spec.label,
            value_field=spec.value_fields,
            stream_type=spec.stream_types,
            include_arrows=spec.include_arrows,
        )
    if spec.builder == "gcc":
        return _make_gcc_graph(
            graph_title=graph_title,
            key=graph_key,
            data=target_graphs[graph_key],
            label=spec.label,
            value_field=spec.value_fields,
            is_utility_profile=spec.utility_profile_flags,
        )
    if spec.builder == "energy_transfer":
        return _make_energy_transfer_diagram_graph(graph_title, target_graphs)
    raise ValueError(f"Unsupported graph builder: {spec.builder!r}.")


def build_graph_set_payload(
    target: BaseTargetModel,
    context: dict,
    graphs: list[dict],
) -> dict:
    """Wrap built graphs in target and zone metadata."""
    return {
        "name": context["graph_title"],
        "target_type": getattr(target, "type", None),
        "period_id": getattr(target, "period_id", None),
        "zone_name": context["zone_name"],
        "zone_address": context["zone_address"],
        "graphs": graphs,
    }


def _make_energy_transfer_diagram_graph(
    graph_title: str,
    target_graphs: dict,
) -> dict:
    diagram = target_graphs[GT.ETD.value]
    temperatures = diagram.get("temperatures", [])
    segments = [
        _create_curve(
            title=str(operation.get("name", "Operation")),
            colour=_streamloc_colour(StreamLoc.Unassigned),
            x_vals=operation.get("stacked_heat", []),
            y_vals=temperatures,
            arrow=ArrowHead.NO_ARROW.value,
            series_label=str(operation.get("name", "Operation")),
            series_id=f"{GT.ETD.value}:{operation.get('name', 'Operation')}",
            series_description=f"{operation.get('mode', 'R')} cascade",
        )
        for operation in diagram.get("operations", [])
    ]
    return {
        "type": GT.ETD.value,
        "name": f"Energy Transfer Diagram: {graph_title}",
        "segments": segments,
    }


__all__ = ["get_output_graph_data"]

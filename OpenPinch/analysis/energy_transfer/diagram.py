"""Energy-transfer diagram and surplus/deficit payload construction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ...domain.configuration import tol
from ...domain.enums import GraphType, ProblemTableLabel, TargetType
from ...domain.problem_table import ProblemTable
from ...domain.targets import EnergyTransferTarget, UtilitySummaryTarget
from .cascade import (
    characterise_cascades,
    compile_temperature_intervals,
    has_problem_table_values,
    simplify_constant_cp_intervals,
    stack_cascades,
    stack_order,
    transpose_operation_cascades,
)
from .serialization import (
    _clean_array,
    _clean_optional,
    _clean_value,
    _save_graph_data,
)


def compute_energy_transfer_target(
    base_target: UtilitySummaryTarget,
    source_targets: Iterable[UtilitySummaryTarget] | None = None,
) -> EnergyTransferTarget:
    """Create a typed energy-transfer target from operation cascades."""
    sources = list(source_targets or [base_target])
    diagram = create_energy_transfer_diagram(sources, base_target=base_target)
    table = create_heat_surplus_deficit_table(diagram)
    return EnergyTransferTarget.model_validate(
        {
            "zone_name": getattr(base_target, "zone_name", None),
            "scope": base_target.scope,
            "zone_type": base_target.zone_type,
            "type": TargetType.ET.value,
            "parent_zone": base_target.parent_zone,
            "config": base_target.config,
            "pt": get_base_problem_table(base_target),
            "graphs": _save_graph_data(diagram),
            "hot_utilities": base_target.hot_utilities,
            "cold_utilities": base_target.cold_utilities,
            "hot_utility_target": base_target.hot_utility_target,
            "cold_utility_target": base_target.cold_utility_target,
            "heat_recovery_target": base_target.heat_recovery_target,
            "heat_recovery_limit": base_target.heat_recovery_limit,
            "degree_of_int": base_target.degree_of_int,
            "utility_cost": base_target.utility_cost,
            "hot_pinch": base_target.hot_pinch,
            "cold_pinch": base_target.cold_pinch,
            "period_id": base_target.period_id,
            "period_idx": base_target.period_idx,
            "base_target_type": base_target.type,
            "base_target_name": base_target.name,
            "heat_surplus_deficit_table": table,
            "energy_transfer_diagram": diagram,
        }
    )


def create_energy_transfer_diagram(
    source_targets: Iterable[UtilitySummaryTarget] | ProblemTable,
    *,
    base_target: UtilitySummaryTarget | None = None,
    simplify: bool = True,
) -> dict[str, Any]:
    """Build operation curves on one shared temperature grid."""
    source_records = normalise_source_targets(source_targets)
    if not source_records:
        return empty_diagram(base_target)
    base_table = (
        get_base_problem_table(base_target) if base_target is not None else None
    )
    temperatures = compile_temperature_intervals(source_records, base_table)
    names, modes, interval_heat, cascades = transpose_operation_cascades(
        source_records,
        temperatures,
    )
    if simplify and interval_heat.shape[1] > 1:
        temperatures, interval_heat, cascades = simplify_constant_cp_intervals(
            temperatures,
            interval_heat,
            cascades[:, 0],
        )
    hot_pinch, cold_pinch = pinch_temperatures(base_target, base_table)
    headers = characterise_cascades(
        names=names,
        modes=modes,
        temperatures=temperatures,
        interval_heat=interval_heat,
        cascades=cascades,
        hot_pinch=hot_pinch,
        cold_pinch=cold_pinch,
    )
    order = stack_order(headers)
    stacked = stack_cascades(cascades[order])
    operations = []
    for stack_index, source_index in enumerate(order):
        header = headers[source_index]
        operations.append(
            {
                "name": names[source_index],
                "mode": modes[source_index],
                "interval_heat": _clean_array(interval_heat[source_index]).tolist(),
                "cascade_heat": _clean_array(cascades[source_index]).tolist(),
                "stacked_heat": _clean_array(stacked[stack_index]).tolist(),
                "weighted_area": _clean_value(header["weighted_area"]),
                "total_area": _clean_value(header["total_area"]),
                "max_heat": _clean_value(header["max_heat"]),
                "sort_key": _clean_value(header["sort_key"]),
                "cross_pinch": bool(header["cross_pinch"]),
            }
        )
    return {
        "temperatures": _clean_array(temperatures).tolist(),
        "operations": operations,
        "base_target_name": getattr(base_target, "name", None),
        "base_target_type": getattr(base_target, "type", None),
        "hot_pinch": _clean_optional(hot_pinch),
        "cold_pinch": _clean_optional(cold_pinch),
    }


def create_heat_surplus_deficit_table(
    diagram: dict[str, Any] | Iterable[UtilitySummaryTarget] | ProblemTable,
) -> list[dict[str, Any]]:
    """Return interval surplus/deficit rows from diagram data."""
    if not isinstance(diagram, dict):
        diagram = create_energy_transfer_diagram(diagram)
    temperatures = np.asarray(diagram.get("temperatures", []), dtype=float)
    operations = list(diagram.get("operations", []))
    if temperatures.size == 0:
        return []
    interval_matrix = np.asarray(
        [operation.get("interval_heat", []) for operation in operations],
        dtype=float,
    )
    if interval_matrix.size == 0:
        interval_matrix = np.zeros((0, max(temperatures.size - 1, 0)))
    rows = []
    names = [str(operation["name"]) for operation in operations]
    for index, temperature in enumerate(temperatures):
        values = (
            np.zeros(len(names), dtype=float)
            if index == 0
            else interval_matrix[:, index - 1]
        )
        row: dict[str, Any] = {
            "temperature": _clean_value(temperature),
            "interval": float(index),
            **{name: _clean_value(value) for name, value in zip(names, values)},
        }
        total = float(np.sum(values)) if values.size else 0.0
        row.update(
            {
                "total_delta_hnet": _clean_value(total),
                "heat_surplus": _clean_value(max(-total, 0.0)),
                "heat_deficit": _clean_value(max(total, 0.0)),
            }
        )
        rows.append(row)
    return rows


def normalise_source_targets(
    source_targets: Iterable[UtilitySummaryTarget] | ProblemTable,
) -> list[dict[str, Any]]:
    if isinstance(source_targets, ProblemTable):
        return [
            {
                "name": "Process",
                "mode": "R",
                "pt": source_targets,
                "hot_utility_target": 0.0,
                "cold_utility_target": 0.0,
            }
        ]
    records = []
    for index, source in enumerate(source_targets):
        name_override = None
        target = source
        if isinstance(source, dict):
            target = source.get("target")
            name_override = source.get("name")
        records.append(
            {
                "name": str(name_override)
                if name_override
                else operation_name(target, index),
                "mode": operation_mode(target),
                "pt": get_source_problem_table(target),
                "hot_utility_target": getattr(target, "hot_utility_target", 0.0),
                "cold_utility_target": getattr(target, "cold_utility_target", 0.0),
            }
        )
    return records


def empty_diagram(
    base_target: UtilitySummaryTarget | None = None,
) -> dict[str, Any]:
    return {
        "temperatures": [],
        "operations": [],
        "base_target_name": getattr(base_target, "name", None),
        "base_target_type": getattr(base_target, "type", None),
        "hot_pinch": _clean_optional(getattr(base_target, "hot_pinch", None)),
        "cold_pinch": _clean_optional(getattr(base_target, "cold_pinch", None)),
    }


def pinch_temperatures(
    base_target: UtilitySummaryTarget | None,
    base_table: ProblemTable | None,
) -> tuple[float | None, float | None]:
    hot_pinch = getattr(base_target, "hot_pinch", None)
    cold_pinch = getattr(base_target, "cold_pinch", None)
    if hot_pinch is not None or cold_pinch is not None:
        return hot_pinch, cold_pinch
    return (None, None) if base_table is None else base_table.pinch_temperatures()


def operation_name(target: UtilitySummaryTarget, index: int) -> str:
    name = getattr(target, "zone_name", None) or getattr(target, "name", None)
    if not name:
        return f"Operation {index + 1}"
    name = str(name)
    suffix = f"/{getattr(target, 'type', '')}"
    return name[: -len(suffix)] if suffix != "/" and name.endswith(suffix) else name


def operation_mode(target: UtilitySummaryTarget) -> str:
    if abs(float(getattr(target, "hot_utility_target", 0.0) or 0.0)) > tol:
        return "H"
    if abs(float(getattr(target, "cold_utility_target", 0.0) or 0.0)) > tol:
        return "C"
    return "R"


def get_base_problem_table(
    base_target: UtilitySummaryTarget | None,
) -> ProblemTable:
    table = getattr(base_target, "pt", None)
    if not isinstance(table, ProblemTable):
        raise TypeError(
            "Energy transfer analysis requires a base target with a ProblemTable."
        )
    return table


def get_source_problem_table(target: UtilitySummaryTarget) -> ProblemTable:
    graphs = getattr(target, "graphs", None)
    if isinstance(graphs, dict):
        gcc = graphs.get(GraphType.GCC.value)
        if (
            isinstance(gcc, ProblemTable)
            and has_problem_table_values(gcc, ProblemTableLabel.T)
            and has_problem_table_values(gcc, ProblemTableLabel.H_NET)
        ):
            return gcc
    return get_base_problem_table(target)


__all__ = [
    "compute_energy_transfer_target",
    "create_energy_transfer_diagram",
    "create_heat_surplus_deficit_table",
]

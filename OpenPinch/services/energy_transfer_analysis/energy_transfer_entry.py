"""Energy-transfer diagram and heat-surplus/deficit table construction."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ...classes.problem_table import ProblemTable
from ...lib.config import tol
from ...lib.enums import GT, PT, TT
from ...lib.schemas.targets import EnergyTransferTarget, UtilitySummaryTarget
from ..common.service_orchestration import (
    apply_zone_config_overrides,
    format_selected_period_suffix,
    record_selected_period,
    target_matches_requested_period,
)

__all__ = [
    "compute_energy_transfer_target",
    "create_energy_transfer_diagram",
    "create_heat_surplus_deficit_table",
    "run_energy_transfer_analysis_service",
]

_MODE_ORDER = {"C": 0, "R": 1, "H": 2}
_WEIGHTING_TEMPERATURE = 99.0
_ENERGY_TRANSFER_TARGET_ORDER = (
    TT.TS.value,
    TT.DI.value,
)


def compute_energy_transfer_target(
    base_target: UtilitySummaryTarget,
    source_targets: Iterable[UtilitySummaryTarget] | None = None,
) -> EnergyTransferTarget:
    """Create an energy-transfer target from operation-level cascades."""
    sources = list(source_targets or [base_target])
    diagram = create_energy_transfer_diagram(sources, base_target=base_target)
    table = create_heat_surplus_deficit_table(diagram)
    return EnergyTransferTarget.model_validate(
        {
            "zone_name": getattr(base_target, "zone_name", None),
            "type": TT.ET.value,
            "parent_zone": base_target.parent_zone,
            "config": base_target.config,
            "pt": _get_base_problem_table(base_target),
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
    """Build operation-level ETD curves on one shared temperature interval grid."""
    source_records = _normalise_source_targets(source_targets)
    if not source_records:
        return _empty_diagram(base_target=base_target)

    base_pt = _get_base_problem_table(base_target) if base_target is not None else None
    temperatures = _compile_temperature_intervals(source_records, base_pt)
    names, modes, interval_heat, cascades = _transpose_operation_cascades(
        source_records,
        temperatures,
    )

    if simplify and interval_heat.shape[1] > 1:
        temperatures, interval_heat, cascades = _simplify_constant_cp_intervals(
            temperatures,
            interval_heat,
            cascades[:, 0],
        )

    hot_pinch, cold_pinch = _pinch_temperatures(base_target, base_pt)
    headers = _characterise_cascades(
        names=names,
        modes=modes,
        temperatures=temperatures,
        interval_heat=interval_heat,
        cascades=cascades,
        hot_pinch=hot_pinch,
        cold_pinch=cold_pinch,
    )
    order = _stack_order(headers)
    stacked = _stack_cascades(cascades[order])

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
    """Return the ETD heat-surplus/deficit table from diagram data."""
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

    rows: list[dict[str, Any]] = []
    operation_names = [str(operation["name"]) for operation in operations]
    for idx, temperature in enumerate(temperatures):
        row: dict[str, Any] = {
            "temperature": _clean_value(temperature),
            "interval": float(idx),
        }
        if idx == 0:
            values = np.zeros(len(operation_names), dtype=float)
        else:
            values = interval_matrix[:, idx - 1]

        for name, value in zip(operation_names, values):
            row[name] = _clean_value(value)

        total = float(np.sum(values)) if values.size else 0.0
        row["total_delta_hnet"] = _clean_value(total)
        row["heat_surplus"] = _clean_value(max(-total, 0.0))
        row["heat_deficit"] = _clean_value(max(total, 0.0))
        rows.append(row)
    return rows


def run_energy_transfer_analysis_service(
    zone,
    args: dict | None = None,
    *,
    refresh_services: dict[str, Any],
    compute_func=compute_energy_transfer_target,
):
    """Create the energy-transfer diagram and heat-surplus/deficit table."""
    apply_zone_config_overrides(zone, args)
    runtime_args = dict(args or {})
    explicit_target_type = _normalize_energy_transfer_base_target_type(
        runtime_args.get("base_target_type")
    )
    idx, sid = record_selected_period(zone, runtime_args)
    runtime_args["period_idx"] = idx
    if sid is not None:
        runtime_args["period_id"] = sid
    compare_args = dict(args or {}) if isinstance(args, dict) else {}
    zone._selected_energy_transfer_base_target_type = None

    for target_type in _get_energy_transfer_candidate_order(
        zone,
        explicit_target_type,
    ):
        target = _ensure_energy_transfer_base_target(
            zone,
            target_type=target_type,
            refresh_args=runtime_args,
            compare_args=compare_args,
            refresh_services=refresh_services,
        )
        if target is None:
            if explicit_target_type is not None:
                raise RuntimeError(
                    "Energy transfer analysis could not produce base target "
                    f"{target_type!r} for zone {zone.name!r}"
                    f"{format_selected_period_suffix(runtime_args)}."
                )
            continue

        zone.add_target(
            compute_func(
                target,
                source_targets=_get_energy_transfer_source_targets(zone, target_type),
            )
        )
        zone._selected_energy_transfer_base_target_type = target_type
        return zone

    raise RuntimeError(
        "Energy transfer analysis could not find a compatible target for zone "
        f"{zone.name!r}{format_selected_period_suffix(runtime_args)} "
        f"using implicit order {' -> '.join(_ENERGY_TRANSFER_TARGET_ORDER)}."
    )


def _normalize_energy_transfer_base_target_type(
    base_target_type: object | None,
) -> str | None:
    """Validate an explicit energy-transfer base target override."""
    if base_target_type is None:
        return None

    normalized = str(base_target_type)
    if normalized not in _ENERGY_TRANSFER_TARGET_ORDER:
        supported = ", ".join(_ENERGY_TRANSFER_TARGET_ORDER)
        raise ValueError(
            "Unsupported energy-transfer base_target_type "
            f"{normalized!r}. Supported types: {supported}."
        )
    return normalized


def _get_energy_transfer_candidate_order(
    zone,
    base_target_type: str | None,
) -> tuple[str, ...]:
    """Return the energy-transfer base-target search order."""
    if base_target_type is not None:
        return (base_target_type,)
    if TT.TS.value in zone.targets or len(zone.subzones) > 0:
        return _ENERGY_TRANSFER_TARGET_ORDER
    return (TT.DI.value,)


def _ensure_energy_transfer_base_target(
    zone,
    *,
    target_type: str,
    refresh_args: dict | None,
    compare_args: dict | None,
    refresh_services: dict[str, Any],
):
    """Ensure an energy-transfer-compatible target exists for this state."""
    target = zone.targets.get(target_type)
    if target_matches_requested_period(
        target,
        args=compare_args,
        period_ids=getattr(zone, "period_ids", None),
    ):
        return target

    refresh_service = refresh_services.get(target_type)
    if refresh_service is None:
        return None

    if target_type == TT.TS.value:
        if len(zone.subzones) == 0:
            return None
        direct_target = zone.targets.get(TT.DI.value)
        if not target_matches_requested_period(
            direct_target,
            args=compare_args,
            period_ids=getattr(zone, "period_ids", None),
        ):
            refresh_services[TT.DI.value](zone, refresh_args)
        for subzone in zone.subzones.values():
            subtarget = subzone.targets.get(TT.DI.value)
            if not target_matches_requested_period(
                subtarget,
                args=compare_args,
                period_ids=getattr(subzone, "period_ids", None),
            ):
                refresh_services[TT.DI.value](subzone, refresh_args)

    refresh_service(zone, refresh_args)
    refreshed_target = zone.targets.get(target_type)
    if target_matches_requested_period(
        refreshed_target,
        args=compare_args,
        period_ids=getattr(zone, "period_ids", None),
    ):
        return refreshed_target
    return None


def _get_energy_transfer_source_targets(zone, base_target_type: str):
    """Return operation-level cascades that make up one ETD target."""
    if base_target_type == TT.TS.value and len(zone.subzones) > 0:
        return [
            {
                "name": source_name,
                "target": source_zone.targets[TT.DI.value],
            }
            for source_name, source_zone in _iter_energy_transfer_source_zones(zone)
            if TT.DI.value in source_zone.targets
        ]
    target = zone.targets.get(base_target_type)
    return [target] if target is not None else []


def _iter_energy_transfer_source_zones(zone):
    """Yield immediate child zones whose GCCs make up one Total Site ETD layer."""
    current_prefix = str(zone.name)
    for subzone in zone.subzones.values():
        source_name = f"{current_prefix}/{subzone.name}"
        yield source_name, subzone


def _normalise_source_targets(
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

    source_records = []
    for index, source in enumerate(source_targets):
        name_override = None
        target = source
        if isinstance(source, dict):
            target = source.get("target")
            name_override = source.get("name")

        pt = _get_source_problem_table(target)
        source_records.append(
            {
                "name": (
                    str(name_override)
                    if name_override
                    else _operation_name(target, index)
                ),
                "mode": _operation_mode(target),
                "pt": pt,
                "hot_utility_target": getattr(target, "hot_utility_target", 0.0),
                "cold_utility_target": getattr(target, "cold_utility_target", 0.0),
            }
        )
    return source_records


def _compile_temperature_intervals(
    source_records: list[dict[str, Any]],
    base_pt: ProblemTable | None,
) -> np.ndarray:
    arrays = [_as_float_array(record["pt"][PT.T]) for record in source_records]
    if base_pt is not None:
        arrays.append(_as_float_array(base_pt[PT.T]))
    non_empty_temperature_arrays = [array for array in arrays if array.size > 0]
    if not non_empty_temperature_arrays:
        return np.array([], dtype=float)
    temperatures = np.concatenate(non_empty_temperature_arrays)
    rounded = np.round(temperatures, _decimal_places())
    unique = np.unique(rounded)
    return np.sort(unique)[::-1]


def _transpose_operation_cascades(
    source_records: list[dict[str, Any]],
    temperatures: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    names = [str(record["name"]) for record in source_records]
    modes = [str(record["mode"]) for record in source_records]
    interval_count = max(len(temperatures) - 1, 0)
    interval_heat = np.zeros((len(source_records), interval_count), dtype=float)
    cascades = np.zeros((len(source_records), len(temperatures)), dtype=float)

    if interval_count == 0:
        return names, modes, interval_heat, cascades

    common_upper = temperatures[:-1]
    common_lower = temperatures[1:]
    dt = common_upper - common_lower

    for row, record in enumerate(source_records):
        pt = record["pt"]
        source_t = _as_float_array(pt[PT.T])
        h_net = _as_float_array(pt[PT.H_NET])
        cascades[row, 0] = h_net[0] if h_net.size else 0.0
        if source_t.size < 2 or h_net.size < 2:
            cascades[row] = cascades[row, 0]
            continue

        if not _has_problem_table_values(pt, PT.CP_NET):
            source_heat = _interp_descending_temperature(
                common_upper,
                source_t,
                h_net,
            )
            sink_heat = _interp_descending_temperature(
                common_lower,
                source_t,
                h_net,
            )
            interval_heat[row] = sink_heat - source_heat
            cascades[row, 1:] = cascades[row, 0] + np.cumsum(interval_heat[row])
            continue

        cp_net = _as_float_array(pt[PT.CP_NET])
        source_upper = source_t[:-1]
        source_lower = source_t[1:]
        active = (common_upper[:, np.newaxis] <= source_upper[np.newaxis, :] + tol) & (
            common_lower[:, np.newaxis] >= source_lower[np.newaxis, :] - tol
        )
        has_interval = np.any(active, axis=1)
        source_indices = np.argmax(active, axis=1)
        interval_cp = np.zeros(interval_count, dtype=float)
        cp_values = cp_net[1:] if cp_net.size > 1 else np.array([], dtype=float)
        if cp_values.size:
            interval_cp[has_interval] = cp_values[source_indices[has_interval]]
        interval_heat[row] = dt * interval_cp
        cascades[row, 1:] = cascades[row, 0] + np.cumsum(interval_heat[row])

    return names, modes, _clean_array(interval_heat), _clean_array(cascades)


def _simplify_constant_cp_intervals(
    temperatures: np.ndarray,
    interval_heat: np.ndarray,
    initial_heat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = temperatures[:-1] - temperatures[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cp = np.divide(
            interval_heat,
            dt[np.newaxis, :],
            out=np.zeros_like(interval_heat),
            where=np.abs(dt[np.newaxis, :]) > tol,
        )
    changes = np.any(np.abs(np.diff(cp, axis=1)) > tol, axis=0)
    breakpoints = np.r_[0, np.flatnonzero(changes) + 1, interval_heat.shape[1]]

    simplified_heat = np.stack(
        [
            np.sum(interval_heat[:, start:end], axis=1)
            for start, end in zip(breakpoints[:-1], breakpoints[1:])
        ],
        axis=1,
    )
    simplified_temperatures = np.r_[
        temperatures[0],
        temperatures[breakpoints[1:]],
    ]
    simplified_cascades = np.column_stack(
        [initial_heat, initial_heat[:, np.newaxis] + np.cumsum(simplified_heat, axis=1)]
    )
    return (
        _clean_array(simplified_temperatures),
        _clean_array(simplified_heat),
        _clean_array(simplified_cascades),
    )


def _characterise_cascades(
    *,
    names: list[str],
    modes: list[str],
    temperatures: np.ndarray,
    interval_heat: np.ndarray,
    cascades: np.ndarray,
    hot_pinch: float | None,
    cold_pinch: float | None,
) -> list[dict[str, Any]]:
    if interval_heat.shape[1] == 0:
        return [_empty_header(name=name, mode=mode) for name, mode in zip(names, modes)]

    dt = temperatures[:-1] - temperatures[1:]
    midpoints = 0.5 * (temperatures[:-1] + temperatures[1:])
    h_avg = 0.5 * (cascades[:, :-1] + cascades[:, 1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        th_area = np.abs(
            np.divide(
                h_avg,
                dt[np.newaxis, :],
                out=np.zeros_like(h_avg),
                where=np.abs(dt[np.newaxis, :]) > tol,
            )
        )

    weights = _pinch_weights(midpoints, hot_pinch, cold_pinch)
    weighted_area = th_area @ weights
    total_area = np.sum(th_area, axis=1)
    max_heat = np.max(np.abs(cascades), axis=1)
    cross_pinch = _cross_pinch_flags(cascades, temperatures, hot_pinch, cold_pinch)
    sort_temperatures = _first_active_temperatures(interval_heat, temperatures, modes)
    sort_keys = np.asarray(
        [
            (1.0 / temp_k) if mode == "C" and abs(temp_k) > tol else temp_k
            for mode, temp_k in zip(modes, sort_temperatures)
        ],
        dtype=float,
    )

    return [
        {
            "name": name,
            "mode": mode,
            "weighted_area": weighted_area[idx],
            "total_area": total_area[idx],
            "max_heat": max_heat[idx],
            "cross_pinch": cross_pinch[idx],
            "sort_temperature": sort_temperatures[idx],
            "sort_key": sort_keys[idx],
        }
        for idx, (name, mode) in enumerate(zip(names, modes))
    ]


def _pinch_weights(
    midpoints: np.ndarray,
    hot_pinch: float | None,
    cold_pinch: float | None,
) -> np.ndarray:
    weights = np.ones_like(midpoints, dtype=float)
    if hot_pinch is None or cold_pinch is None:
        return weights
    above = midpoints > hot_pinch + tol
    below = midpoints < cold_pinch - tol
    weights[above] = 1.0 / (
        np.abs(midpoints[above] - hot_pinch) / _WEIGHTING_TEMPERATURE + 1.0
    )
    weights[below] = 1.0 / (
        np.abs(midpoints[below] - cold_pinch) / _WEIGHTING_TEMPERATURE + 1.0
    )
    return weights


def _cross_pinch_flags(
    cascades: np.ndarray,
    temperatures: np.ndarray,
    hot_pinch: float | None,
    cold_pinch: float | None,
) -> np.ndarray:
    if hot_pinch is None or cold_pinch is None or cascades.shape[1] < 2:
        return np.zeros(cascades.shape[0], dtype=bool)
    interval_bottom = temperatures[1:]
    pinch_band = (interval_bottom <= hot_pinch + tol) & (
        interval_bottom >= cold_pinch - tol
    )
    crosses = (np.abs(cascades[:, 1:]) > tol) & pinch_band[np.newaxis, :]
    return np.any(crosses, axis=1)


def _first_active_temperatures(
    interval_heat: np.ndarray,
    temperatures: np.ndarray,
    modes: list[str],
) -> np.ndarray:
    active = np.abs(interval_heat) > tol
    first_indices = np.argmax(active, axis=1)
    has_active = np.any(active, axis=1)
    fallback = temperatures[0] if temperatures.size else 0.0
    first_t = np.where(has_active, temperatures[first_indices], fallback) + 273.15
    return np.asarray(first_t, dtype=float)


def _stack_order(headers: list[dict[str, Any]]) -> np.ndarray:
    indexed = list(enumerate(headers))
    indexed.sort(
        key=lambda item: (
            _MODE_ORDER.get(str(item[1]["mode"]), len(_MODE_ORDER)),
            -float(item[1]["sort_key"]),
            str(item[1]["name"]),
        )
    )
    return np.asarray([idx for idx, _ in indexed], dtype=int)


def _stack_cascades(cascades: np.ndarray) -> np.ndarray:
    if cascades.size == 0:
        return cascades
    return np.cumsum(cascades, axis=0)


def _empty_diagram(base_target: UtilitySummaryTarget | None = None) -> dict[str, Any]:
    return {
        "temperatures": [],
        "operations": [],
        "base_target_name": getattr(base_target, "name", None),
        "base_target_type": getattr(base_target, "type", None),
        "hot_pinch": _clean_optional(getattr(base_target, "hot_pinch", None)),
        "cold_pinch": _clean_optional(getattr(base_target, "cold_pinch", None)),
    }


def _empty_header(name: str, mode: str) -> dict[str, Any]:
    return {
        "name": name,
        "mode": mode,
        "weighted_area": 0.0,
        "total_area": 0.0,
        "max_heat": 0.0,
        "cross_pinch": False,
        "sort_temperature": 0.0,
        "sort_key": 0.0,
    }


def _pinch_temperatures(
    base_target: UtilitySummaryTarget | None,
    base_pt: ProblemTable | None,
) -> tuple[float | None, float | None]:
    hot_pinch = getattr(base_target, "hot_pinch", None)
    cold_pinch = getattr(base_target, "cold_pinch", None)
    if hot_pinch is not None or cold_pinch is not None:
        return hot_pinch, cold_pinch
    if base_pt is None:
        return None, None
    return base_pt.pinch_temperatures()


def _operation_name(target: UtilitySummaryTarget, index: int) -> str:
    name = getattr(target, "zone_name", None) or getattr(target, "name", None)
    if not name:
        return f"Operation {index + 1}"
    name = str(name)
    suffix = f"/{getattr(target, 'type', '')}"
    if suffix != "/" and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _operation_mode(target: UtilitySummaryTarget) -> str:
    if abs(float(getattr(target, "hot_utility_target", 0.0) or 0.0)) > tol:
        return "H"
    if abs(float(getattr(target, "cold_utility_target", 0.0) or 0.0)) > tol:
        return "C"
    return "R"


def _get_base_problem_table(base_target: UtilitySummaryTarget | None) -> ProblemTable:
    pt = getattr(base_target, "pt", None)
    if not isinstance(pt, ProblemTable):
        raise TypeError(
            "Energy transfer analysis requires a base target with a ProblemTable."
        )
    return pt


def _get_source_problem_table(target: UtilitySummaryTarget) -> ProblemTable:
    graphs = getattr(target, "graphs", None)
    if isinstance(graphs, dict):
        gcc = graphs.get(GT.GCC.value)
        if (
            isinstance(gcc, ProblemTable)
            and _has_problem_table_values(gcc, PT.T)
            and _has_problem_table_values(gcc, PT.H_NET)
        ):
            return gcc
    return _get_base_problem_table(target)


def _has_problem_table_values(pt: ProblemTable, column: PT) -> bool:
    try:
        values = _as_float_array(pt[column])
    except KeyError, TypeError, ValueError:
        return False
    return values.size > 0 and not bool(np.all(np.isnan(values)))


def _interp_descending_temperature(
    temperatures: np.ndarray,
    source_t: np.ndarray,
    source_heat: np.ndarray,
) -> np.ndarray:
    return np.interp(
        temperatures,
        source_t[::-1],
        source_heat[::-1],
        left=source_heat[-1],
        right=source_heat[0],
    )


def _save_graph_data(diagram: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {GT.ETD.value: diagram}


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _decimal_places() -> int:
    return int(-np.log10(tol))


def _clean_array(values: np.ndarray) -> np.ndarray:
    return np.where(np.abs(values) <= tol, 0.0, values)


def _clean_optional(value: float | None) -> float | None:
    return None if value is None else _clean_value(value)


def _clean_value(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) <= tol else value

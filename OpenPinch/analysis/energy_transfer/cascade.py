"""Temperature-grid cascade transformation and stacking."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.configuration import tol
from ...domain.enums import PT
from ...domain.problem_table import ProblemTable
from .serialization import _as_float_array, _clean_array, _decimal_places

MODE_ORDER = {"C": 0, "R": 1, "H": 2}
WEIGHTING_TEMPERATURE = 99.0


def compile_temperature_intervals(
    source_records: list[dict[str, Any]],
    base_table: ProblemTable | None,
) -> np.ndarray:
    """Return the descending union of all source temperature grids."""
    arrays = [_as_float_array(record["pt"][PT.T]) for record in source_records]
    if base_table is not None:
        arrays.append(_as_float_array(base_table[PT.T]))
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return np.array([], dtype=float)
    unique = np.unique(np.round(np.concatenate(arrays), _decimal_places()))
    return np.sort(unique)[::-1]


def transpose_operation_cascades(
    source_records: list[dict[str, Any]],
    temperatures: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Transpose operation cascades onto one shared interval grid."""
    names = [str(record["name"]) for record in source_records]
    modes = [str(record["mode"]) for record in source_records]
    interval_count = max(len(temperatures) - 1, 0)
    interval_heat = np.zeros((len(source_records), interval_count), dtype=float)
    cascades = np.zeros((len(source_records), len(temperatures)), dtype=float)
    if interval_count == 0:
        return names, modes, interval_heat, cascades

    common_upper = temperatures[:-1]
    common_lower = temperatures[1:]
    delta_t = common_upper - common_lower
    for row, record in enumerate(source_records):
        table = record["pt"]
        source_t = _as_float_array(table[PT.T])
        h_net = _as_float_array(table[PT.H_NET])
        cascades[row, 0] = h_net[0] if h_net.size else 0.0
        if source_t.size < 2 or h_net.size < 2:
            cascades[row] = cascades[row, 0]
            continue
        if not has_problem_table_values(table, PT.CP_NET):
            source_heat = interp_descending_temperature(common_upper, source_t, h_net)
            sink_heat = interp_descending_temperature(common_lower, source_t, h_net)
            interval_heat[row] = sink_heat - source_heat
            cascades[row, 1:] = cascades[row, 0] + np.cumsum(interval_heat[row])
            continue

        cp_net = _as_float_array(table[PT.CP_NET])
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
        interval_heat[row] = delta_t * interval_cp
        cascades[row, 1:] = cascades[row, 0] + np.cumsum(interval_heat[row])
    return names, modes, _clean_array(interval_heat), _clean_array(cascades)


def simplify_constant_cp_intervals(
    temperatures: np.ndarray,
    interval_heat: np.ndarray,
    initial_heat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge adjacent intervals with identical operation heat capacities."""
    delta_t = temperatures[:-1] - temperatures[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cp = np.divide(
            interval_heat,
            delta_t[np.newaxis, :],
            out=np.zeros_like(interval_heat),
            where=np.abs(delta_t[np.newaxis, :]) > tol,
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
    simplified_temperatures = np.r_[temperatures[0], temperatures[breakpoints[1:]]]
    simplified_cascades = np.column_stack(
        [
            initial_heat,
            initial_heat[:, np.newaxis] + np.cumsum(simplified_heat, axis=1),
        ]
    )
    return tuple(
        _clean_array(values)
        for values in (
            simplified_temperatures,
            simplified_heat,
            simplified_cascades,
        )
    )


def characterise_cascades(
    *,
    names: list[str],
    modes: list[str],
    temperatures: np.ndarray,
    interval_heat: np.ndarray,
    cascades: np.ndarray,
    hot_pinch: float | None,
    cold_pinch: float | None,
) -> list[dict[str, Any]]:
    """Build deterministic cascade metadata used for stack ordering."""
    if interval_heat.shape[1] == 0:
        return [empty_header(name, mode) for name, mode in zip(names, modes)]
    delta_t = temperatures[:-1] - temperatures[1:]
    midpoints = 0.5 * (temperatures[:-1] + temperatures[1:])
    average_heat = 0.5 * (cascades[:, :-1] + cascades[:, 1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        temperature_heat_area = np.abs(
            np.divide(
                average_heat,
                delta_t[np.newaxis, :],
                out=np.zeros_like(average_heat),
                where=np.abs(delta_t[np.newaxis, :]) > tol,
            )
        )
    weighted_area = temperature_heat_area @ pinch_weights(
        midpoints,
        hot_pinch,
        cold_pinch,
    )
    total_area = np.sum(temperature_heat_area, axis=1)
    max_heat = np.max(np.abs(cascades), axis=1)
    cross_pinch = cross_pinch_flags(
        cascades,
        temperatures,
        hot_pinch,
        cold_pinch,
    )
    sort_temperatures = first_active_temperatures(interval_heat, temperatures)
    sort_keys = np.asarray(
        [
            1.0 / temperature if mode == "C" and abs(temperature) > tol else temperature
            for mode, temperature in zip(modes, sort_temperatures)
        ],
        dtype=float,
    )
    return [
        {
            "name": name,
            "mode": mode,
            "weighted_area": weighted_area[index],
            "total_area": total_area[index],
            "max_heat": max_heat[index],
            "cross_pinch": cross_pinch[index],
            "sort_temperature": sort_temperatures[index],
            "sort_key": sort_keys[index],
        }
        for index, (name, mode) in enumerate(zip(names, modes))
    ]


def pinch_weights(
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
        np.abs(midpoints[above] - hot_pinch) / WEIGHTING_TEMPERATURE + 1.0
    )
    weights[below] = 1.0 / (
        np.abs(midpoints[below] - cold_pinch) / WEIGHTING_TEMPERATURE + 1.0
    )
    return weights


def cross_pinch_flags(
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
    return np.any(
        (np.abs(cascades[:, 1:]) > tol) & pinch_band[np.newaxis, :],
        axis=1,
    )


def first_active_temperatures(
    interval_heat: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    active = np.abs(interval_heat) > tol
    first_indices = np.argmax(active, axis=1)
    fallback = temperatures[0] if temperatures.size else 0.0
    return np.asarray(
        np.where(np.any(active, axis=1), temperatures[first_indices], fallback)
        + 273.15,
        dtype=float,
    )


def stack_order(headers: list[dict[str, Any]]) -> np.ndarray:
    indexed = sorted(
        enumerate(headers),
        key=lambda item: (
            MODE_ORDER.get(str(item[1]["mode"]), len(MODE_ORDER)),
            -float(item[1]["sort_key"]),
            str(item[1]["name"]),
        ),
    )
    return np.asarray([index for index, _ in indexed], dtype=int)


def stack_cascades(cascades: np.ndarray) -> np.ndarray:
    return cascades if cascades.size == 0 else np.cumsum(cascades, axis=0)


def empty_header(name: str, mode: str) -> dict[str, Any]:
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


def has_problem_table_values(table: ProblemTable, column: PT) -> bool:
    try:
        values = _as_float_array(table[column])
    except KeyError, TypeError, ValueError:
        return False
    return values.size > 0 and not bool(np.all(np.isnan(values)))


def interp_descending_temperature(
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

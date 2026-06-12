"""Exergy targeting analysis helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ...lib.config import C_to_K, tol
from ...lib.enums import GT, PT, TT
from ..common.service_orchestration import (
    apply_zone_config_overrides,
    format_selected_state_suffix,
    record_selected_state,
    target_matches_requested_state,
)

__all__ = [
    "apply_exergy_if_enabled",
    "apply_exergy_targeting",
    "build_exergy_gcc_curve",
    "build_exergy_nlp_curves",
    "compute_exergetic_temperature",
    "run_exergy_targeting_service",
]

_EXERGY_TARGET_ORDER = (
    TT.TS.value,
    TT.IHP.value,
    TT.DHP.value,
    TT.DI.value,
)


################################################################################
# Public API
################################################################################


def compute_exergetic_temperature(
    T: float, T_ref_in_C: float = 15.0, units_of_T: str = "C"
) -> float:
    """Calculate the exergetic temperature difference relative to ``T_ref``."""
    # Marmolejo-Correa, D., Gundersen, T., 2013. New Graphical Representation of
    # Exergy Applied to Low Temperature Process Design.
    if units_of_T not in ("C", "K"):
        raise ValueError("units must be either 'C' or 'K'")

    T_amb = T_ref_in_C + C_to_K
    T_K = T + C_to_K if units_of_T == "C" else T

    if T_K <= 0:
        raise ValueError("Absolute temperature must be > 0 K")

    ratio = T_K / T_amb
    return T_amb * (ratio - 1 - math.log(ratio))


def build_exergy_gcc_curve(
    *,
    temperatures: Iterable[float],
    heat_loads: Iterable[float],
    t_env: float,
    dt_cont_half: float = 0.0,
) -> dict[str, list[float]]:
    """Transform one GCC-like curve into an exergetic GCC output."""
    t_vals = _to_float_array(temperatures)
    h_vals = _to_float_array(heat_loads)
    _validate_curve_lengths(t_vals, h_vals)

    t_ex_vals: list[float] = []
    x_vals: list[float] = []
    cumulative_x = 0.0
    min_x = 0.0

    for i in range(1, len(t_vals)):
        t_upper = float(t_vals[i - 1])
        t_lower = float(t_vals[i])
        delta_t = t_upper - t_lower
        if abs(delta_t) <= tol:
            continue

        cp_net = float((h_vals[i] - h_vals[i - 1]) / delta_t)
        if cp_net > tol:
            offset = abs(dt_cont_half)
        elif cp_net < -tol:
            offset = -abs(dt_cont_half)
        else:
            offset = 0.0

        boundaries = _insert_breaks(
            [t_upper, t_lower],
            [t_env - offset],
        )
        transformed = [
            compute_exergetic_temperature(
                boundary + offset,
                T_ref_in_C=t_env,
                units_of_T="C",
            )
            for boundary in boundaries
        ]

        if not t_ex_vals:
            t_ex_vals.append(float(transformed[0]))
            x_vals.append(cumulative_x)
        elif (
            abs(t_ex_vals[-1] - transformed[0]) > tol
            or abs(x_vals[-1] - cumulative_x) > tol
        ):
            # Preserve sign-change and offset-change corners as vertical steps.
            t_ex_vals.append(float(transformed[0]))
            x_vals.append(cumulative_x)

        for j in range(1, len(boundaries)):
            cumulative_x += cp_net * (transformed[j - 1] - transformed[j])
            min_x = min(min_x, cumulative_x)
            t_ex_vals.append(float(transformed[j]))
            x_vals.append(cumulative_x)

    if not t_ex_vals:
        t_ex_vals = [
            compute_exergetic_temperature(
                float(t_vals[0]),
                T_ref_in_C=t_env,
                units_of_T="C",
            )
        ]
        x_vals = [0.0]

    shifted_x = _round_small_noise(np.asarray(x_vals, dtype=float) - min_x)
    shifted_t = _round_small_noise(np.asarray(t_ex_vals, dtype=float))
    return {
        PT.T.value: shifted_t.tolist(),
        PT.X_GCC.value: shifted_x.tolist(),
    }


def build_exergy_nlp_curves(
    *,
    temperatures: Iterable[float],
    branches: Iterable[tuple[str, Iterable[float]]],
    t_env: float,
) -> dict[str, Any]:
    """Build aggregated exergy surplus/deficit curves from NLP-like branches."""
    t_vals = _to_float_array(temperatures)
    branch_specs = [
        (kind, _to_float_array(values))
        for kind, values in branches
        if _should_use_series(values)
    ]
    if not branch_specs:
        zeros = np.zeros_like(t_vals, dtype=float)
        return {
            PT.T.value: _build_exergy_temperature_grid(t_vals, t_env).tolist(),
            PT.X_SUR.value: zeros.tolist(),
            PT.X_DEF.value: zeros.tolist(),
            "source_total": 0.0,
            "sink_total": 0.0,
        }

    for _, values in branch_specs:
        _validate_curve_lengths(t_vals, values)

    split_t = _insert_breaks(t_vals.tolist(), [t_env])
    t_grid = np.asarray(split_t, dtype=float)
    tex_grid = _build_exergy_temperature_grid(t_grid, t_env)
    interpolated = [
        (kind, _interpolate_profile(t_vals, values, t_grid))
        for kind, values in branch_specs
    ]

    source_increments: list[float] = []
    sink_increments: list[float] = []
    source_total = 0.0
    sink_total = 0.0

    for i in range(1, len(t_grid)):
        t_upper = float(t_grid[i - 1])
        t_lower = float(t_grid[i])
        delta_t = t_upper - t_lower
        if abs(delta_t) <= tol:
            source_increments.append(0.0)
            sink_increments.append(0.0)
            continue

        tex_upper = float(tex_grid[i - 1])
        tex_lower = float(tex_grid[i])
        is_above_ambient = ((t_upper + t_lower) / 2.0) >= t_env
        source_inc = 0.0
        sink_inc = 0.0

        for kind, values in interpolated:
            cp = float((values[i] - values[i - 1]) / delta_t)
            exergy_increment = abs(cp * (tex_upper - tex_lower))
            if exergy_increment <= tol:
                continue

            is_hot_branch = kind == "hot"
            if (is_hot_branch and is_above_ambient) or (
                not is_hot_branch and not is_above_ambient
            ):
                source_inc += exergy_increment
            else:
                sink_inc += exergy_increment

        source_total += source_inc
        sink_total += sink_inc
        source_increments.append(source_inc)
        sink_increments.append(sink_inc)

    surplus_profile = [source_total]
    running_source = source_total
    for increment in source_increments:
        running_source = max(running_source - increment, 0.0)
        surplus_profile.append(running_source)

    deficit_profile = [0.0]
    running_sink = 0.0
    for increment in sink_increments:
        running_sink += increment
        deficit_profile.append(running_sink)

    return {
        PT.T.value: _round_small_noise(tex_grid).tolist(),
        PT.X_SUR.value: _round_small_noise(np.asarray(surplus_profile)).tolist(),
        PT.X_DEF.value: _round_small_noise(np.asarray(deficit_profile)).tolist(),
        "source_total": float(source_total),
        "sink_total": float(sink_total),
    }


def apply_exergy_targeting(target: Any) -> Any:
    """Enrich one existing target with exergy graphs and scalar metrics."""
    spec = _resolve_target_exergy_spec(target)
    if spec is None:
        return target

    gcc_output = build_exergy_gcc_curve(
        temperatures=spec["temperatures"],
        heat_loads=spec["gcc_series"],
        t_env=spec["t_env"],
        dt_cont_half=spec["dt_cont_half"],
    )
    nlp_output = build_exergy_nlp_curves(
        temperatures=spec["temperatures"],
        branches=spec["branches"],
        t_env=spec["t_env"],
    )

    target.graphs[GT.GCC_X.value] = gcc_output
    target.graphs[GT.NLP_X.value] = {
        PT.T.value: nlp_output[PT.T.value],
        PT.X_SUR.value: nlp_output[PT.X_SUR.value],
        PT.X_DEF.value: nlp_output[PT.X_DEF.value],
    }
    target.exergy_sources = float(nlp_output["source_total"])
    target.exergy_sinks = float(nlp_output["sink_total"])
    target.ETE = (
        (target.exergy_sinks / target.exergy_sources)
        if target.exergy_sources > tol
        else 0.0
    )
    target.exergy_req_min = float(gcc_output[PT.X_GCC.value][0])
    target.exergy_des_min = float(gcc_output[PT.X_GCC.value][-1])
    return target


def apply_exergy_if_enabled(
    target: Any,
    zone,
    *,
    apply_func=apply_exergy_targeting,
) -> Any:
    """Attach exergy outputs to one target when the feature is enabled."""
    if target is None or not bool(getattr(zone.config, "DO_EXERGY_TARGETING", False)):
        return target
    return apply_func(target)


def run_exergy_targeting_service(
    zone,
    args: dict | None = None,
    *,
    apply_func=apply_exergy_targeting,
):
    """Enrich the first compatible existing target family with exergy outputs."""
    apply_zone_config_overrides(zone, args)
    zone.config.DO_EXERGY_TARGETING = True
    runtime_args = dict(args or {})
    runtime_args["DO_EXERGY_TARGETING"] = True
    explicit_target_type = _normalize_exergy_base_target_type(
        runtime_args.get("base_target_type")
    )
    idx, sid = record_selected_state(zone, runtime_args)
    runtime_args["idx"] = idx
    if sid is not None:
        runtime_args["state_id"] = sid
    compare_args = dict(args or {}) if isinstance(args, dict) else {}
    zone._selected_exergy_target_type = None

    for target_type in _get_exergy_candidate_order(explicit_target_type):
        target = zone.targets.get(target_type)
        if not target_matches_requested_state(
            target,
            args=compare_args,
            state_ids=getattr(zone, "state_ids", None),
        ):
            if explicit_target_type is not None:
                raise RuntimeError(
                    "Exergy targeting requires an existing target "
                    f"{target_type!r} on zone {zone.name!r}"
                    f"{format_selected_state_suffix(runtime_args)}. "
                    "Run the corresponding base targeting accessor first."
                )
            continue

        zone.add_target(apply_exergy_if_enabled(target, zone, apply_func=apply_func))
        zone._selected_exergy_target_type = target_type
        return zone

    raise RuntimeError(
        "Exergy targeting could not find a compatible existing target for zone "
        f"{zone.name!r}{format_selected_state_suffix(runtime_args)} "
        f"using implicit order {' -> '.join(_EXERGY_TARGET_ORDER)}. "
        "Run a compatible thermal target accessor first."
    )


################################################################################
# Helper functions
################################################################################


def _normalize_exergy_base_target_type(
    base_target_type: object | None,
) -> str | None:
    """Validate an explicit exergy base target override."""
    if base_target_type is None:
        return None

    normalized = str(base_target_type)
    if normalized not in _EXERGY_TARGET_ORDER:
        supported = ", ".join(_EXERGY_TARGET_ORDER)
        raise ValueError(
            "Unsupported exergy base_target_type "
            f"{normalized!r}. Supported types: {supported}."
        )
    return normalized


def _get_exergy_candidate_order(
    base_target_type: str | None,
) -> tuple[str, ...]:
    """Return the exact exergy target search order for this call."""
    if base_target_type is not None:
        return (base_target_type,)
    return _EXERGY_TARGET_ORDER


def _resolve_target_exergy_spec(target: Any) -> dict[str, Any] | None:
    target_type = getattr(target, "type", None)
    t_env = float(getattr(getattr(target, "config", None), "T_ENV", 15.0))
    dt_cont_half = (
        abs(float(getattr(getattr(target, "config", None), "DT_CONT", 0.0))) / 2
    )

    if target_type == TT.DI.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[PT.T],
            gcc_series=_first_available_column(pt, [PT.H_NET_A, PT.H_NET, PT.H_NET_UT]),
            branches=[
                ("hot", _optional_column(pt, PT.H_NET_HOT)),
                ("cold", _optional_column(pt, PT.H_NET_COLD)),
            ],
            t_env=t_env,
            dt_cont_half=dt_cont_half,
        )

    if target_type == TT.TS.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[PT.T],
            gcc_series=_first_available_column(pt, [PT.H_NET_UT]),
            branches=[
                ("hot", _optional_column(pt, PT.H_HOT_UT)),
                ("cold", _optional_column(pt, PT.H_COLD_UT)),
            ],
            t_env=t_env,
            dt_cont_half=dt_cont_half,
        )

    if target_type == TT.DHP.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[PT.T],
            gcc_series=_first_available_column(pt, [PT.H_NET_W_AIR, PT.H_NET_A]),
            branches=[
                ("hot", _optional_column(pt, PT.H_NET_HOT)),
                ("cold", _optional_column(pt, PT.H_NET_COLD)),
                ("hot", _optional_column(pt, PT.H_HOT_UT)),
                ("cold", _optional_column(pt, PT.H_COLD_UT)),
                ("hot", _optional_column(pt, PT.H_HOT_HP)),
                ("cold", _optional_column(pt, PT.H_COLD_HP)),
            ],
            t_env=t_env,
            dt_cont_half=dt_cont_half,
        )

    if target_type == TT.IHP.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[PT.T],
            gcc_series=_first_available_column(pt, [PT.H_NET_UT, PT.H_NET_HP]),
            branches=[
                ("hot", _optional_column(pt, PT.H_HOT_UT)),
                ("cold", _optional_column(pt, PT.H_COLD_UT)),
                ("hot", _optional_column(pt, PT.H_HOT_HP)),
                ("cold", _optional_column(pt, PT.H_COLD_HP)),
            ],
            t_env=t_env,
            dt_cont_half=dt_cont_half,
        )

    return None


def _make_target_spec(
    *,
    temperatures,
    gcc_series,
    branches,
    t_env: float,
    dt_cont_half: float,
) -> dict[str, Any] | None:
    if gcc_series is None:
        return None
    usable_branches = [
        (kind, values)
        for kind, values in branches
        if values is not None and _should_use_series(values)
    ]
    return {
        "temperatures": _to_float_array(temperatures),
        "gcc_series": _to_float_array(gcc_series),
        "branches": usable_branches,
        "t_env": float(t_env),
        "dt_cont_half": float(dt_cont_half),
    }


def _optional_column(table: Any, column_key) -> np.ndarray | None:
    try:
        values = np.asarray(table[column_key], dtype=float)
    except KeyError, TypeError, ValueError:
        return None
    if values.size == 0 or np.all(~np.isfinite(values)):
        return None
    return values


def _first_available_column(
    table: Any, column_keys: Iterable[Any]
) -> np.ndarray | None:
    for column_key in column_keys:
        values = _optional_column(table, column_key)
        if values is not None:
            return values
    return None


def _validate_curve_lengths(t_vals: np.ndarray, x_vals: np.ndarray) -> None:
    if t_vals.shape != x_vals.shape:
        raise ValueError("Temperature and curve values must have matching lengths.")


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _should_use_series(values: Iterable[float]) -> bool:
    numeric = np.asarray(list(values), dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return False
    return bool(np.any(np.abs(finite) > tol))


def _insert_breaks(
    temperatures: list[float],
    breakpoints: Iterable[float],
) -> list[float]:
    if len(temperatures) <= 1:
        return [float(value) for value in temperatures]

    descending = temperatures[0] >= temperatures[-1]
    output = [float(temperatures[0])]
    for upper, lower in zip(temperatures[:-1], temperatures[1:]):
        for breakpoint in breakpoints:
            point = float(breakpoint)
            if _strictly_between(point, float(upper), float(lower), descending):
                if abs(output[-1] - point) > tol:
                    output.append(point)
        if abs(output[-1] - float(lower)) > tol:
            output.append(float(lower))
    return output


def _strictly_between(value: float, a: float, b: float, descending: bool) -> bool:
    if descending:
        return (a - tol) > value > (b + tol)
    return (a + tol) < value < (b - tol)


def _interpolate_profile(
    temperatures: np.ndarray,
    values: np.ndarray,
    target_temperatures: np.ndarray,
) -> np.ndarray:
    order = np.argsort(temperatures)
    sorted_t = temperatures[order]
    sorted_v = values[order]
    return np.interp(target_temperatures, sorted_t, sorted_v)


def _build_exergy_temperature_grid(
    temperatures: np.ndarray,
    t_env: float,
) -> np.ndarray:
    return np.asarray(
        [
            compute_exergetic_temperature(temp, T_ref_in_C=t_env, units_of_T="C")
            for temp in temperatures
        ],
        dtype=float,
    )


def _round_small_noise(values: np.ndarray) -> np.ndarray:
    rounded = np.asarray(values, dtype=float)
    rounded[np.abs(rounded) <= tol] = 0.0
    return rounded

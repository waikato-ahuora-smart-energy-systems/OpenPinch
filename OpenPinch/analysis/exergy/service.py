"""Exergy targeting analysis helpers."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Iterable

import numpy as np

from ...domain.configuration import C_to_K, tol
from ...domain.enums import GraphType, ProblemTableLabel, TargetType
from ..targeting.context import (
    clone_target_with_zone_settings,
    format_selected_period_suffix,
    normalize_base_target_type,
    prepare_enrichment_run,
    target_matches_requested_period,
)

__all__ = [
    "apply_exergy_targeting",
    "build_exergy_gcc_curve",
    "build_exergy_nlp_curves",
    "compute_exergetic_temperature",
    "run_exergy_targeting_service",
]

_EXERGY_TARGET_ORDER = (
    TargetType.II.value,
    TargetType.IHP.value,
    TargetType.DHP.value,
    TargetType.DI.value,
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
    dt_cont_shift: float = 0.0,
    hot_shifts: Iterable[float] | None = None,
    cold_shifts: Iterable[float] | None = None,
    is_real_temperature: bool = False,
) -> dict[str, list[float]]:
    """Transform one GCC-like curve into an exergetic GCC output.

    A GCC on shifted temperatures is put back on real temperatures interval by
    interval: up by its hot shift for hot-dominated (surplus) intervals, down
    by its cold shift for cold-dominated ones. ``hot_shifts`` and
    ``cold_shifts`` give one shift per interval (the CP-weighted contribution of
    the streams in it); ``dt_cont_shift`` is used where they are absent or NaN.
    A curve already on real temperatures (``is_real_temperature``, e.g. a
    Total Site utility profile) is not moved.
    """
    t_vals = _to_float_array(temperatures)
    h_vals = _to_float_array(heat_loads)
    _validate_curve_lengths(t_vals, h_vals)
    n_intervals = max(len(t_vals) - 1, 0)
    if is_real_temperature:
        hot_offsets = cold_offsets = np.zeros(n_intervals, dtype=float)
    else:
        hot_offsets = _interval_shifts(hot_shifts, n_intervals, dt_cont_shift)
        cold_offsets = _interval_shifts(cold_shifts, n_intervals, dt_cont_shift)

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
            offset = float(hot_offsets[i - 1])
        elif cp_net < -tol:
            offset = -float(cold_offsets[i - 1])
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
        ProblemTableLabel.T.value: shifted_t.tolist(),
        ProblemTableLabel.X_GCC.value: shifted_x.tolist(),
    }


def build_exergy_nlp_curves(
    *,
    temperatures: Iterable[float],
    branches: Iterable[tuple],
    t_env: float,
    dt_cont_shift: float = 0.0,
    hot_shifts: Iterable[float] | None = None,
    cold_shifts: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Build aggregated exergy surplus/deficit curves from NLP-like branches.

    Each branch is ``(kind, values)`` or ``(kind, values, is_real)``. A branch
    on shifted temperatures moves to real temperatures interval by interval
    (hot up by its hot shift, cold down by its cold shift, as for the exergetic
    GCC) and keeps its heat capacity flow there. Intervals with different
    shifts may then overlap on real temperatures, which is physical: their
    streams overlap too. A branch already on real temperatures (``is_real``,
    e.g. a Total Site utility profile or heat-pump streams) is not moved.
    """
    t_vals = _to_float_array(temperatures)
    branch_specs = [
        (branch[0], _to_float_array(branch[1]), bool(branch[2:3] and branch[2]))
        for branch in branches
        if _should_use_series(branch[1])
    ]
    if not branch_specs:
        zeros = np.zeros_like(t_vals, dtype=float)
        return {
            ProblemTableLabel.T.value: _build_exergy_temperature_grid(
                t_vals, t_env
            ).tolist(),
            ProblemTableLabel.X_SUR.value: zeros.tolist(),
            ProblemTableLabel.X_DEF.value: zeros.tolist(),
            "source_total": 0.0,
            "sink_total": 0.0,
        }

    for _, values, _is_real in branch_specs:
        _validate_curve_lengths(t_vals, values)

    n_intervals = max(len(t_vals) - 1, 0)
    hot_offsets = _interval_shifts(hot_shifts, n_intervals, dt_cont_shift)
    cold_offsets = _interval_shifts(cold_shifts, n_intervals, dt_cont_shift)

    # (is_hot, real upper, real lower, |CP|) for every branch interval.
    segments: list[tuple[bool, float, float, float]] = []
    for kind, values, is_real in branch_specs:
        is_hot = kind == "hot"
        for k in range(n_intervals):
            delta_t = float(t_vals[k] - t_vals[k + 1])
            if abs(delta_t) <= tol:
                continue
            cp = float((values[k + 1] - values[k]) / delta_t)
            if not math.isfinite(cp) or abs(cp) <= tol:
                continue
            if is_real:
                offset = 0.0
            elif is_hot:
                offset = float(hot_offsets[k])
            else:
                offset = -float(cold_offsets[k])
            upper, lower = sorted(
                (float(t_vals[k]) + offset, float(t_vals[k + 1]) + offset),
                reverse=True,
            )
            segments.append((is_hot, upper, lower, abs(cp)))

    ends = [point for _, upper, lower, _ in segments for point in (upper, lower)]
    union = np.unique(np.concatenate([t_vals, np.asarray(ends, dtype=float)]))[::-1]
    split_t = _insert_breaks(union.tolist(), [t_env])
    t_grid = np.asarray(split_t, dtype=float)
    tex_grid = _build_exergy_temperature_grid(t_grid, t_env)

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

        tex_span = abs(float(tex_grid[i - 1]) - float(tex_grid[i]))
        is_above_ambient = ((t_upper + t_lower) / 2.0) >= t_env
        source_inc = 0.0
        sink_inc = 0.0

        for is_hot, upper, lower, cp in segments:
            if upper < t_upper - tol or lower > t_lower + tol:
                continue
            exergy_increment = cp * tex_span
            if exergy_increment <= tol:
                continue
            if is_hot == is_above_ambient:
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
        ProblemTableLabel.T.value: _round_small_noise(tex_grid).tolist(),
        ProblemTableLabel.X_SUR.value: _round_small_noise(
            np.asarray(surplus_profile)
        ).tolist(),
        ProblemTableLabel.X_DEF.value: _round_small_noise(
            np.asarray(deficit_profile)
        ).tolist(),
        "source_total": float(source_total),
        "sink_total": float(sink_total),
    }


def apply_exergy_targeting(target: Any, *, dt_cont_multiplier: float = 1.0) -> Any:
    """Enrich one existing target with exergy graphs and scalar metrics.

    Exergy uses real temperatures. Each interval of a shifted GCC or NLP curve
    is un-shifted by the CP-weighted contribution of the target zone's streams
    and utilities in it, so streams with their own ``dt_cont`` are un-shifted by
    what they were shifted by. Where no stream data covers an interval,
    ``THERMAL_DT_CONT`` times the zone's ``dt_cont_multiplier`` is used.

    Curves already on real temperatures are not moved: the Total Site utility
    profiles (indirect and indirect heat-pump targets) and heat-pump stream
    columns, whose streams carry no temperature contribution.
    """
    spec = _resolve_target_exergy_spec(target, dt_cont_multiplier=dt_cont_multiplier)
    if spec is None:
        return target

    gcc_output = build_exergy_gcc_curve(
        temperatures=spec["temperatures"],
        heat_loads=spec["gcc_series"],
        t_env=spec["t_env"],
        dt_cont_shift=spec["dt_cont_shift"],
        hot_shifts=spec.get("hot_shifts"),
        cold_shifts=spec.get("cold_shifts"),
        is_real_temperature=spec.get("gcc_is_real", False),
    )
    nlp_output = build_exergy_nlp_curves(
        temperatures=spec["temperatures"],
        branches=spec["branches"],
        t_env=spec["t_env"],
        dt_cont_shift=spec["dt_cont_shift"],
        hot_shifts=spec.get("hot_shifts"),
        cold_shifts=spec.get("cold_shifts"),
    )

    target.graphs[GraphType.GCC_X.value] = gcc_output
    target.graphs[GraphType.NLP_X.value] = {
        ProblemTableLabel.T.value: nlp_output[ProblemTableLabel.T.value],
        ProblemTableLabel.X_SUR.value: nlp_output[ProblemTableLabel.X_SUR.value],
        ProblemTableLabel.X_DEF.value: nlp_output[ProblemTableLabel.X_DEF.value],
    }
    target.exergy_sources = float(nlp_output["source_total"])
    target.exergy_sinks = float(nlp_output["sink_total"])
    target.ETE = (
        (target.exergy_sinks / target.exergy_sources)
        if target.exergy_sources > tol
        else 0.0
    )
    target.exergy_req_min = float(gcc_output[ProblemTableLabel.X_GCC.value][0])
    target.exergy_des_min = float(gcc_output[ProblemTableLabel.X_GCC.value][-1])
    return target


def run_exergy_targeting_service(
    zone,
    args: dict | None = None,
    *,
    apply_func=apply_exergy_targeting,
):
    """Enrich the first compatible existing target family with exergy outputs."""
    runtime_args, compare_args, explicit_target_type = prepare_enrichment_run(
        zone, args, _EXERGY_TARGET_ORDER, service="exergy"
    )
    zone._selected_exergy_target_type = None

    for target_type in _get_exergy_candidate_order(explicit_target_type):
        target = zone.targets.get(target_type)
        if not target_matches_requested_period(
            target,
            args=compare_args,
            period_ids=getattr(zone, "period_ids", None),
            config=zone.config,
        ):
            if explicit_target_type is not None:
                raise RuntimeError(
                    "Exergy targeting requires an existing target "
                    f"{target_type!r} on zone {zone.name!r}"
                    f"{format_selected_period_suffix(runtime_args)}. "
                    "Run the corresponding base targeting accessor first."
                )
            continue

        # Enrichment uses this invocation's settings, not the base run's settings.
        target = clone_target_with_zone_settings(target, zone, prefix="ENV_")
        enriched = (
            apply_func(target, dt_cont_multiplier=float(zone.dt_cont_multiplier))
            if apply_func is apply_exergy_targeting
            else apply_func(target)
        )
        zone.add_target(enriched, invalidate_dependents=False)
        zone._selected_exergy_target_type = target_type
        return zone

    raise RuntimeError(
        "Exergy targeting could not find a compatible existing target for zone "
        f"{zone.name!r}{format_selected_period_suffix(runtime_args)} "
        f"using implicit order {' -> '.join(_EXERGY_TARGET_ORDER)}. "
        "Run a compatible thermal target accessor first."
    )


################################################################################
# Helper functions
################################################################################


_normalize_exergy_base_target_type = partial(
    normalize_base_target_type,
    supported=_EXERGY_TARGET_ORDER,
    service="exergy",
)


def _get_exergy_candidate_order(
    base_target_type: str | None,
) -> tuple[str, ...]:
    """Return the exact exergy target search order for this call."""
    if base_target_type is not None:
        return (base_target_type,)
    return _EXERGY_TARGET_ORDER


def _resolve_target_exergy_spec(
    target: Any, *, dt_cont_multiplier: float = 1.0
) -> dict[str, Any] | None:
    target_type = getattr(target, "type", None)
    config = getattr(target, "config", None)
    environment = getattr(config, "environment", None)
    thermal = getattr(config, "thermal", None)
    t_env = float(getattr(environment, "temperature", 15.0))
    dt_cont_shift = abs(float(getattr(thermal, "dt_cont", 0.0))) * float(
        dt_cont_multiplier
    )
    stream_shifts = _zone_stream_shift_sources(target)

    if target_type == TargetType.DI.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[ProblemTableLabel.T],
            gcc_series=_first_available_column(
                pt,
                [
                    ProblemTableLabel.H_NET_A,
                    ProblemTableLabel.H_NET,
                    ProblemTableLabel.H_NET_UT,
                ],
            ),
            branches=[
                ("hot", _optional_column(pt, ProblemTableLabel.H_NET_HOT)),
                ("cold", _optional_column(pt, ProblemTableLabel.H_NET_COLD)),
            ],
            t_env=t_env,
            dt_cont_shift=dt_cont_shift,
            stream_shifts=stream_shifts,
        )

    if target_type == TargetType.II.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[ProblemTableLabel.T],
            gcc_series=_first_available_column(pt, [ProblemTableLabel.H_NET_UT]),
            # The site utility profiles are built on real utility temperatures.
            gcc_is_real=True,
            branches=[
                ("hot", _optional_column(pt, ProblemTableLabel.H_HOT_UT), True),
                ("cold", _optional_column(pt, ProblemTableLabel.H_COLD_UT), True),
            ],
            t_env=t_env,
            dt_cont_shift=dt_cont_shift,
            stream_shifts=stream_shifts,
        )

    if target_type == TargetType.DHP.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[ProblemTableLabel.T],
            gcc_series=_first_available_column(
                pt, [ProblemTableLabel.H_NET_W_AIR, ProblemTableLabel.H_NET_A]
            ),
            branches=[
                ("hot", _optional_column(pt, ProblemTableLabel.H_NET_HOT)),
                ("cold", _optional_column(pt, ProblemTableLabel.H_NET_COLD)),
                ("hot", _optional_column(pt, ProblemTableLabel.H_HOT_UT)),
                ("cold", _optional_column(pt, ProblemTableLabel.H_COLD_UT)),
                # Heat-pump streams carry no temperature contribution, so
                # their columns are already on real temperatures.
                ("hot", _optional_column(pt, ProblemTableLabel.H_HOT_HP), True),
                ("cold", _optional_column(pt, ProblemTableLabel.H_COLD_HP), True),
            ],
            t_env=t_env,
            dt_cont_shift=dt_cont_shift,
            stream_shifts=stream_shifts,
        )

    if target_type == TargetType.IHP.value:
        pt = getattr(target, "pt", None)
        if pt is None:
            return None
        return _make_target_spec(
            temperatures=pt[ProblemTableLabel.T],
            gcc_series=_first_available_column(
                pt, [ProblemTableLabel.H_NET_UT, ProblemTableLabel.H_NET_HP]
            ),
            # Built on the Total Site utility profiles plus heat-pump streams
            # with no temperature contribution: all on real temperatures.
            gcc_is_real=True,
            branches=[
                ("hot", _optional_column(pt, ProblemTableLabel.H_HOT_UT), True),
                ("cold", _optional_column(pt, ProblemTableLabel.H_COLD_UT), True),
                ("hot", _optional_column(pt, ProblemTableLabel.H_HOT_HP), True),
                ("cold", _optional_column(pt, ProblemTableLabel.H_COLD_HP), True),
            ],
            t_env=t_env,
            dt_cont_shift=dt_cont_shift,
            stream_shifts=stream_shifts,
        )

    return None


def _zone_stream_shift_sources(target: Any) -> tuple[list, list, int | None] | None:
    """Return the hot and cold streams (with utilities) behind ``target``."""
    zone = getattr(target, "parent_zone", None)
    if zone is None:
        return None
    try:
        hot = [*zone.hot_streams, *zone.hot_utilities]
        cold = [*zone.cold_streams, *zone.cold_utilities]
    except AttributeError, TypeError:
        return None
    return hot, cold, getattr(target, "period_idx", None)


def _make_target_spec(
    *,
    temperatures,
    gcc_series,
    branches,
    t_env: float,
    dt_cont_shift: float,
    stream_shifts: tuple[list, list, int | None] | None = None,
    gcc_is_real: bool = False,
) -> dict[str, Any] | None:
    if gcc_series is None:
        return None
    usable_branches = [
        tuple(branch)
        for branch in branches
        if branch[1] is not None and _should_use_series(branch[1])
    ]
    t_vals = _to_float_array(temperatures)
    hot_shifts = cold_shifts = None
    if stream_shifts is not None:
        # Each stream's own contribution, CP-weighted per interval, so streams
        # with their own dt_cont are un-shifted by what they were shifted by.
        hot_streams, cold_streams, period_idx = stream_shifts
        hot_shifts = _stream_interval_shifts(t_vals, hot_streams, period_idx)
        cold_shifts = _stream_interval_shifts(t_vals, cold_streams, period_idx)
    return {
        "temperatures": t_vals,
        "gcc_series": _to_float_array(gcc_series),
        "branches": usable_branches,
        "t_env": float(t_env),
        "dt_cont_shift": float(dt_cont_shift),
        "gcc_is_real": bool(gcc_is_real),
        "hot_shifts": hot_shifts,
        "cold_shifts": cold_shifts,
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


def _interval_shifts(
    shifts: Iterable[float] | None, n_intervals: int, default: float
) -> np.ndarray:
    """Return one non-negative shift per interval, ``default`` where unknown."""
    output = np.full(n_intervals, abs(float(default)), dtype=float)
    if shifts is None:
        return output
    values = np.asarray(list(shifts), dtype=float).ravel()
    if values.size != n_intervals:
        raise ValueError("Interval shifts must have one value per interval.")
    known = np.isfinite(values)
    output[known] = np.abs(values[known])
    return output


def _stream_interval_shifts(
    temperatures: np.ndarray,
    streams: Iterable[Any],
    period_idx: int | None,
) -> np.ndarray:
    """Return each shifted interval's CP-weighted delta-T contribution.

    A stream (or each segment of a segmented stream) counts in the intervals
    its shifted range covers. Intervals no stream covers are NaN.
    """
    from ...domain.stream_collection import StreamCollection

    t_vals = np.asarray(temperatures, dtype=float)
    n_intervals = max(t_vals.size - 1, 0)
    weighted = np.zeros(n_intervals, dtype=float)
    weights = np.zeros(n_intervals, dtype=float)
    if not n_intervals:
        return weighted
    mid = 0.5 * (t_vals[:-1] + t_vals[1:])

    def value(item, name: str) -> float:
        return float(StreamCollection._value_at_idx(getattr(item, name), period_idx))

    for stream in streams or ():
        if not getattr(stream, "is_active", True):
            continue
        parts = stream.segments if getattr(stream, "has_segments", False) else ()
        for part in parts or (stream,):
            try:
                low = value(part, "shifted_minimum_temperature")
                high = value(part, "shifted_maximum_temperature")
                cp = value(part, "heat_capacity_flowrate")
                shift = abs(value(part, "effective_delta_t_contribution"))
            except AttributeError, IndexError, TypeError, ValueError:
                continue
            if not all(map(math.isfinite, (low, high, cp, shift))) or cp <= tol:
                continue
            inside = (mid > low - tol) & (mid < high + tol)
            weighted[inside] += cp * shift
            weights[inside] += cp

    shifts = np.full(n_intervals, np.nan, dtype=float)
    covered = weights > tol
    shifts[covered] = weighted[covered] / weights[covered]
    return shifts


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

"""Logical heat pump and refrigeration loads from net load profile kinks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....contracts.hpr_load_kinks import LOGICAL_LOAD_KINDS, HprLoadKinkKind

__all__ = ["HprLoadKinkProfile", "LoadKinkVertex", "find_hpr_load_kinks"]

# Duty below this fraction of the profile's total, or a temperature change
# below this many kelvin, is treated as zero.
_DUTY_RELATIVE_TOLERANCE = 1e-9
_TEMPERATURE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class LoadKinkVertex:
    """One interior vertex in canonical units (kW, shifted degC, K)."""

    load_fraction: float
    duty: float
    temperature: float
    slope_before: float
    slope_after: float | None
    kind: HprLoadKinkKind


@dataclass(frozen=True)
class HprLoadKinkProfile:
    """Every interior vertex of one net load profile and its logical loads."""

    maximum_duty: float
    vertices: tuple[LoadKinkVertex, ...]
    logical_load_fractions: tuple[float, ...]


def find_hpr_load_kinks(
    T_vals: np.ndarray,
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    *,
    is_refrigeration: bool = False,
    steepening: float = 2.0,
) -> HprLoadKinkProfile:
    """Classify the vertices of the sink (or source) net load profile.

    Rows run from the hottest temperature down, as in a problem table. A heat
    pump walks the sink profile up from the pinch; refrigeration walks the
    source profile down from it. The slope is the temperature change per unit
    load fraction. A vertex is a logical load when the profile steps (changes
    temperature with no duty) or its slope after is at least ``steepening``
    times its slope before. The full load, 1.0, is always a logical load.
    """
    if not steepening > 1.0:
        raise ValueError("steepening must be greater than 1.")
    temperatures = np.asarray(T_vals, dtype=float)
    if is_refrigeration:
        duties = np.abs(np.asarray(H_net_hot, dtype=float))
    else:
        duties = np.abs(np.asarray(H_net_cold, dtype=float))[::-1]
        temperatures = temperatures[::-1]
    maximum_duty = float(np.nanmax(duties, initial=0.0))
    if not 0.0 < maximum_duty < np.inf:
        profile = "refrigeration source" if is_refrigeration else "heat pump sink"
        raise ValueError(f"The net load profile has no {profile} duty.")
    duty_tolerance = _DUTY_RELATIVE_TOLERANCE * maximum_duty

    start = int(np.flatnonzero(duties <= duty_tolerance)[-1])
    end = int(np.flatnonzero(duties >= maximum_duty - duty_tolerance)[0])
    points = [(duties[start], temperatures[start])]
    for duty, temperature in zip(
        duties[start + 1 : end + 1], temperatures[start + 1 : end + 1]
    ):
        last_duty, last_temperature = points[-1]
        if (
            duty - last_duty > duty_tolerance
            or abs(temperature - last_temperature) > _TEMPERATURE_TOLERANCE
        ):
            points.append((duty, temperature))

    vertices = []
    for before, vertex, after in zip(points, points[1:], points[2:]):
        if vertex[0] - before[0] <= duty_tolerance:
            # The top of a step is the same load as its foot.
            continue
        slope_before = _slope(before, vertex, maximum_duty)
        slope_after = (
            None
            if after[0] - vertex[0] <= duty_tolerance
            else _slope(vertex, after, maximum_duty)
        )
        vertices.append(
            LoadKinkVertex(
                load_fraction=float(vertex[0] / maximum_duty),
                duty=float(vertex[0]),
                temperature=float(vertex[1]),
                slope_before=slope_before,
                slope_after=slope_after,
                kind=_kind(slope_before, slope_after, steepening),
            )
        )
    logical = tuple(v.load_fraction for v in vertices if v.kind in LOGICAL_LOAD_KINDS)
    return HprLoadKinkProfile(
        maximum_duty=maximum_duty,
        vertices=tuple(vertices),
        logical_load_fractions=logical + (1.0,),
    )


def _slope(start, end, maximum_duty: float) -> float:
    """Temperature change per unit load fraction between two points."""
    return float(abs(end[1] - start[1]) / ((end[0] - start[0]) / maximum_duty))


def _kind(
    slope_before: float, slope_after: float | None, steepening: float
) -> HprLoadKinkKind:
    if slope_after is None:
        return HprLoadKinkKind.STEP
    if slope_after > 0.0 and slope_after >= steepening * slope_before:
        return HprLoadKinkKind.STEEPENS
    if slope_before > 0.0 and slope_before >= steepening * slope_after:
        return HprLoadKinkKind.FLATTENS
    return HprLoadKinkKind.MINOR

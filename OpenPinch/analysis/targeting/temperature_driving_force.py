"""Temperature-driving-force calculations for Composite Curve analysis."""

from __future__ import annotations

import math

import numpy as np

from ...domain.configuration import duty_tolerance, heat_resolution, tol
from ..numerics import interp_with_plateaus

__all__ = ["get_temperature_driving_forces"]

################################################################################
# Public API
################################################################################


def get_temperature_driving_forces(
    T_hot: np.array,
    H_hot: np.array,
    T_cold: np.array,
    H_cold: np.array,
    min_dT: float = 0,
) -> dict[str, np.ndarray]:
    """Compute interval temperature-driving-force data between two composites.

    Parameters
    ----------
    T_hot, H_hot : np.array
        Temperature and heat-flow coordinates for the hot composite curve.
    T_cold, H_cold : np.array
        Temperature and heat-flow coordinates for the cold composite curve.
    min_dT : float, default=0
        Minimum approach-temperature offset subtracted from each end-point
        driving force.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping with interval grid and per-interval values:
        ``h_vals``, ``delta_T1``, ``delta_T2``, ``dh_vals``, ``t_h1``,
        ``t_h2``, ``t_c1``, ``t_c2``.

    Raises
    ------
    ValueError
        If input arrays are empty, mismatched in length, or represent
        unbalanced hot/cold duties.
    """
    dp = int(-math.log10(tol))
    T_hot = np.asarray(T_hot, dtype=float).round(dp)
    T_cold = np.asarray(T_cold, dtype=float).round(dp)
    # Heat flows round to the problem's resolution: 6 dp from 1 kW up, finer
    # below, so a small problem keeps its significant digits.
    resolution = heat_resolution(H_hot, H_cold)
    h_dp = int(math.ceil(-math.log10(resolution))) if resolution > 0 else dp
    H_hot = np.asarray(H_hot, dtype=float).round(h_dp)
    H_cold = np.asarray(H_cold, dtype=float).round(h_dp)

    if T_hot.size != H_hot.size or T_cold.size != H_cold.size:
        raise ValueError(
            "Composite curve temperature and heat arrays must be the same length."
        )
    if T_hot.size == 0 or T_cold.size == 0:
        raise ValueError("Composite curve arrays cannot be empty.")
    duty_hot = np.max(H_hot) - np.min(H_hot)
    duty_cold = np.max(H_cold) - np.min(H_cold)
    # Rounding residue scales with the duty, so the balance test does too.
    if abs(duty_hot - duty_cold) > duty_tolerance(duty_hot, duty_cold):
        raise ValueError(
            "The temperature driving force plot requires the inputted "
            "composite curves to be balanced."
        )

    H_hot, T_hot = _normalise_curve(H_hot, T_hot, resolution)
    H_cold, T_cold = _normalise_curve(H_cold, T_cold, resolution)

    # Collect a unified heat-load grid across both curves
    h_vals = _build_h_grid(H_hot, H_cold)
    h_start = h_vals[:-1]
    h_end = h_vals[1:]
    dh_vals = h_end - h_start

    # Interpolate temperatures for each H at both ends
    t_h1 = interp_with_plateaus(H_hot, T_hot, h_start, side="right", tol=resolution)
    t_h2 = interp_with_plateaus(H_hot, T_hot, h_end, side="left", tol=resolution)
    t_c1 = interp_with_plateaus(H_cold, T_cold, h_start, side="right", tol=resolution)
    t_c2 = interp_with_plateaus(H_cold, T_cold, h_end, side="left", tol=resolution)

    # Each slice ends at its left limit, also where a composite steps in
    # temperature at constant heat flow, so the LMTD is that of the slice.
    delta_T1 = t_h1 - t_c1 - min_dT
    delta_T2 = t_h2 - t_c2 - min_dT

    return {
        "h_vals": h_vals,
        "delta_T1": delta_T1,
        "delta_T2": delta_T2,
        "dh_vals": dh_vals,
        "t_h1": t_h1,
        "t_h2": t_h2,
        "t_c1": t_c1,
        "t_c2": t_c2,
    }


################################################################################
# Helper functions
################################################################################


def _build_h_grid(h_hot: np.ndarray, h_cold: np.ndarray) -> np.ndarray:
    """Create a sorted heat-flow grid shared by hot and cold composites.

    Parameters
    ----------
    h_hot, h_cold : np.ndarray
        Heat-flow coordinates.

    Returns
    -------
    np.ndarray
        Sorted union of both heat-flow arrays.
    """
    return np.union1d(h_hot, h_cold).astype(float, copy=False)


def _normalise_curve(
    h_vals: np.ndarray, t_vals: np.ndarray, resolution: float = tol
) -> tuple[np.ndarray, np.ndarray]:
    """Orient a composite curve and shift heat flow so it starts at zero.

    Parameters
    ----------
    h_vals : np.ndarray
        Heat-flow coordinates.
    t_vals : np.ndarray
        Temperature coordinates aligned with ``h_vals``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized heat-flow and temperature arrays.

    Raises
    ------
    ValueError
        If ``h_vals`` and ``t_vals`` do not have equal length.
    """
    if h_vals.size != t_vals.size:
        raise ValueError(
            "Composite curve temperature and heat arrays must be the same length."
        )

    # Orient so heat flow increases from lower to higher values
    if h_vals[0] > h_vals[-1]:
        h_vals = h_vals[::-1]
        t_vals = t_vals[::-1]

    # Shift the curve so it begins at zero heat load
    offset = h_vals[0]
    if abs(offset) > resolution:
        h_vals = h_vals - offset

    return h_vals, t_vals

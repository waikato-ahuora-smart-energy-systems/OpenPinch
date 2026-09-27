"""Log-mean temperature difference helpers used by area targeting routines."""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_LMTD_from_dts",
    "compute_LMTD_from_ts",
]


def compute_LMTD_from_dts(
    delta_T1: float | list | np.ndarray,
    delta_T2: float | list | np.ndarray,
) -> np.ndarray:
    """Return the LMTD for a counterflow heat exchanger from end-point deltas."""
    # Check temperature directions for counter-current assumption
    delta_T1 = np.array(delta_T1)
    delta_T2 = np.array(delta_T2)

    if delta_T1.round(6).min() <= 0 or delta_T2.round(6).min() <= 0:
        raise ValueError(
            f"Invalid temperature differences: ΔT1={delta_T1}, ΔT2={delta_T2}"
        )
    mask_equal = np.isclose(delta_T1, delta_T2, atol=1e-6)
    lmtd = np.empty_like(delta_T1, dtype=float)
    arithmetic = (delta_T1 + delta_T2) / 2
    np.copyto(lmtd, arithmetic, where=mask_equal)
    np.divide(
        delta_T1 - delta_T2,
        np.log(delta_T1 / delta_T2),
        out=lmtd,
        where=~mask_equal,
    )
    return lmtd


def compute_LMTD_from_ts(
    T_hot_in: float | list | np.ndarray,
    T_hot_out: float | list | np.ndarray,
    T_cold_in: float | list | np.ndarray,
    T_cold_out: float | list | np.ndarray,
) -> float:
    """Return the LMTD for a counterflow heat exchanger from temperatures."""
    T_hot_in = np.array(T_hot_in)
    T_hot_out = np.array(T_hot_out)
    T_cold_in = np.array(T_cold_in)
    T_cold_out = np.array(T_cold_out)

    # Check temperature directions for counter-current assumption
    if T_hot_in < T_hot_out:
        raise ValueError("Hot fluid must cool down (T_hot_in > T_hot_out)")
    if T_cold_out < T_cold_in:
        raise ValueError("Cold fluid must heat up (T_cold_out > T_cold_in)")

    return compute_LMTD_from_dts(
        T_hot_in - T_cold_out,  # Inlet diff (hottest hot - hottest cold)
        T_hot_out - T_cold_in,  # Outlet diff (coldest hot - coldest cold)
    )

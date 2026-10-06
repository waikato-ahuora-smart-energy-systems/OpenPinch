"""Heat available to an ORC from the process surplus below the pinch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["OrcHeatSource", "orc_heat_source_from_gcc"]


@dataclass(frozen=True)
class OrcHeatSource:
    """The grand composite curve below the pinch, on shifted temperatures.

    ``T`` falls from the pinch to the coldest interval and ``H`` is the heat
    cascaded down through each temperature (zero at the pinch). An ORC
    evaporator is a heat sink: taking ``Q`` at shifted temperature ``T_s``
    lowers the cascade by ``Q`` at every temperature below ``T_s``, so the
    total an ORC can take at or above ``T_s`` without adding hot utility is
    the smallest cascaded heat at or below ``T_s``.
    """

    T: np.ndarray
    H: np.ndarray

    def __post_init__(self) -> None:
        T = np.asarray(self.T, dtype=float)
        H = np.asarray(self.H, dtype=float)
        if T.ndim != 1 or T.shape != H.shape or T.size < 2:
            raise ValueError("ORC heat source needs matching T and H arrays.")
        if np.any(np.diff(T) > 0.0):
            raise ValueError("ORC heat source temperatures must fall.")
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "H", np.maximum(H, 0.0))

    @property
    def T_pinch(self) -> float:
        """Shifted pinch temperature, the hottest an ORC may take heat at."""
        return float(self.T[0])

    @property
    def T_min(self) -> float:
        """Coldest shifted temperature of the surplus."""
        return float(self.T[-1])

    @property
    def surplus(self) -> float:
        """Heat rejected to cold utility below the pinch, in kW."""
        return float(self.H[-1])

    def heat_at(self, T_s: float) -> float:
        """Cascaded heat at shifted temperature ``T_s`` (linear in T)."""
        return float(np.interp(T_s, self.T[::-1], self.H[::-1]))

    def capacity(self, T_s: float) -> float:
        """Most heat sinks at or above ``T_s`` can take without hot utility."""
        if T_s >= self.T_pinch:
            return 0.0
        below = self.H[self.T <= T_s]
        at = self.heat_at(max(T_s, self.T_min))
        return float(min(at, below.min())) if below.size else at


def orc_heat_source_from_gcc(
    T_vals: np.ndarray,
    H_net: np.ndarray,
    *,
    tol: float = 1e-6,
) -> OrcHeatSource | None:
    """Return the GCC below the pinch, or ``None`` if there is no surplus.

    ``T_vals`` are the problem table's shifted temperatures (falling) and
    ``H_net`` its grand composite curve. The pinch is the coldest zero of the
    curve, so a threshold problem with no hot utility starts at the top.
    """
    T = np.asarray(T_vals, dtype=float)
    H = np.asarray(H_net, dtype=float)
    if T.size < 2 or T.shape != H.shape:
        return None
    order = np.argsort(-T, kind="stable")
    T, H = T[order], H[order]
    zeros = np.flatnonzero(H <= tol)
    if zeros.size == 0:
        return None
    start = int(zeros[-1])
    T_below, H_below = T[start:], H[start:]
    # Collapse repeated temperatures, keeping the lowest cascaded heat.
    keep = np.ones(T_below.size, dtype=bool)
    for i in range(1, T_below.size):
        if abs(T_below[i] - T_below[i - 1]) <= tol:
            H_below[i] = min(H_below[i], H_below[i - 1])
            keep[i - 1] = False
    T_below, H_below = T_below[keep], H_below[keep]
    if T_below.size < 2 or float(H_below.max()) <= tol:
        return None
    return OrcHeatSource(T=T_below, H=H_below)

"""Independent fine-slice Bath-formula area integration for area-target tests.

The oracle builds the hot and cold balanced composite curves directly from the
stream and utility segments (real temperatures, heat flows and film
coefficients), slices the shared heat-flow axis finely, and sums
``dQ * (r_hot + r_cold) / (T_hot - T_cold)`` at slice midpoints, where ``r`` is
the heat-weighted film resistance of the streams active at that point. It
shares no code with OpenPinch's problem table or driving-force routines.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bath_area", "composite_segments"]


def composite_segments(segments, *, isothermal_dt: float = 1e-9):
    """Return composite pieces ``(H_lo, H_hi, T_lo, T_hi, r)`` in ascending T.

    ``segments`` is an iterable of ``(t_min, t_max, heat_flow, htc)`` tuples.
    A segment whose temperature span is below ``isothermal_dt`` is treated as
    isothermal: it adds a constant-temperature piece of width ``heat_flow``.
    """
    sensible = []
    isothermal = []
    for t_min, t_max, heat_flow, htc in segments:
        if heat_flow <= 0.0:
            continue
        if t_max - t_min < isothermal_dt:
            isothermal.append((0.5 * (t_min + t_max), heat_flow, htc))
        else:
            cp = heat_flow / (t_max - t_min)
            sensible.append((t_min, t_max, cp, cp / htc))

    temperatures = sorted(
        {t for lo, hi, _, _ in sensible for t in (lo, hi)}
        | {t for t, _, _ in isothermal}
    )
    pieces = []
    h = 0.0
    for i, t_lo in enumerate(temperatures):
        q_iso = sum(q for t, q, _ in isothermal if t == t_lo)
        if q_iso > 0.0:
            r_iso = sum(q / htc for t, q, htc in isothermal if t == t_lo) / q_iso
            pieces.append((h, h + q_iso, t_lo, t_lo, r_iso))
            h += q_iso
        if i + 1 == len(temperatures):
            break
        t_hi = temperatures[i + 1]
        active = [
            (cp, rcp) for lo, hi, cp, rcp in sensible if lo <= t_lo and hi >= t_hi
        ]
        cp_sum = sum(cp for cp, _ in active)
        if cp_sum <= 0.0:
            continue
        dh = cp_sum * (t_hi - t_lo)
        pieces.append((h, h + dh, t_lo, t_hi, sum(rcp for _, rcp in active) / cp_sum))
        h += dh
    return np.asarray(pieces, dtype=float)


def _evaluate(pieces: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return composite temperature and resistance at heat flows ``h``."""
    k = np.clip(np.searchsorted(pieces[:, 1], h), 0, len(pieces) - 1)
    h_lo, h_hi, t_lo, t_hi, r = pieces[k].T
    t = t_lo + (h - h_lo) / (h_hi - h_lo) * (t_hi - t_lo)
    return t, r


def bath_area(hot_segments, cold_segments, *, slices: int = 400_000) -> float:
    """Integrate the Bath area between the hot and cold balanced composites."""
    hot = composite_segments(hot_segments)
    cold = composite_segments(cold_segments)
    total = hot[-1, 1]
    if not np.isclose(total, cold[-1, 1], rtol=1e-6, atol=0.0):
        raise ValueError(f"Composites are not balanced: {total} vs {cold[-1, 1]}")
    edges = np.linspace(0.0, total, slices + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    t_hot, r_hot = _evaluate(hot, mid)
    t_cold, r_cold = _evaluate(cold, mid)
    return float(np.sum(np.diff(edges) * (r_hot + r_cold) / (t_hot - t_cold)))

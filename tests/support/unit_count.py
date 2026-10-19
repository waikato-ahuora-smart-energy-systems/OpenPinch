"""Independent minimum-unit count, N = S - 1 per pinch region.

The oracle cascades the balanced problem (process streams plus utilities at
their targeted duties) on shifted temperatures, splits it at every zero of
the cascade, and counts in each region the streams whose shifted range
overlaps it. It shares no code with OpenPinch's problem table.
"""

from __future__ import annotations

import numpy as np

__all__ = ["minimum_units"]


def minimum_units(hot, cold, *, rtol: float = 1e-6) -> int:
    """Return sum(S - 1) over the pinch regions of a balanced problem.

    ``hot`` and ``cold`` are iterables of ``(stream_id, t_min_star,
    t_max_star, heat_flow)``. Segments sharing a ``stream_id`` count once.
    """
    hot = [s for s in hot if s[3] > 0.0]
    cold = [s for s in cold if s[3] > 0.0]
    temperatures = np.unique([t for s in hot + cold for t in (s[1], s[2])])[::-1]

    def released(streams, t_hi, t_lo):
        total = 0.0
        for _, t_min, t_max, q in streams:
            overlap = min(t_hi, t_max) - max(t_lo, t_min)
            if overlap > 0.0:
                total += q * overlap / (t_max - t_min)
        return total

    cascade = [0.0]
    for t_hi, t_lo in zip(temperatures[:-1], temperatures[1:]):
        cascade.append(
            cascade[-1] + released(hot, t_hi, t_lo) - released(cold, t_hi, t_lo)
        )
    cascade = np.asarray(cascade)
    scale = max(sum(s[3] for s in hot), sum(s[3] for s in cold))
    zeros = np.flatnonzero(np.abs(cascade) <= rtol * scale)
    zeros = np.union1d(zeros, [0, len(cascade) - 1])

    units = 0
    for upper, lower in zip(zeros[:-1], zeros[1:]):
        t_hi, t_lo = temperatures[upper], temperatures[lower]
        present = {
            sid
            for sid, t_min, t_max, _ in hot + cold
            if min(t_hi, t_max) - max(t_lo, t_min) > 0.0
        }
        if present:
            units += len(present) - 1
    return units

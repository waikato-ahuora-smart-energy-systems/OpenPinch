"""Hold a sliced utility's slices in series.

A sliding utility cut into slices (``../../solver/utility_slices.py``) is one
utility flow passing every slice in turn, so each slice's duty is the
utility's times the slice's share of its temperature range. Each pair of
neighbouring slices is held to that ratio in every period, stream-end and
staged duties together. A model that holds each slice at its level target
meets it already; one that prices utilities freely needs it, or a slice
could return the utility part-way down its range.
"""

from __future__ import annotations

from ...solver.utility_slices import utility_slices
from .._stagewise import staged_utilities as _staged


def set_utility_slice_coupling(owner) -> None:
    """Add ``Q[s] * share[t] == Q[t] * share[s]`` for neighbouring slices."""

    slices = utility_slices(getattr(owner, "solver_arrays", None))
    if not slices:
        return
    for side, axis in (("hot", "hot_utilities"), ("cold", "cold_utilities")):
        names = list(owner.solver_arrays.utility_identities[axis])
        chains: dict[str, list[tuple[int, float]]] = {}
        for u, name in enumerate(names):
            if name in slices.get(axis, {}):
                parent, share = slices[axis][name]
                chains.setdefault(parent, []).append((u, float(share)))
        for chain in chains.values():
            for (s, share_s), (t, share_t) in zip(chain[:-1], chain[1:]):
                for n in range(owner.N_periods):
                    owner.m.Equation(
                        _duty(owner, side, n, s) * share_t
                        == _duty(owner, side, n, t) * share_s
                    )


def _duty(owner, side: str, n: int, u: int):
    grid = owner.Q_h_by_period[n] if side == "hot" else owner.Q_c_by_period[n]
    terms = [row[u] for row in grid if row[u] is not None]
    terms += _staged.staged_utility_loads(owner, side, n, u)
    return sum(terms)


__all__ = ["set_utility_slice_coupling"]

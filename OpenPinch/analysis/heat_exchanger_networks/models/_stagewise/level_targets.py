"""Hold utility levels at given duties over a group of streams.

A site network keeps each process's utility levels at the duties its pinch
design targeted. A target names a side, the utility indices of one level
(several when the level is cut into slices), the process streams it serves
(cold streams for a hot utility, hot streams for a cold one) and a duty per
period. Stream-end and staged units count together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import staged_utilities as _staged

# Relative slack on each target, as in the pinch design sides: rounding
# between targeting and the model, not a tolerance prices could exploit.
LEVEL_BAND = 1e-7


@dataclass(frozen=True)
class UtilityLevelTarget:
    """One level's duty per period over a group of process streams."""

    side: Literal["hot", "cold"]
    utilities: tuple[int, ...]
    streams: tuple[int, ...]
    duties: tuple[float, ...]


def set_level_target_equations(owner) -> None:
    """Add ``duty == target`` (within the band) for each level target."""

    for target in getattr(owner, "level_targets", ()):
        for n in range(owner.N_periods):
            duty = _level_duty(owner, target, n)
            level = float(target.duties[n])
            band = LEVEL_BAND * max(1.0, abs(level))
            if isinstance(duty, int | float):
                if abs(duty - level) > band:
                    raise ValueError(
                        f"A {target.side} utility level target of {level} kW "
                        "has no exchanger that can serve it."
                    )
                continue
            owner.m.Equation(duty <= level + band)
            owner.m.Equation(duty >= level - band)


def _level_duty(owner, target: UtilityLevelTarget, n: int):
    hot = target.side == "hot"
    grid = owner.Q_h_by_period[n] if hot else owner.Q_c_by_period[n]
    staging = getattr(owner, "staged_utilities", False)
    if staging:
        staged = owner.Q_hs_by_period[n] if hot else owner.Q_cs_by_period[n]
    terms = []
    for s in target.streams:
        for u in target.utilities:
            if grid[s][u] is not None:
                terms.append(grid[s][u])
            if staging:
                terms += [
                    staged[s][u][k]
                    for k in _staged.interior_boundaries(owner)
                    if staged[s][u][k] is not None
                ]
    return sum(terms)


__all__ = ["LEVEL_BAND", "UtilityLevelTarget", "set_level_target_equations"]

"""Hold utility levels at given duties over a group of streams.

A site network keeps each process's utility levels at the duties its pinch
design targeted. A target names a side, the utility indices of one level
(several when the level is cut into slices), the process streams it serves
(cold streams for a hot utility, hot streams for a cold one) and a duty per
period. Stream-end and staged units count together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ...solver.utility_slices import utility_slices
from . import staged_utilities as _staged

# Relative slack on each target, as in the pinch design sides: rounding
# between targeting and the model, not a tolerance prices could exploit.
LEVEL_BAND = 1e-7


@dataclass(frozen=True)
class UtilityLevelTarget:
    """One level's duty per period over a group of process streams.

    ``at_least`` holds the duty at or above the target rather than at it.
    """

    side: Literal["hot", "cold"]
    utilities: tuple[int, ...]
    streams: tuple[int, ...]
    duties: tuple[float, ...]
    at_least: bool = False


def set_level_target_equations(owner) -> None:
    """Add ``duty == target`` (within the band) for each level target, or
    ``duty >= target`` (less the band) for an ``at_least`` target."""

    for target in getattr(owner, "level_targets", ()):
        for n in range(owner.N_periods):
            duty = _level_duty(owner, target, n)
            level = float(target.duties[n])
            band = LEVEL_BAND * max(1.0, abs(level))
            if isinstance(duty, int | float):
                short = duty < level - band
                if short or (not target.at_least and duty > level + band):
                    raise ValueError(
                        f"A {target.side} utility level target of {level} kW "
                        "has no exchanger that can serve it."
                    )
                continue
            if not target.at_least:
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


def site_level_targets(
    solver_arrays,
    processes: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    at_least: bool = False,
) -> tuple[tuple[UtilityLevelTarget, ...], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Each process's levels as targets, and each stream's process as its group.

    ``processes`` maps a process's stream-key prefix (``"<process>.<stream>"``)
    to its targeted duty per period by utility name (``"Hot Utility.LPS"``).
    A sliced utility's slices count as their parent. A stream of no listed
    process is in the group ``""``. ``at_least`` holds each level at or
    above its target.
    """

    identities = solver_arrays.stream_identities
    groups = (
        tuple(
            _process_of(name, processes) for name in identities["hot_process_streams"]
        ),
        tuple(
            _process_of(name, processes) for name in identities["cold_process_streams"]
        ),
    )
    slices = utility_slices(solver_arrays)
    targets: list[UtilityLevelTarget] = []
    for side, axis, served in (
        ("hot", "hot_utilities", groups[1]),
        ("cold", "cold_utilities", groups[0]),
    ):
        names = list(solver_arrays.utility_identities[axis])
        parents = [slices.get(axis, {}).get(name, (name, 1.0))[0] for name in names]
        for process, levels in processes.items():
            streams = tuple(s for s, group in enumerate(served) if group == process)
            for name, duties in levels.items():
                utilities = tuple(
                    u for u, parent in enumerate(parents) if parent == name
                )
                if not utilities:
                    continue
                targets.append(
                    UtilityLevelTarget(
                        side,
                        utilities,
                        streams,
                        tuple(float(d) for d in duties),
                        at_least,
                    )
                )
    return tuple(targets), groups


def _process_of(stream: str, processes: Mapping[str, object]) -> str:
    matches = [process for process in processes if stream.startswith(f"{process}.")]
    return max(matches, key=len, default="")


__all__ = [
    "LEVEL_BAND",
    "UtilityLevelTarget",
    "set_level_target_equations",
    "site_level_targets",
]

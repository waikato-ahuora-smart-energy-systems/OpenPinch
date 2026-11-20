"""Synthesis search width and depth levels.

``HENS_SEARCH_WIDTH`` sets the dt_cont multipliers synthesis runs at.
``HENS_SEARCH_DEPTH`` sets which stages run and how hard each searches:

* 0: heuristic pinch design only, no evolution.
* 1: heuristic pinch design, then network evolution two branches wide.
* 2: heuristic and MILP pinch design; MILP networks go through the thermal
  derivative method, heuristic networks straight to evolution.
* 3: as 2 with more heuristic networks seeding evolution and every MILP
  network also straight to evolution.
* 4: as 3 with still more seeds and wider evolution.

An option set explicitly overrides the value its level gives. At depths 0
and 1 the level's heuristic network count falls with the case size
(``scaled_heuristic_networks``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WIDTH_MULTIPLIERS: dict[int, tuple[float, ...]] = {
    0: (1.0,),
    1: (0.75, 1.0, 1.5),
    2: (0.5, 0.75, 1.0, 1.5, 2.0),
    3: (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0),
}


@dataclass(frozen=True)
class DepthLevel:
    """Stages and effort of one search depth."""

    evolution: bool
    heuristic_networks: int
    milp: bool
    evolution_breadth: int
    thermal_derivative: bool = False
    milp_seeds_evolution: bool = False
    # Heuristic networks per multiplier that seed evolution, cheapest first.
    heuristic_evolution_seeds: int = 3


DEPTH_LEVELS: dict[int, DepthLevel] = {
    0: DepthLevel(False, 50, False, 1),
    1: DepthLevel(True, 50, False, 2),
    2: DepthLevel(True, 50, True, 2, thermal_derivative=True),
    3: DepthLevel(
        True,
        50,
        True,
        2,
        thermal_derivative=True,
        milp_seeds_evolution=True,
        heuristic_evolution_seeds=20,
    ),
    4: DepthLevel(
        True,
        50,
        True,
        3,
        thermal_derivative=True,
        milp_seeds_evolution=True,
        heuristic_evolution_seeds=50,
    ),
}


# Up to this many streams a case keeps its level's heuristic network count.
FULL_HEURISTIC_STREAMS = 30
# The fewest heuristic networks (and side proposals) a scaled count gives.
MIN_SCALED_HEURISTIC_NETWORKS = 10
# Depths whose heuristic network count scales with the case size.
SCALED_HEURISTIC_DEPTHS = frozenset({0, 1})


def scaled_heuristic_networks(networks: int, streams: int) -> int:
    """A depth level's heuristic network count for a case of ``streams``.

    It caps the proposals realised per side and the networks paired. Each
    side realisation and network costs more on a larger case, so above
    ``FULL_HEURISTIC_STREAMS`` streams the count falls with the square of the
    stream count, to no fewer than ``MIN_SCALED_HEURISTIC_NETWORKS``.
    """

    if streams <= FULL_HEURISTIC_STREAMS:
        return networks
    return max(
        MIN_SCALED_HEURISTIC_NETWORKS,
        networks * FULL_HEURISTIC_STREAMS**2 // streams**2,
    )


@dataclass(frozen=True)
class ResolvedSearch:
    """The options a search width and depth set, after explicit overrides."""

    width: int
    depth: int
    level: DepthLevel
    dt_cont_multipliers: tuple[float, ...]
    heuristic_networks: int
    heuristic_evolution_seeds: int
    milp: bool
    evm_n_ad_branches: int
    evm_n_rm_branches: int


def reference_dt_cont_multiplier(multipliers) -> float:
    """The multiplier a run falls back to when nothing else gives one.

    1.0, the targeting contributions, when it is among the multipliers (as in
    every width level); otherwise the first configured multiplier.
    """

    values = tuple(float(value) for value in multipliers)
    return 1.0 if 1.0 in values else values[0]


def resolve_search(hens: Any, *, streams: int | None = None) -> ResolvedSearch:
    """Return the search options of a HEN configuration group.

    With ``streams``, the case's process stream count, the heuristic network
    count a depth 0 or 1 level gives is scaled to the case size.
    """

    width, depth = int(hens.search_width), int(hens.search_depth)
    level = DEPTH_LEVELS[depth]
    level_networks = level.heuristic_networks
    if streams is not None and depth in SCALED_HEURISTIC_DEPTHS:
        level_networks = scaled_heuristic_networks(level_networks, streams)

    def pick(value, default):
        return default if value is None else value

    return ResolvedSearch(
        width=width,
        depth=depth,
        level=level,
        dt_cont_multipliers=tuple(
            float(value)
            for value in pick(hens.dt_cont_multipliers, WIDTH_MULTIPLIERS[width])
        ),
        heuristic_networks=int(pick(hens.heuristic_pdm_networks, level_networks)),
        heuristic_evolution_seeds=int(
            pick(hens.heuristic_pdm_evolution_seeds, level.heuristic_evolution_seeds)
        ),
        milp=bool(pick(hens.pdm_milp, level.milp)),
        evm_n_ad_branches=int(pick(hens.evm_n_ad_branches, level.evolution_breadth)),
        evm_n_rm_branches=int(pick(hens.evm_n_rm_branches, level.evolution_breadth)),
    )


__all__ = [
    "DEPTH_LEVELS",
    "FULL_HEURISTIC_STREAMS",
    "MIN_SCALED_HEURISTIC_NETWORKS",
    "SCALED_HEURISTIC_DEPTHS",
    "WIDTH_MULTIPLIERS",
    "DepthLevel",
    "ResolvedSearch",
    "reference_dt_cont_multiplier",
    "resolve_search",
    "scaled_heuristic_networks",
]

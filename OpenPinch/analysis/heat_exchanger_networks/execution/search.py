"""Synthesis search width and depth levels.

``HENS_SEARCH_WIDTH`` sets the dt_cont multipliers synthesis runs at.
``HENS_SEARCH_DEPTH`` sets which stages run and how hard each searches:

* 0: heuristic pinch design only, no evolution.
* 1: heuristic pinch design, then network evolution.
* 2: heuristic and MILP pinch design; MILP networks go through the thermal
  derivative method, heuristic networks straight to evolution.
* 3: as 2 with more heuristic networks, extra minimum-unit MILP networks,
  every MILP network also straight to evolution, and wider evolution.
* 4: as 3, wider still.

An option set explicitly overrides the value its level gives.
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
    min_unit_networks: int
    extra_unit_networks: int
    evolution_breadth: int
    thermal_derivative: bool = False
    milp_seeds_evolution: bool = False


DEPTH_LEVELS: dict[int, DepthLevel] = {
    0: DepthLevel(False, 10, False, 0, 0, 1),
    1: DepthLevel(True, 10, False, 0, 0, 1),
    2: DepthLevel(True, 10, True, 0, 0, 1, thermal_derivative=True),
    3: DepthLevel(
        True, 20, True, 3, 2, 2, thermal_derivative=True, milp_seeds_evolution=True
    ),
    4: DepthLevel(
        True, 50, True, 5, 5, 3, thermal_derivative=True, milp_seeds_evolution=True
    ),
}


@dataclass(frozen=True)
class ResolvedSearch:
    """The options a search width and depth set, after explicit overrides."""

    width: int
    depth: int
    level: DepthLevel
    dt_cont_multipliers: tuple[float, ...]
    heuristic_networks: int
    milp: bool
    min_unit_networks: int
    extra_unit_networks: int
    evm_n_ad_branches: int
    evm_n_rm_branches: int


def resolve_search(hens: Any) -> ResolvedSearch:
    """Return the search options of a HEN configuration group."""

    width, depth = int(hens.search_width), int(hens.search_depth)
    level = DEPTH_LEVELS[depth]

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
        heuristic_networks=int(
            pick(hens.heuristic_pdm_networks, level.heuristic_networks)
        ),
        milp=bool(pick(hens.pdm_milp, level.milp)),
        min_unit_networks=int(
            pick(hens.pdm_min_unit_networks, level.min_unit_networks)
        ),
        extra_unit_networks=int(
            pick(hens.pdm_extra_unit_networks, level.extra_unit_networks)
        ),
        evm_n_ad_branches=int(pick(hens.evm_n_ad_branches, level.evolution_breadth)),
        evm_n_rm_branches=int(pick(hens.evm_n_rm_branches, level.evolution_breadth)),
    )


__all__ = [
    "DEPTH_LEVELS",
    "WIDTH_MULTIPLIERS",
    "DepthLevel",
    "ResolvedSearch",
    "resolve_search",
]

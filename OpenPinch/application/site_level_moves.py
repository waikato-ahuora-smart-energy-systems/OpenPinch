"""Level-temperature moves on a placed total site design.

A placed level is a main: a hot and a cold utility of one name
(``site_levels.placed_case``). A move shifts both by a step, keeping every
placed level in order, apart from the others, and within the site's stream
temperatures. The site is then designed again at the shifted level, each
process holding its levels re-targeted there, and the move is adopted only
on a lower site total annual cost. With no improving move the step halves,
down to its minimum. The structure is not fixed by the move itself: the
re-design starts from the current best network's recovery matches.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .site_levels import site_cost

if TYPE_CHECKING:
    from .problem import PinchProblem

# Placed level names (``site_levels.placed_case``).
_PLACED = re.compile(r"(iso|sensible)_\d+")
# A relative gain below this is no gain: re-solves differ by solver noise.
_GAIN = 1e-6


@dataclass(frozen=True)
class LevelMove:
    """One tried move: ``status`` is ``"adopted"``, ``"rejected"`` (no lower
    cost) or ``"failed"`` (the design raised). The first entry is the
    unshifted levels evaluated as a move (``"baseline"``, level ``""`` and
    shift 0, or ``"failed"``)."""

    level: str
    shift: float
    total_annual_cost: float | None
    status: str
    message: str | None = None

    def row(self) -> dict[str, Any]:
        """The move as a JSON-safe record."""
        return {
            "level": self.level,
            "shift": self.shift,
            "total_annual_cost": self.total_annual_cost,
            "status": self.status,
            "message": self.message,
        }


def movable_levels(case: "PinchProblem") -> tuple[str, ...]:
    """The placed levels of ``case``, by name."""
    names = {
        utility["name"]
        for utility in case.to_problem_json()["utilities"]
        if _PLACED.fullmatch(utility["name"])
    }
    return tuple(sorted(names))


def _values(quantity) -> list[float]:
    return list(quantity["values"]) if "values" in quantity else [quantity["value"]]


def _shift(quantity, shift: float):
    if "values" in quantity:
        return quantity | {"values": [v + shift for v in quantity["values"]]}
    return quantity | {"value": quantity["value"] + shift}


def _span(utilities) -> tuple[float, float]:
    """The lowest and highest temperature of a level's utilities."""
    temperatures = [
        t
        for u in utilities
        for key in ("t_supply", "t_target")
        for t in _values(u[key])
    ]
    return min(temperatures), max(temperatures)


def shifted_case(
    case: "PinchProblem", level: str, shift: float
) -> "PinchProblem | None":
    """``case`` with placed ``level`` moved by ``shift`` (K), or ``None``.

    The move is refused when the level would touch or pass another placed
    level, or leave the range of the site's stream temperatures.
    """
    source = case.to_problem_json()
    by_level: dict[str, list] = {}
    for utility in source["utilities"]:
        if _PLACED.fullmatch(utility["name"]):
            by_level.setdefault(utility["name"], []).append(utility)
    order = sorted(by_level, key=lambda name: _span(by_level[name]))
    for utility in by_level[level]:
        for key in ("t_supply", "t_target"):
            utility[key] = _shift(utility[key], shift)
    spans = [_span(by_level[name]) for name in order]
    if any(low <= below for (_, below), (low, _) in zip(spans, spans[1:])):
        return None
    streams = [
        t
        for stream in source["streams"]
        for key in ("t_supply", "t_target")
        for t in _values(stream[key])
    ]
    low, high = _span(by_level[level])
    if low < min(streams) or high > max(streams):
        return None
    return type(case)(source, project_name=case.project_name)


def _level_key(case: "PinchProblem") -> tuple:
    """The placed levels' temperatures, as a key for a designed case."""
    return tuple(
        (
            u["name"],
            u["type"],
            *(round(t, 9) for k in ("t_supply", "t_target") for t in _values(u[k])),
        )
        for u in sorted(
            case.to_problem_json()["utilities"], key=lambda u: (u["name"], u["type"])
        )
        if _PLACED.fullmatch(u["name"])
    )


def _seed_key(view) -> tuple:
    """The recovery matches a move from ``view`` is seeded with."""
    from ..analysis.heat_exchanger_networks.site.seeds import _recovery

    network = view.selected_network
    if not hasattr(network, "exchangers"):
        return ()
    return tuple(sorted(e.model_dump_json() for e in _recovery(network)))


def level_duties(
    case: "PinchProblem", multiplier: float
) -> dict[str, dict[str, list[float]]]:
    """Each process zone's targeted duty per period by utility identity.

    As a process's level-held pinch design targets its levels: each process
    alone, with every utility of the site, at the dt_cont multiplier's
    approach (``seeds.process_level_duties`` reads the same from a network).
    A process of only hot or only cold streams is targeted the same way.
    """
    from ..analysis.heat_exchanger_networks.site.levels import (
        COLD_PREFIX,
        HOT_PREFIX,
    )
    from ..analysis.heat_exchanger_networks.solver.arrays import (
        nominal_approach_temperature,
    )
    from ..analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
        _calculate_openpinch_targets,
    )
    from .case_derivation import _zone_case

    duties = {}
    for name in case.design._process_zones():
        process = _zone_case(case, case._resolve_target_zone(name))
        root = process._require_prepared_root_zone()
        approach = nominal_approach_temperature(root.config, multiplier)
        # From the zone, not solver arrays: a process of only hot or only
        # cold streams has none.
        hot = tuple(HOT_PREFIX + utility.name for utility in root.hot_utilities)
        cold = tuple(COLD_PREFIX + utility.name for utility in root.cold_utilities)
        periods = _calculate_openpinch_targets(
            process, dTmin=approach, hot_utilities=hot, cold_utilities=cold
        )
        levels: dict[str, list[float]] = {}
        for period in periods:
            for names, targets in (
                (hot, period.hot_utility_level_targets),
                (cold, period.cold_utility_level_targets),
            ):
                for identity, duty in zip(names, targets, strict=True):
                    levels.setdefault(identity, []).append(float(duty))
        duties[name] = levels
    return duties


def move_seed(case: "PinchProblem", network):
    """A site seed for ``case``: ``network``'s recovery matches, with each
    process's level duties re-targeted at ``case``'s levels."""
    from ..analysis.heat_exchanger_networks.site.seeds import _multiplier, _recovery
    from ..domain.heat_exchanger_network import HeatExchangerNetwork

    multiplier = _multiplier(network)
    return HeatExchangerNetwork(
        exchangers=tuple(_recovery(network)),
        run_id="site-level-move",
        method="site_pinch_design",
        stage_count=network.stage_count,
        summary_metrics={"dt_cont_multiplier": multiplier, "site_seed": "level_move"},
        source_metadata={"site_processes": level_duties(case, multiplier)},
    )


def refine_levels(
    case: "PinchProblem",
    view,
    *,
    evaluate: Callable[["PinchProblem", Any], Any],
    step: float,
    step_min: float,
    levels: Sequence[str] | None = None,
    expired: Callable[[], bool] | None = None,
):
    """Move ``case``'s placed levels while the site cost falls.

    Each round tries every level of ``levels`` (default: every placed
    level) up and down by ``step``; ``evaluate(shifted, view)`` designs the
    shifted case from the current view (a move is seeded with its selected
    network's recovery matches). The unshifted levels are first
    evaluated the same way, as the baseline the moves must beat (the
    start's network may come from a seed no move builds). The cheapest
    move below the current cost is adopted; with none, the step halves,
    and the moves stop once it is below ``step_min``. Levels already
    designed from the same seed are not designed again: a move back to them
    is ``"rejected"`` with message ``"already designed"``; a failed design
    may be tried again. Before each design ``expired()``
    is checked; once it is true the moves end with a ``"stopped"`` entry
    (message ``"time limit"``) and the best levels so far. A missing synthesis
    solver or dependency is raised; any other failure of a design is
    recorded. Returns the final case and view, or the start's when no
    adopted move costs less than it, and every move tried.
    """
    from ..analysis.heat_exchanger_networks.solver.dependencies import (
        MissingSynthesisDependencyError,
        MissingSynthesisSolverError,
    )

    levels = movable_levels(case) if levels is None else tuple(levels)
    log: list[LevelMove] = []
    start = (case, view, site_cost(view.selected_network))
    if not (step > 0 and step >= step_min):
        return case, view, ()

    def stopped() -> bool:
        if expired is not None and expired():
            log.append(LevelMove("", 0.0, None, "stopped", "time limit"))
            return True
        return False

    if stopped():
        return case, view, tuple(log)
    designed = {(_level_key(case), _seed_key(view))}
    # The start's network may come from a seed no move builds, so moves are
    # judged against the unshifted levels evaluated the way a move is.
    try:
        baseline = evaluate(case, view)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        raise
    except (ValueError, RuntimeError) as exc:
        log.append(LevelMove("", 0.0, None, "failed", str(exc)))
        cost = start[2]
    else:
        cost = site_cost(baseline.selected_network)
        log.append(LevelMove("", 0.0, cost, "baseline"))
    halted = False
    while step > 0 and step >= step_min and not halted:
        best = None
        for level in levels:
            for shift in (step, -step):
                shifted = shifted_case(case, level, shift)
                if shifted is None:
                    continue
                key = (_level_key(shifted), _seed_key(view))
                if key in designed:
                    log.append(
                        LevelMove(level, shift, None, "rejected", "already designed")
                    )
                    continue
                if stopped():
                    halted = True
                    break
                try:
                    moved = evaluate(shifted, view)
                except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
                    raise
                except (ValueError, RuntimeError) as exc:
                    log.append(LevelMove(level, shift, None, "failed", str(exc)))
                    continue
                designed.add(key)
                moved_cost = site_cost(moved.selected_network)
                log.append(LevelMove(level, shift, moved_cost, "rejected"))
                if best is None or moved_cost < best[2]:
                    best = (len(log) - 1, shifted, moved_cost, moved)
            if halted:
                break
        if best is not None and best[2] < cost - _GAIN * abs(cost):
            index, case, cost, view = best
            log[index] = replace(log[index], status="adopted")
        elif not halted:
            step /= 2
    if start[2] <= cost:
        return start[0], start[1], tuple(log)
    return case, view, tuple(log)


def refine_selected(runs, *, evaluate, step: float, step_min: float, expired=None):
    """``runs`` with the cheapest designed count refined by level moves.

    Returns the runs, the refined one replaced by its moved case, view and
    cost, and the moves tried.
    """
    from .site_levels import select_level_count

    selected = select_level_count(runs)
    case, view, log = refine_levels(
        selected.case,
        selected.view,
        evaluate=evaluate,
        step=step,
        step_min=step_min,
        expired=expired,
    )
    refined = replace(
        selected,
        case=case,
        view=view,
        total_annual_cost=site_cost(view.selected_network),
    )
    return tuple(refined if run is selected else run for run in runs), log


__all__ = [
    "LevelMove",
    "level_duties",
    "movable_levels",
    "move_seed",
    "refine_levels",
    "refine_selected",
    "shifted_case",
]

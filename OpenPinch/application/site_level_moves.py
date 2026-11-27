"""Level-temperature moves on a placed total site design.

A placed level is a main: a hot and a cold utility of one name
(``site_levels.placed_case``). A move shifts both by a step, keeping every
placed level in order, apart from the others, and within the site's stream
temperatures. The site is then designed again at the shifted level, each
process holding its levels re-targeted there, and the move is adopted only
on a lower site total annual cost. With no improving move the step halves,
down to its minimum. The structure is not fixed by the move itself: the
re-design starts from the current best network's recovery matches and
staged heaters and coolers.
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
    cost), ``"failed"`` (the design raised), ``"refused"`` (the levels would
    meet or leave the streams; not designed) or ``"skipped"`` (the levels
    were designed already: the cached cost, or no cost after a failure, is
    logged). The first entry is the
    unshifted levels evaluated as a move (``"baseline"``, level ``""`` and
    shift 0, or ``"failed"``); a last ``"start kept"`` entry records that the
    sweep's own network was returned over adopted moves."""

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
    return _shifted(case, level, shift)[0]


def _shifted(
    case: "PinchProblem", level: str, shift: float
) -> "tuple[PinchProblem | None, str | None]":
    """``shifted_case``'s result, and why the move is refused (or ``None``)."""
    source = case.to_problem_json()
    by_level: dict[str, list] = {}
    for utility in source["utilities"]:
        if _PLACED.fullmatch(utility["name"]):
            by_level.setdefault(utility["name"], []).append(utility)
    order = sorted(by_level, key=lambda name: _span(by_level[name]))
    for utility in by_level[level]:
        for key in ("t_supply", "t_target"):
            utility[key] = _shift(utility[key], shift)
    for below, above in zip(order, order[1:]):
        if _span(by_level[above])[0] <= _span(by_level[below])[1]:
            return None, f"{below} and {above} would touch or cross"
    streams = [
        t
        for stream in source["streams"]
        for key in ("t_supply", "t_target")
        for t in _values(stream[key])
    ]
    low, high = _span(by_level[level])
    if low < min(streams) or high > max(streams):
        return None, (
            f"{level} would leave the stream temperatures "
            f"({min(streams):g} to {max(streams):g} C)"
        )
    return type(case)(source, project_name=case.project_name), None


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
    """A site seed for ``case``: ``network``'s recovery matches and staged
    heaters and coolers (as a site seed keeps them, so a stream keeps a
    stage for each level between its matches), with each process's level
    duties re-targeted at ``case``'s levels."""
    from ..analysis.heat_exchanger_networks.site.seeds import (
        _multiplier,
        _recovery,
        _staged,
    )
    from ..domain.heat_exchanger_network import HeatExchangerNetwork

    multiplier = _multiplier(network)
    return HeatExchangerNetwork(
        exchangers=tuple(_recovery(network) + _staged(network)),
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
    network's ``move_seed``). The unshifted levels are first
    evaluated the same way, as the baseline the moves must beat (the
    start's network may come from a seed no move builds). The cheapest
    move below the current cost is adopted; with none, the step halves,
    and the moves stop once it is below ``step_min``. Levels already
    designed (the full tuple of level positions, whatever the seed) are not
    designed again: a move to them is ``"skipped"`` with the cached cost and
    message ``"already designed"``, or, when their design failed, no cost
    and ``"already failed: <message>"``. The cached cost cannot win: every
    designed cost is at least the current one. A move ``shifted_case``
    refuses is logged ``"refused"`` with the reason. Before each design
    ``expired()``
    is checked; once it is true the moves end with a ``"stopped"`` entry
    (message ``"time limit"``) and the best levels so far. A missing synthesis
    solver or dependency is raised; any other failure of a design is
    recorded. Returns the final case and view (the baseline re-design when
    no move beats it), or the start's when it costs no more, and every move
    tried; when the start is kept over an adopted move, the log ends with a
    ``"start kept"`` entry at the start's cost.
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
    # Each designed level position: its site cost, or the failure message.
    designed: dict[tuple, tuple[float | None, str | None]] = {}
    # The start's network may come from a seed no move builds, so moves are
    # judged against the unshifted levels evaluated the way a move is.
    try:
        baseline = evaluate(case, view)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        raise
    except (ValueError, RuntimeError) as exc:
        log.append(LevelMove("", 0.0, None, "failed", str(exc)))
        designed[_level_key(case)] = (None, str(exc))
        cost = start[2]
    else:
        cost = site_cost(baseline.selected_network)
        log.append(LevelMove("", 0.0, cost, "baseline"))
        # The re-design stands for the unshifted levels from here: it is
        # returned when nothing beats it, and moves are seeded from it.
        view = baseline
        designed[_level_key(case)] = (cost, None)
    halted = False
    while step > 0 and step >= step_min and not halted:
        best = None
        for level in levels:
            for shift in (step, -step):
                shifted, refusal = _shifted(case, level, shift)
                if shifted is None:
                    log.append(LevelMove(level, shift, None, "refused", refusal))
                    continue
                key = _level_key(shifted)
                if key in designed:
                    cached, failure = designed[key]
                    message = (
                        "already designed"
                        if failure is None
                        else f"already failed: {failure}"
                    )
                    log.append(LevelMove(level, shift, cached, "skipped", message))
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
                    designed[key] = (None, str(exc))
                    continue
                moved_cost = site_cost(moved.selected_network)
                designed[key] = (moved_cost, None)
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
        if any(move.status == "adopted" for move in log):
            log.append(LevelMove("", 0.0, start[2], "start kept"))
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

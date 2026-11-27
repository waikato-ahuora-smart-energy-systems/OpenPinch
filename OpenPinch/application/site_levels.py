"""A total site network design over a sweep of utility level counts.

Step 1 of total site heat exchanger network synthesis: for each level count,
utility placement places that many levels for the site, the site is designed
with them, and the cheapest design over the counts is selected. A count whose
placement is infeasible, or whose design fails, is recorded and skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .problem import PinchProblem

# A count needs two isothermal levels: utility placement places no fewer.
_MIN_ISOTHERMAL = 2
# Generated level names, after their side: ``hot_iso_1``, ``cold_sensible_2``.
_PLACED_KINDS = ("iso_", "sensible_")
# Placed levels whose supply and target both lie closer than this (K) are
# one main in effect; synthesis treats a utility spanning less as isothermal
# (``MINIMUM_SLICED_RANGE``).
MINIMUM_LEVEL_GAP = 1.0


@dataclass(frozen=True)
class LevelCountRun:
    """One level count of the sweep and what became of it.

    ``status`` is ``"designed"``, ``"infeasible"`` (no placement),
    ``"failed"`` (the design raised) or ``"not run"`` (after an early stop).
    """

    count: int
    isothermal: int
    sensible: int
    status: str
    view: Any = None
    total_annual_cost: float | None = None
    message: str | None = None
    # The case the count was designed on, for level moves.
    case: Any = None

    def row(self) -> dict[str, Any]:
        """The run as a JSON-safe record."""
        return {
            "count": self.count,
            "isothermal": self.isothermal,
            "sensible": self.sensible,
            "status": self.status,
            "total_annual_cost": self.total_annual_cost,
            "message": self.message,
        }


def _sensible_utilities(problem: "PinchProblem") -> int:
    """How many sensible levels the case has on its busier side.

    A utility with segments or a profile, or whose temperature changes by
    more than synthesis's ``MINIMUM_SLICED_RANGE`` (1 K) in some period, is
    sensible: steam's nominal 0.1 K span is isothermal. A ``Both`` utility
    counts on each side. Each placed level serves both sides, so the count
    is the larger side's, as utility placement infers its counts.
    """
    from ..analysis.heat_exchanger_networks.solver.utility_slices import (
        MINIMUM_SLICED_RANGE,
    )
    from ..contracts.utility_placement import (
        UtilityLevelKind,
        UtilityPlacementOptions,
    )
    from ..domain.enums import StreamType
    from .utility_placement import _utility_kind

    options = UtilityPlacementOptions(
        default_isothermal_span={"value": MINIMUM_SLICED_RANGE, "unit": "delta_degC"}
    )
    config = problem._require_prepared_root_zone().config
    sensible = [
        utility.type
        for utility in problem.validate().utilities
        if utility.active
        and utility.name.strip().casefold() not in {"hu", "cu"}
        and _utility_kind(utility, options=options, config=config)
        is UtilityLevelKind.SENSIBLE
    ]
    return max(
        sum(kind in (side, StreamType.Both.value) for kind in sensible)
        for side in (StreamType.Hot.value, StreamType.Cold.value)
    )


def level_split(problem: "PinchProblem", count: int) -> tuple[int, int]:
    """``count`` levels as (isothermal, sensible), after the case's utilities.

    The case's sensible utilities stay sensible while two isothermal levels
    remain; the rest of the count is isothermal.
    """
    if count < _MIN_ISOTHERMAL:
        raise ValueError(f"a level count must be at least {_MIN_ISOTHERMAL}.")
    sensible = min(_sensible_utilities(problem), count - _MIN_ISOTHERMAL)
    return count - sensible, sensible


def placed_case(
    problem: "PinchProblem", count: int, *, placement=None
) -> "PinchProblem":
    """The case with ``count`` site levels placed by utility placement.

    Each placed level is a generated pair, serving heating and cooling. Its
    hot and cold utility are renamed to one name (``hot_iso_1`` and
    ``cold_iso_1`` to ``iso_1``), so the site accounts the pair as one main
    (``../analysis/heat_exchanger_networks/site/levels.py``). Placement runs
    over every period of the site.

    Where placement fell back on its own ``HU`` or ``CU`` (beyond the site's
    temperatures, priced as the case's utility nearest it), the case keeps
    that utility of its own instead, at its own temperatures and price; a
    side with none raises ``NoFeasiblePlacementError``. A placed level
    whose supply and target both lie within ``MINIMUM_LEVEL_GAP`` of an
    earlier one's is dropped: the two are one main. The case always keeps
    its own utility that rejects heat to ambient (``_ambient_sink``), since
    every rejected heat ends there.
    """
    from .case_derivation import with_utilities_from

    isothermal, sensible = level_split(problem, count)
    placed = problem.target.utility_placement(
        isothermal=isothermal,
        sensible=sensible,
        period_ids=tuple(problem.period_ids),
        options=placement,
    )
    source = with_utilities_from(problem, placed).to_problem_json()
    fallbacks = _fallback_utilities(placed)
    utilities = [
        utility
        for utility in source["utilities"]
        if (utility["name"], utility["type"]) not in fallbacks
    ]
    for utility in utilities:
        side, _, name = utility["name"].partition("_")
        if side == utility["type"].casefold() and name.startswith(_PLACED_KINDS):
            utility["name"] = name
    source["utilities"] = _distinct_levels(utilities) + _stand_ins(problem, fallbacks)
    for sink in _ambient_sinks(problem):
        if not any(
            (u["name"], u["type"]) == (sink["name"], sink["type"])
            for u in source["utilities"]
        ):
            source["utilities"].append(sink)
    return type(problem)(source, project_name=problem.project_name)


def _ambient_sinks(problem: "PinchProblem") -> list[dict]:
    """The case's own utilities that reject heat to ambient, as given.

    Every rejected heat ends at ambient, so a placed case keeps them whatever
    the placed levels are: in each period, the coldest given cold utility
    whose supply is not below ``ENV_TEMPERATURE`` (with none, the warmest),
    as the utility pricing takes it (``_problem/input/utilities.py``).
    """
    from ..analysis.targeting.utilities import is_default_utility

    root = problem._require_prepared_root_zone()
    t_env = float(root.config.environment.temperature)
    periods = max(1, len(problem.period_ids))

    def supply(utility, n: int) -> float:
        value = utility.supply_temperature.to("degC")
        return float(value[n].value if value.num_periods > 1 else value.value)

    given = [u for u in root.cold_utilities if not is_default_utility(u)]
    chosen: list[str] = []
    for n in range(periods):
        above = [u for u in given if supply(u, n) >= t_env]
        pick = (
            min(above, key=lambda u: supply(u, n))
            if above
            else max(given, key=lambda u: supply(u, n), default=None)
        )
        if pick is not None and pick.name not in chosen:
            chosen.append(pick.name)
    raw = problem.to_problem_json()["utilities"]
    return [
        u
        for name in chosen
        for u in raw
        if u["name"] == name and u["type"] in ("Cold", "Both")
    ]


def _fallback_utilities(
    placed,
) -> dict[tuple[str, str], tuple[tuple[str | None, float], ...]]:
    """Placement's fallback utilities, ``(name, type)`` to each period's
    ``(period_id, supply)`` (degC): a multiperiod placement may place one
    fallback at a different supply in each period."""
    from .utility_placement import _level_temperature

    supplies: dict[tuple[str, str], set[tuple[str | None, float]]] = {}
    for row in placed.utility_placement_result.retarget_comparison:
        if row.is_fallback:
            supplies.setdefault((row.utility, row.side.value.capitalize()), set()).add(
                (row.period_id, float(_level_temperature(row.supply_temperature)))
            )
    return {
        key: tuple(sorted(values, key=lambda item: (item[1], str(item[0]))))
        for key, values in supplies.items()
    }


def _distinct_levels(utilities: list[dict]) -> list[dict]:
    """``utilities`` without each placed level whose temperatures all lie
    within ``MINIMUM_LEVEL_GAP`` of an earlier placed level's."""
    from .site_level_moves import _PLACED, _values

    kept: dict[str, list[float]] = {}
    dropped: set[str] = set()
    for utility in utilities:
        name = utility["name"]
        if not _PLACED.fullmatch(name) or name in kept or name in dropped:
            continue
        # Sorted, so a pair's hot and cold utility give the same ends.
        ends = sorted(
            _degc(utility[key], value)
            for key in ("t_supply", "t_target")
            for value in _values(utility[key])
        )
        if any(
            len(other) == len(ends)
            and all(abs(a - b) < MINIMUM_LEVEL_GAP for a, b in zip(other, ends))
            for other in kept.values()
        ):
            dropped.add(name)
        else:
            kept[name] = ends
    return [utility for utility in utilities if utility["name"] not in dropped]


def _degc(quantity: Mapping, value: float) -> float:
    from ..domain.value import Value

    return Value(value, quantity.get("unit") or "degC").to("degC").value


def _stand_ins(
    problem: "PinchProblem",
    fallbacks: Mapping[tuple[str, str], Sequence[tuple[str | None, float]]],
) -> list[dict]:
    """The case's own utility for each fallback: its side's nearest by
    supply temperature in the period placement placed it, as placement
    priced the fallback, for each period's supply (their union). A period
    the case does not name is matched at its first period."""
    from ..analysis.utility_placement.errors import NoFeasiblePlacementError
    from .utility_placement import _resolve_placement_zone, _utility_sources

    zone = _resolve_placement_zone(problem, None)
    raw = problem.to_problem_json()["utilities"]
    periods = problem.period_ids
    # By name and type: a hot and a cold utility may share a name, and a
    # ``Both`` utility serving both fallbacks is kept once.
    kept: dict[tuple[str, str], dict] = {}
    for (name, side), supplies in fallbacks.items():
        for period_id, supply in supplies:
            sources = _utility_sources(
                problem, zone, raw, period_idx=periods.get(period_id)
            )
            own = [
                source
                for source in sources[side]
                if source[0].strip().casefold() not in {"hu", "cu"}
            ]
            if not own:
                raise NoFeasiblePlacementError(
                    code="no_fallback_utility",
                    message=(
                        f"placement needs its fallback {name}, and the case has "
                        f"no {side.casefold()} utility of its own to take its "
                        "place."
                    ),
                )
            nearest = min(own, key=lambda source: abs(source[1] - supply))
            utility = nearest[3]
            kept.setdefault((utility["name"], utility["type"]), utility)
    return list(kept.values())


def site_cost(network) -> float:
    """A site design's total annual cost: the site's when it reports one."""
    site = (network.summary_metrics or {}).get("site_total_annual_cost")
    return float(network.total_annual_cost if site is None else site)


def run_level_counts(
    problem: "PinchProblem",
    counts: Sequence[int],
    *,
    design: Callable[["PinchProblem"], Any],
    place: Callable[..., "PinchProblem"] = placed_case,
    early_stop: bool = False,
    placement=None,
    expired: Callable[[], bool] | None = None,
) -> tuple[LevelCountRun, ...]:
    """Design the site once per level count, in order.

    ``place(problem, count, placement=)`` gives the case for a count and
    ``design(case)`` its design view. A missing synthesis solver or
    dependency is raised; any other failure is recorded with its count. With
    ``early_stop``, the sweep ends once two designed counts in a row cost
    more than the cheapest before them. Once a count is designed, a count
    is started only while ``expired()`` is false; the rest are ``"not run"``
    with message ``"time limit"``.
    """
    from ..analysis.heat_exchanger_networks.solver.dependencies import (
        MissingSynthesisDependencyError,
        MissingSynthesisSolverError,
    )
    from ..analysis.utility_placement.errors import UtilityPlacementError

    runs: list[LevelCountRun] = []
    best = float("inf")
    rises = 0
    for count in counts:
        isothermal, sensible = level_split(problem, count)
        split = {"count": count, "isothermal": isothermal, "sensible": sensible}
        if early_stop and rises >= 2:
            runs.append(LevelCountRun(**split, status="not run"))
            continue
        if expired is not None and best < float("inf") and expired():
            runs.append(LevelCountRun(**split, status="not run", message="time limit"))
            continue
        try:
            case = place(problem, count, placement=placement)
        except UtilityPlacementError as exc:
            runs.append(LevelCountRun(**split, status="infeasible", message=str(exc)))
            continue
        try:
            view = design(case)
        except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
            raise
        except (ValueError, RuntimeError) as exc:
            runs.append(LevelCountRun(**split, status="failed", message=str(exc)))
            continue
        cost = site_cost(view.selected_network)
        runs.append(
            LevelCountRun(
                **split,
                status="designed",
                view=view,
                total_annual_cost=cost,
                case=case,
            )
        )
        rises = rises + 1 if cost > best else 0
        best = min(best, cost)
    return tuple(runs)


def select_level_count(runs: Sequence[LevelCountRun]) -> LevelCountRun:
    """The designed count of least cost; the first such on a tie."""
    from ..analysis.heat_exchanger_networks.errors import WorkflowContractError

    designed = [run for run in runs if run.status == "designed"]
    if not designed:
        raise WorkflowContractError(
            "no level count gave a site design: "
            + "; ".join(f"{run.count}: {run.status} ({run.message})" for run in runs)
        )
    return min(designed, key=lambda run: run.total_annual_cost)


def level_sweep_view(problem: "PinchProblem", runs, *, settings: Mapping):
    """The selected count's view, holding every count's networks.

    The result ranks every network of each designed count by site cost,
    each tagged with ``summary_metrics["utility_level_count"]``, and selects
    the cheapest. Its provenance names the problem's root zone and records
    ``settings`` with the sweep (``HENS_SITE_LEVEL_SWEEP``, one row per
    count). It is not committed as the problem's design: its networks name
    the placed case's utilities, which the problem lacks.
    """
    from ..analysis.heat_exchanger_networks.results.selection import (
        ranked_networks,
        select_network,
    )
    from ._problem.targeting.provenance import make_provenance

    selected = select_level_count(runs)
    ranked = []
    for run in runs:
        if run.status != "designed":
            continue
        for outcome in ranked_networks(run.view.result):
            network = outcome.network.model_copy(
                update={
                    "summary_metrics": dict(outcome.network.summary_metrics)
                    | {"utility_level_count": run.count}
                }
            )
            ranked.append(
                outcome.model_copy(
                    update={"network": network, "objective_value": site_cost(network)}
                )
            )
    result = selected.view.result
    result.provenance = make_provenance(
        problem,
        "design.total_site_heat_exchanger_network",
        problem._require_prepared_root_zone(),
        period_ids=tuple(problem.period_ids),
        settings=dict(settings)
        | {"HENS_SITE_LEVEL_SWEEP": [run.row() for run in runs]},
    )
    result.ranked_networks = tuple(ranked)
    select_network(result)
    # Not committed as the problem's design: the networks name the placed
    # case's utilities, which the problem lacks.
    return selected.view


__all__ = [
    "LevelCountRun",
    "level_split",
    "level_sweep_view",
    "placed_case",
    "run_level_counts",
    "select_level_count",
    "site_cost",
]

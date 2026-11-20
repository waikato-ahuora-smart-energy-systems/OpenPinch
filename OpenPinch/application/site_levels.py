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
    for utility in source["utilities"]:
        side, _, name = utility["name"].partition("_")
        if side == utility["type"].casefold() and name.startswith(_PLACED_KINDS):
            utility["name"] = name
    return type(problem)(source, project_name=problem.project_name)


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
    count). The result is committed to the problem's design results when
    it has results.
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
    if problem._results is not None:
        problem._results.design = result
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

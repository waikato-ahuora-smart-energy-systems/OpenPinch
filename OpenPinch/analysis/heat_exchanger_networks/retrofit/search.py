"""Retrofit search: one modification at a time on an existing network.

Derived from the network evolution method: each depth adds one
modification (a new match or added area on an existing unit) to every
network in a beam, solves each new structure once with the fixed-structure
model's "retrofit" goal, and keeps the best. The best design at each depth
makes the frontier over the number of steps.
"""

from __future__ import annotations

import math
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from ....contracts.synthesis.retrofit import (
    HeatExchangerNetworkRetrofitDesign,
    HeatExchangerNetworkRetrofitModification,
    HeatExchangerNetworkRetrofitResult,
)
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ...economics import compute_capital_recovery_factor
from ...parallelism import supported_workers
from ..context import ensure_target_results
from ..duty_optimisation import (
    DEFAULT_FEASIBILITY_APPROACH,
    _structural_exchangers,
    build_duty_optimisation_request,
    build_solution_network,
    fixed_network_structure,
    verify_solution,
)
from ..execution.settings import configured_stage_solver, stage_solver_options
from .frontier import mark_dominated
from .moves import (
    NewMatch,
    NewUtility,
    apply_new_match,
    apply_new_utility,
    apply_relocation,
    match_potential,
    new_splits,
    new_utility_candidates,
    screen_new_matches,
)

# Added area below this (m2) is no modification.
_MIN_ADDED_AREA = 1e-3
# An existing unit needing this share of its widened area is at its cap.
_BINDING_SHARE = 0.99
# The structure strategy keeps a unit's added area only from this share of
# its installed area, and at least this many m2.
_MIN_KEPT_SHARE = 0.01
_MIN_KEPT_AREA = 1.0
STRATEGIES = ("stepwise", "structure")


@dataclass(frozen=True)
class RetrofitSettings:
    """Search limits and retrofit fixed costs.

    ``max_modifications`` is the search depth, ``beam_width`` the designs kept
    per depth and ``new_match_candidates`` the screened new matches tried from
    each. ``area_tolerance`` widens every installed area (a fraction) and
    ``approach`` is the minimum approach (K) on every exchanger.
    ``max_added_area_fraction`` caps the area added to an existing exchanger
    as a fraction of its installed area (None: no cap).
    ``new_unit_cost`` ($ per new exchanger), ``area_addition_cost`` ($ per
    existing exchanger given more area) and ``split_cost`` ($ per stream a new
    match splits) add to the area cost (the ``COSTING_HX_AREA_*`` law on the
    added area). ``relocation_cost`` ($ per existing recovery exchanger moved
    to another match or position, plus ``split_cost`` per new split) enables
    relocation moves (all ranked by ceiling in the structure search, the
    ``relocation_candidates`` best scored in the stepwise one);
    ``None`` leaves them out. ``new_utility_units`` also tries each new match
    and relocation with each new end heater or cooler the network lacks, in
    the same step, priced as one more new unit.

    ``strategy`` "structure" (the default) makes one structural change per step
    (new match, relocation, paired utility): ``ceiling_candidates`` screened
    changes are ranked by the utility cost their structure reaches with
    unlimited area, the ``new_match_candidates`` best are solved with every
    existing exchanger free to grow, and a unit keeps added area of at least
    1 % of its installed area (1 m2 or more). "stepwise" makes one
    modification per step, an area addition being one.
    """

    max_modifications: int = 3
    beam_width: int = 3
    new_match_candidates: int = 4
    area_tolerance: float = 0.02
    max_added_area_fraction: float | None = None
    approach: float = DEFAULT_FEASIBILITY_APPROACH
    new_unit_cost: float = 0.0
    area_addition_cost: float = 0.0
    split_cost: float = 0.0
    relocation_cost: float | None = None
    relocation_candidates: int = 8
    new_utility_units: bool = False
    strategy: str = "structure"
    ceiling_candidates: int = 12


@dataclass(frozen=True)
class _Move:
    kind: str
    exchanger_id: str
    match: NewMatch | None = None
    splits: tuple[str, ...] = ()
    utility_id: str | None = None


@dataclass(frozen=True)
class _State:
    network: HeatExchangerNetwork
    enlarged: frozenset[str]
    moves: tuple[_Move, ...]


@dataclass(frozen=True)
class _Evaluated:
    state: _State
    design: HeatExchangerNetworkRetrofitDesign
    solved: HeatExchangerNetwork
    binding: tuple[str, ...]
    added: dict[str, float] = field(default_factory=dict)


def run_retrofit_search(
    problem, existing: HeatExchangerNetwork, settings: RetrofitSettings
) -> HeatExchangerNetworkRetrofitResult:
    """Search modifications of ``existing``, one at a time.

    Every exchanger of ``existing`` needs its installed ``area``; a
    ``ValueError`` names those without one. ``max_added_area_fraction`` must
    be None or finite and non-negative.
    """

    if settings.strategy not in STRATEGIES:
        raise ValueError(
            f"strategy must be one of {STRATEGIES}; got {settings.strategy!r}."
        )
    fraction = settings.max_added_area_fraction
    if fraction is not None and not (math.isfinite(fraction) and fraction >= 0.0):
        raise ValueError(
            "max_added_area_fraction must be None or a finite fraction of at "
            f"least 0; got {fraction!r}."
        )
    missing = [
        str(e.exchanger_id)
        for e in _structural_exchangers(existing)
        if not (e.area or 0.0) > 0.0
    ]
    if missing:
        raise ValueError(
            "retrofit needs the installed area of every existing exchanger; "
            "none on " + ", ".join(missing) + "."
        )
    ids = [str(e.exchanger_id) for e in _structural_exchangers(existing)]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(
            "retrofit needs a unique id on every existing exchanger; "
            "repeated: " + ", ".join(duplicates) + "."
        )
    ensure_target_results(problem, {}, None)
    evaluator = _Evaluator(problem, existing, settings)
    base = evaluator.evaluate(_State(existing, frozenset(), ()))
    if base is None:
        raise ValueError(
            "the existing network does not meet its targets at its installed "
            "areas; check the areas or raise the area tolerance."
        )
    frontier = [base.design]
    beam = [base]
    seen = {_signature(base.state, evaluator.existing)}
    structural = settings.strategy == "structure"
    for _depth in range(max(0, int(settings.max_modifications))):
        children = []
        for parent in beam:
            states = []
            for state in _children(parent, settings, evaluator):
                if structural:
                    state = _structure_only(state)
                signature = _signature(state, evaluator.existing)
                if signature in seen:
                    continue
                seen.add(signature)
                states.append(state)
            if structural:
                states = evaluator.best_ceilings(states)
            solve = evaluator.sized if structural else evaluator.evaluate
            children.extend(
                evaluated
                for evaluated in evaluator.map(solve, states)
                if evaluated is not None
            )
        if not children:
            break
        children.sort(key=lambda child: -child.design.net_annual_benefit)
        beam = children[: max(1, int(settings.beam_width))]
        frontier.append(beam[0].design)
    return HeatExchangerNetworkRetrofitResult(
        frontier=mark_dominated(frontier), evaluated_designs=evaluator.evaluations
    )


def _children(parent: _Evaluated, settings: RetrofitSettings, evaluator):
    """Each one-modification step from ``parent``: added area on each
    existing unit at its cap, the screened new matches, then (with a
    ``relocation_cost``) each existing recovery unit not yet moved that
    shares a stream with a screened match, moved there. With
    ``new_utility_units`` each new match and relocation is also tried with
    each new end utility exchanger the network lacks."""

    state = parent.state
    structural = settings.strategy == "structure"
    if structural:
        settings = replace(settings, new_match_candidates=settings.ceiling_candidates)
    for exchanger_id in () if structural else parent.binding:
        yield replace(
            state,
            enlarged=state.enlarged | {exchanger_id},
            moves=state.moves + (_Move("area_addition", exchanger_id),),
        )
    for network, move in _structural_steps(parent, settings, evaluator):
        yield replace(state, network=network, moves=state.moves + (move,))
        if not settings.new_utility_units:
            continue
        for utility in new_utility_candidates(network, **evaluator.utility_axes):
            utility_id = _new_utility_id(network, utility)
            yield replace(
                state,
                network=apply_new_utility(network, utility, utility_id),
                moves=state.moves + (replace(move, utility_id=utility_id),),
            )


def _structural_steps(parent: _Evaluated, settings: RetrofitSettings, evaluator):
    """The screened new matches and, with a ``relocation_cost``, the
    relocations from ``parent``, each as ``(network, move)``."""

    network = parent.state.network
    matches = screen_new_matches(
        parent.solved,
        evaluator.ends,
        approach=settings.approach,
        limit=settings.new_match_candidates,
    )
    for match in matches:
        exchanger_id = _new_id(network, match)
        move = _Move("new_match", exchanger_id, match, new_splits(network, match))
        yield apply_new_match(network, match, exchanger_id), move
    if settings.relocation_cost is None:
        return
    for unit, match in _relocations(
        parent, evaluator.ends, settings, evaluator.existing
    ):
        remaining = network.model_copy(
            update={
                "exchangers": tuple(
                    e for e in network.exchangers if e.exchanger_id != unit
                )
            }
        )
        move = _Move("relocation", unit, match, new_splits(remaining, match))
        yield apply_relocation(network, unit, match), move


def _relocations(parent: _Evaluated, ends, settings: RetrofitSettings, existing):
    """The relocations from ``parent``, most promising first.

    Each existing recovery unit not yet moved may go to any screened match
    that shares one of its streams (its own pair elsewhere is a repiping).
    A move scores the match's potential less the unit's present duty, the
    heat it gives up, best first. The stepwise search keeps the
    ``relocation_candidates`` best positive scores; the structure search
    keeps every move, as the ceiling ranking judges a move by what its whole
    structure can reach and the score barely tracks that. A unit that is the
    only exchanger on one of its streams stays on that stream.
    """

    solved = parent.solved
    moved = {m.exchanger_id for m in parent.state.moves if m.kind == "relocation"}
    duties = {
        e.exchanger_id: (e.source_stream, e.sink_stream, e.period_states[0].duty)
        for e in solved.exchangers
        if e.kind is HeatExchangerKind.RECOVERY
        and e.exchanger_id in existing
        and e.exchanger_id not in moved
    }
    served = Counter(
        stream
        for e in parent.state.network.exchangers
        for stream in (e.source_stream, e.sink_stream)
    )
    # Every screened match, not just the best few.
    pool = screen_new_matches(
        solved, ends, approach=settings.approach, limit=sys.maxsize
    )
    scored = []
    for match in pool:
        potential = match_potential(solved, ends, match, approach=settings.approach)
        for unit, (hot, cold, duty) in duties.items():
            stranded = (match.hot != hot and served[hot] == 1) or (
                match.cold != cold and served[cold] == 1
            )
            if (match.hot == hot or match.cold == cold) and not stranded:
                score = potential - duty
                if score > 0.0 or settings.strategy == "structure":
                    scored.append((score, unit, match))
    scored.sort(key=lambda item: -item[0])
    if settings.strategy == "structure":
        return [(unit, match) for _score, unit, match in scored]
    limit = max(0, int(settings.relocation_candidates))
    return [(unit, match) for _score, unit, match in scored[:limit]]


def _new_utility_id(network: HeatExchangerNetwork, utility: NewUtility) -> str:
    """``new:<utility>-><stream>`` for a heater, ``new:<stream>-><utility>``
    for a cooler, with ``:2``, ``:3`` and so on when taken."""

    taken = {e.exchanger_id for e in network.exchangers}
    pair = (
        (utility.utility, utility.stream)
        if utility.heater
        else (utility.stream, utility.utility)
    )
    stem = f"new:{pair[0]}->{pair[1]}"
    exchanger_id, n = stem, 1
    while exchanger_id in taken:
        n += 1
        exchanger_id = f"{stem}:{n}"
    return exchanger_id


def _new_id(network: HeatExchangerNetwork, match: NewMatch) -> str:
    taken = {e.exchanger_id for e in network.exchangers}
    stem = f"new:{match.hot}->{match.cold}"
    exchanger_id, n = stem, 1
    while exchanger_id in taken:
        n += 1
        exchanger_id = f"{stem}:{n}"
    return exchanger_id


def _structure_only(state: _State) -> _State:
    """``state`` without its area additions: the structure strategy sizes
    area afresh for every structure."""

    return replace(
        state,
        enlarged=frozenset(),
        moves=tuple(m for m in state.moves if m.kind != "area_addition"),
    )


def _signature(state: _State, existing):
    """Recovery matches by stage, with the existing unit installed at each
    (new units are alike), the utility exchangers, and the enlarged units:
    one per structure."""

    recovery = frozenset(
        (
            e.source_stream,
            e.sink_stream,
            e.stage,
            e.exchanger_id if e.exchanger_id in existing else None,
        )
        for e in state.network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY
    )
    utilities = frozenset(
        (e.source_stream, e.sink_stream, e.stage)
        for e in state.network.exchangers
        if e.kind is not HeatExchangerKind.RECOVERY
    )
    return recovery, utilities, state.enlarged


class _Evaluator:
    """Solves retrofit structures on one problem's solver arrays."""

    def __init__(self, problem, existing: HeatExchangerNetwork, settings) -> None:
        from ..solver.arrays import problem_to_solver_arrays

        config = problem.master_zone.config
        self.settings = settings
        self.existing = frozenset(
            e.exchanger_id for e in _structural_exchangers(existing)
        )
        self.installed = {
            e.exchanger_id: float(e.area) for e in _structural_exchangers(existing)
        }
        self.arrays = problem_to_solver_arrays(problem, 2.0 * config.thermal.dt_cont)
        hens = config.hens
        self.solver = configured_stage_solver(hens, "HENS_SOLVER_EVM")
        self.solver_options = stage_solver_options(hens, "evm", self.solver)
        self.tol = float(hens.solve_tolerance)
        self.max_parallel = supported_workers(max(1, int(hens.max_parallel)))
        self.names = _existing_names(self.arrays.axis_maps, existing)
        self.ends = {
            self.names[name]: ends for name, ends in _stream_ends(self.arrays).items()
        }
        axes = self.arrays.axis_maps
        self.utility_axes = {
            "hot_streams": tuple(n for n, e in self.ends.items() if e[0] > e[1]),
            "cold_streams": tuple(n for n, e in self.ends.items() if e[0] <= e[1]),
            "hot_utilities": tuple(self.names[n] for n in axes["hot_utilities"]),
            "cold_utilities": tuple(self.names[n] for n in axes["cold_utilities"]),
        }
        self.base_utility_cost: float | None = None
        self.evaluations = 0
        self._count_lock = threading.Lock()
        costing = config.costing
        self.area_coeff = float(costing.hx_area_coeff)
        self.area_exp = float(costing.hx_area_exp)
        self.recovery_factor = float(
            compute_capital_recovery_factor(costing.discount_rate, costing.service_life)
        )

    def evaluate(self, state: _State) -> _Evaluated | None:
        """Solve ``state``; ``None`` if the solve fails."""

        from ..models.fixed_structure import FixedStructureModel

        with self._count_lock:
            self.evaluations += 1
        settings = self.settings
        request = build_duty_optimisation_request(
            state.network, objective="cost", min_approach_temperature=settings.approach
        )
        structure = fixed_network_structure(request, self.arrays.axis_maps)
        exchangers = _structural_exchangers(state.network)
        ids = [exchangers[p].exchanger_id for p in structure.positions]
        enlargeable = {
            kind: frozenset(
                index
                for (slot_kind, index), exchanger_id in zip(structure.slots, ids)
                if slot_kind == kind and exchanger_id in state.enlarged
            )
            for kind in ("recovery", "utility")
        }
        structure = replace(
            structure,
            spec=replace(
                structure.spec,
                area_tolerance=float(settings.area_tolerance),
                max_added_area_fraction=settings.max_added_area_fraction,
                enlargeable_recovery=enlargeable["recovery"],
                enlargeable_utilities=enlargeable["utility"],
            ),
        )
        model = FixedStructureModel(
            name="retrofit",
            solver=self.solver,
            solver_arrays=self.arrays,
            spec=structure.spec,
            minimisation_goal="retrofit",
            default_approach=settings.approach,
            tol=self.tol,
            solver_options=dict(self.solver_options),
        )
        # Read the solved values, then remove the model's run directory
        # (Couenne writes one per model) whether or not it is kept.
        try:
            model.optimise(print_output=False)
            if getattr(model, "mSuccess", 0) != 1:
                return None
            solved = build_solution_network(request, structure, model, self.arrays)
            # The solver accepts loosely converged points: drop a design whose
            # approaches, heat balances or areas fail the duty optimiser's
            # checks, judging installed areas as the model did (widened).
            if verify_solution(structure, model, self._widened(solved)):
                return None
            results = dict(zip(ids, structure.results(model), strict=True))
            widen = 1.0 + float(settings.area_tolerance)
            binding = tuple(
                exchanger_id
                for exchanger_id in ids
                if exchanger_id in self.existing
                and exchanger_id not in state.enlarged
                and results[exchanger_id].active
                and max(p.required_area for p in results[exchanger_id].periods)
                >= _BINDING_SHARE * widen * results[exchanger_id].area
            )
            metrics = solved.summary_metrics
            utility_cost = float(model.utility_cost_value)
            if self.base_utility_cost is None:
                self.base_utility_cost = utility_cost
            savings = self.base_utility_cost - utility_cost
            stages = {e.exchanger_id: e.stage for e in state.network.exchangers}
            modifications = tuple(
                self._modification(
                    move,
                    results[move.exchanger_id],
                    stages[move.exchanger_id],
                    results.get(move.utility_id),
                )
                for move in state.moves
            )
            if any(m is None for m in modifications):
                return None
            capital = sum(m.fixed_cost + m.area_cost for m in modifications)
            annualised = self.recovery_factor * capital
            design = HeatExchangerNetworkRetrofitDesign(
                modifications=modifications,
                network=solved,
                hot_utility=float(metrics["hot_utility_load"]),
                cold_utility=float(metrics["cold_utility_load"]),
                utility_cost=utility_cost,
                annual_savings=savings,
                capital_cost=capital,
                annualised_capital_cost=annualised,
                net_annual_benefit=savings - annualised,
                payback_years=capital / savings if savings > 0.0 else None,
            )
            added = {i: float(r.added_area) for i, r in results.items()}
            return _Evaluated(
                state, design, _with_names(solved, self.names), binding, added
            )
        finally:
            model.cleanup()

    def map(self, solve, states: list[_State]) -> list:
        """``solve`` applied to each of ``states``, in order: on
        ``HENS_MAX_PARALLEL`` threads when it is above 1 (each solve is an
        external solver subprocess), otherwise serially."""

        if self.max_parallel > 1 and len(states) > 1:
            workers = min(self.max_parallel, len(states))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(solve, states))
        return [solve(state) for state in states]

    def best_ceilings(self, states: list[_State]) -> list[_State]:
        """For each kind of change (new match, relocation), the
        ``new_match_candidates`` states whose structures reach the lowest
        utility cost with unlimited area, best first. Separate quotas keep
        many relocations from crowding out the new matches, and the other
        way round. With ``HENS_MAX_PARALLEL`` above 1 the ceilings are
        solved on that many threads (each solve is an Ipopt subprocess); the
        ranking is the same either way."""

        with self._count_lock:
            self.evaluations += len(states)
        costs = self.map(self.ceiling, states)
        ceilings = [
            (cost, n, state) for n, (cost, state) in enumerate(zip(costs, states))
        ]
        ranked = sorted(
            (item for item in ceilings if item[0] is not None),
            key=lambda item: (item[0], item[1]),
        )
        limit = max(0, int(self.settings.new_match_candidates))
        kept: dict[str, int] = {}
        best = []
        for _cost, _n, state in ranked:
            kind = state.moves[-1].kind
            if kept.get(kind, 0) < limit:
                kept[kind] = kept.get(kind, 0) + 1
                best.append(state)
        return best

    def ceiling(self, state: _State) -> float | None:
        """The utility cost ($/y) ``state``'s structure reaches with
        unlimited area (its network pinch); ``None`` if the solve fails."""

        from ..models.fixed_structure import FixedStructureModel

        request = build_duty_optimisation_request(
            state.network,
            objective="cost",
            min_approach_temperature=self.settings.approach,
        )
        structure = fixed_network_structure(request, self.arrays.axis_maps)
        model = FixedStructureModel(
            name="retrofit-ceiling",
            solver=self.solver,
            solver_arrays=self.arrays,
            spec=replace(structure.spec, recovery_areas={}, utility_areas={}),
            minimisation_goal="utility cost",
            default_approach=self.settings.approach,
            tol=self.tol,
            solver_options=dict(self.solver_options),
        )
        try:
            model.optimise(print_output=False)
            if getattr(model, "mSuccess", 0) != 1:
                return None
            return float(model.utility_cost_value)
        finally:
            model.cleanup()

    def sized(self, state: _State) -> _Evaluated | None:
        """``state``'s structure with its added area sized: solved with every
        existing unit free to grow, then again with only the units whose
        added area is worth keeping, each an area addition."""

        free = self.evaluate(replace(state, enlarged=frozenset(self.installed)))
        if free is None:
            return self.evaluate(state)
        keep = tuple(
            sorted(
                unit
                for unit, added in free.added.items()
                if unit in self.installed
                and added >= max(_MIN_KEPT_AREA, _MIN_KEPT_SHARE * self.installed[unit])
            )
        )
        if not keep:
            return self.evaluate(state)
        sized = self.evaluate(
            replace(
                state,
                enlarged=frozenset(keep),
                moves=state.moves
                + tuple(_Move("area_addition", unit) for unit in keep),
            )
        )
        return sized if sized is not None else self.evaluate(state)

    def _widened(self, network: HeatExchangerNetwork) -> HeatExchangerNetwork:
        """``network`` with each existing unit's installed area widened by
        ``area_tolerance``, as the model rates it."""

        tolerance = float(self.settings.area_tolerance)
        return network.model_copy(
            update={
                "exchangers": tuple(
                    e.model_copy(
                        update={
                            "area": e.area + tolerance * self.installed[e.exchanger_id]
                        }
                    )
                    if e.exchanger_id in self.installed and e.area is not None
                    else e
                    for e in network.exchangers
                )
            }
        )

    def _modification(
        self, move: _Move, result, stage: int | None, utility=None
    ) -> HeatExchangerNetworkRetrofitModification | None:
        """``move`` priced from its unit's added area; ``None`` when it does
        nothing (no added area, or a unit with no duty). A relocation adds no
        area. A new or moved match reports ``stage``, its stage in the
        design's network. ``utility`` is the result of the new utility
        exchanger paired with the move, priced as one more new unit; it too
        must carry duty."""

        if not result.active or (utility is not None and not utility.active):
            return None
        settings = self.settings
        splits = settings.split_cost * len(move.splits)
        if move.kind == "relocation":
            added, fixed = 0.0, settings.relocation_cost + splits
        else:
            added = float(result.added_area)
            if added < _MIN_ADDED_AREA:
                return None
            fixed = (
                settings.new_unit_cost + splits
                if move.kind == "new_match"
                else settings.area_addition_cost
            )
        utility_area = 0.0 if utility is None else float(utility.added_area)
        if utility is not None:
            fixed += settings.new_unit_cost
        match = move.match
        return HeatExchangerNetworkRetrofitModification(
            kind=move.kind,
            exchanger_id=move.exchanger_id,
            added_area=added,
            fixed_cost=float(fixed),
            area_cost=self._area_cost(added) + self._area_cost(utility_area),
            hot_stream=match.hot if match else None,
            cold_stream=match.cold if match else None,
            stage=stage if match else None,
            new_stage=bool(match and match.new_stage),
            new_splits=move.splits,
            utility_exchanger_id=move.utility_id,
            utility_area=utility_area,
        )

    def _area_cost(self, area: float) -> float:
        """Installed cost ($) of ``area`` m2 of new area."""

        return self.area_coeff * area**self.area_exp if area > 0.0 else 0.0


def _existing_names(axis_maps, existing: HeatExchangerNetwork) -> dict[str, str]:
    """Each solver stream name mapped to the name ``existing`` uses for it.

    The existing network may name a stream without its zone prefix
    (``H1`` for ``Process A.H1``); streams it does not name keep theirs.
    """

    used = {
        name for e in existing.exchangers for name in (e.source_stream, e.sink_stream)
    }
    names: dict[str, str] = {}
    for axis in axis_maps.values():
        for key in axis:
            key = str(key)
            matches = [n for n in used if n == key or key.endswith("." + n)]
            names[key] = matches[0] if len(matches) == 1 else key
    return names


def _with_names(network: HeatExchangerNetwork, names) -> HeatExchangerNetwork:
    """``network`` with its stream names mapped through ``names``."""

    return network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(
                    update={
                        "source_stream": names.get(e.source_stream, e.source_stream),
                        "sink_stream": names.get(e.sink_stream, e.sink_stream),
                    }
                )
                for e in network.exchangers
            )
        }
    )


def _stream_ends(arrays) -> dict[str, tuple[float, float, float]]:
    """Each process stream's supply and target temperature and heat capacity
    flowrate, in the first period."""

    data = arrays.arrays
    ends: dict[str, tuple[float, float, float]] = {}
    for side, axis in (("h", "hot_process_streams"), ("c", "cold_process_streams")):
        for name, index in arrays.axis_maps[axis].items():
            ends[str(name)] = (
                float(data[f"T_{side}_in_period"][0][index]),
                float(data[f"T_{side}_out_period"][0][index]),
                float(data[f"f_{side}_period"][0][index]),
            )
    return ends

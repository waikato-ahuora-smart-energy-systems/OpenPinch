"""Retrofit search: one modification at a time on an existing network.

Derived from the network evolution method: each depth adds one
modification (a new match or added area on an existing unit) to every
network in a beam, solves each new structure once with the fixed-structure
model's "retrofit" goal, and keeps the best. The best design at each depth
makes the frontier over the number of steps.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from collections import Counter
from collections.abc import Mapping
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
from ...heat_pumps.common.compressor_efficiency import compressor_efficiency_model
from ...heat_pumps.cycles.limits import CycleLimits
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
from ..solver.arrays import annual_cost_scales
from .frontier import mark_dominated
from .heat_pumps import (
    HeatPumpCandidate,
    carnot_rating,
    condenser_name,
    evaporator_name,
    heat_pump_candidates,
    screen_heat_pumps,
    vapour_compression_capital,
    vapour_compression_rating,
    with_heat_pump_utilities,
)
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
# Area value ($/y per m2) below which more area on a unit is worth nothing.
_MIN_AREA_VALUE = 1e-3
_MIN_KEPT_AREA = 1.0
STRATEGIES = ("stepwise", "structure")
HEAT_PUMP_CYCLES = (None, "carnot", "vapour_compression")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrofitSettings:
    """Search limits and retrofit fixed costs.

    ``max_modifications`` is the search depth, ``beam_width`` the designs kept
    per depth and ``new_match_candidates`` the screened new matches tried from
    each. ``area_tolerance`` widens every installed area (a fraction) and
    ``approach`` is the minimum approach (K) on every exchanger; ``None``
    gives each exchanger the sum of its two sides' temperature contributions
    and screens new matches and heat pumps at twice ``THERMAL_DT_CONT``.
    ``max_added_area_fraction`` caps the area added to an existing exchanger
    as a fraction of its installed area (None: no cap).
    ``new_unit_cost`` ($ per new exchanger), ``area_addition_cost`` ($ per
    existing exchanger given more area) and ``split_cost`` ($ per stream a new
    match splits) add to the area cost (the ``COSTING_HX_AREA_*`` law on the
    added area). ``relocation_cost`` ($ per existing recovery exchanger moved
    to another match or position, plus ``split_cost`` per new split) enables
    relocation moves (in the structure search up to
    ``ceiling_relocations`` per design, the best scored first, ranked by
    ceiling; in the stepwise one the ``relocation_candidates`` best scored);
    ``None`` leaves them out. ``new_utility_units`` also tries each new match
    and relocation with each new end heater or cooler the network lacks, in
    the same step, priced as one more new unit. ``enhancement_factors``
    (existing exchanger id to the factor its heat transfer coefficient can
    be raised by, tube inserts for example; None: no enhancement) enables
    enhancement moves, each costing ``enhancement_cost`` ($) plus
    ``enhancement_area_cost`` ($ per m2 of the unit's installed area).
    ``enhancement_pressure_drops`` (exchanger id to the stream its insert
    is on and the pressure drop, kPa, it adds) and
    ``pressure_drop_allowances`` (stream to the extra pressure drop, kPa, it
    can take) keep every design's inserts within each stream's allowance.
    ``repiping_cost`` ($ per stream) adds to a new match, relocation or
    new end heater or cooler the cost of each stream whose piping it
    changes. ``heat_pump_cycle`` enables heat pump moves (None: none): an
    evaporator at the end of a hot stream and a condenser at the end of a
    cold one, rated by the HPR service's cycle models, with work at
    ``COSTING_HPR_ELE_PRICE``. "carnot" rates them with
    ``HPR_ETA_II_CARNOT``, pricing the evaporator and condenser like any new
    exchanger (``new_unit_cost`` and their area) and the compressor and its
    package at ``heat_pump_cost`` ($ per kW of condenser capacity).
    "vapour_compression" rates them with a single stage on the best of
    ``HPR_REFRIGERANTS`` within the HPR cycle limits, and prices the whole
    machine with the HPR service's installed capital.

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
    approach: float | None = DEFAULT_FEASIBILITY_APPROACH
    new_unit_cost: float = 0.0
    area_addition_cost: float = 0.0
    split_cost: float = 0.0
    relocation_cost: float | None = None
    relocation_candidates: int = 8
    new_utility_units: bool = False
    strategy: str = "structure"
    ceiling_candidates: int = 12
    ceiling_relocations: int = 400
    enhancement_factors: Mapping[str, float] | None = None
    enhancement_cost: float = 0.0
    enhancement_area_cost: float = 0.0
    enhancement_pressure_drops: Mapping[str, tuple[str, float]] | None = None
    pressure_drop_allowances: Mapping[str, float] | None = None
    repiping_cost: float = 0.0
    heat_pump_cycle: str | None = None
    heat_pump_cost: float = 0.0


@dataclass(frozen=True)
class _Move:
    kind: str
    exchanger_id: str
    match: NewMatch | None = None
    splits: tuple[str, ...] = ()
    utility_id: str | None = None
    repiped: tuple[str, ...] = ()
    heat_pump: HeatPumpCandidate | None = None


@dataclass(frozen=True)
class _State:
    network: HeatExchangerNetwork
    enlarged: frozenset[str]
    moves: tuple[_Move, ...]
    enhanced: frozenset[str] = frozenset()


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
    be None or finite and non-negative, and every exchanger id and stream in
    ``enhancement_factors``, ``enhancement_pressure_drops`` and
    ``pressure_drop_allowances`` must be in the existing network.
    """

    if settings.strategy not in STRATEGIES:
        raise ValueError(
            f"strategy must be one of {STRATEGIES}; got {settings.strategy!r}."
        )
    if settings.heat_pump_cycle not in HEAT_PUMP_CYCLES:
        raise ValueError(
            f"heat_pump_cycle must be one of {HEAT_PUMP_CYCLES}; got "
            f"{settings.heat_pump_cycle!r}."
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
    streams = {
        str(name)
        for e in _structural_exchangers(existing)
        for name in (e.source_stream, e.sink_stream)
    }
    drops = settings.enhancement_pressure_drops or {}
    for name, given, known, what in (
        ("enhancement_factors", settings.enhancement_factors or {}, ids, "exchangers"),
        ("enhancement_pressure_drops", drops, ids, "exchangers"),
        (
            "enhancement_pressure_drops",
            {stream for stream, _drop in drops.values()},
            streams,
            "streams",
        ),
        (
            "pressure_drop_allowances",
            settings.pressure_drop_allowances or {},
            streams,
            "streams",
        ),
    ):
        unknown = sorted(set(given).difference(known))
        if unknown:
            raise ValueError(
                f"{name} names {what} not in the existing network: "
                + ", ".join(unknown)
                + "."
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
    # A unit given area earlier is not at its cap, but an insert may still
    # replace part of that area, so enlarged units stay candidates.
    factors = settings.enhancement_factors or {}
    candidates = parent.binding + tuple(
        sorted(state.enlarged.difference(parent.binding))
    )
    for exchanger_id in () if structural else candidates:
        if (
            exchanger_id in factors
            and exchanger_id not in state.enhanced
            and _within_pressure_drop(
                state.enhanced | {exchanger_id}, settings, state.network
            )
        ):
            yield replace(
                state,
                enhanced=state.enhanced | {exchanger_id},
                moves=state.moves + (_Move("enhancement", exchanger_id),),
            )
    for network, move in _structural_steps(parent, settings, evaluator):
        yield replace(state, network=network, moves=state.moves + (move,))
        if not settings.new_utility_units or move.kind == "heat_pump":
            continue
        for utility in new_utility_candidates(network, **evaluator.utility_axes):
            utility_id = _new_utility_id(network, utility)
            yield replace(
                state,
                network=apply_new_utility(network, utility, utility_id),
                moves=state.moves
                + (
                    replace(
                        move,
                        utility_id=utility_id,
                        # Each stream is repiped (and charged) once.
                        repiped=tuple(dict.fromkeys(move.repiped + (utility.stream,))),
                    ),
                ),
            )


def _within_pressure_drop(
    enhanced, settings: RetrofitSettings, network: HeatExchangerNetwork
) -> bool:
    """Whether the inserts of ``enhanced`` fit every stream's pressure
    drop allowance (a stream without one takes any) in ``network``. An
    insert whose stream its exchanger no longer serves (a relocation took
    it away) never fits."""

    drops = settings.enhancement_pressure_drops or {}
    allowances = settings.pressure_drop_allowances or {}
    served = {
        e.exchanger_id: {str(e.source_stream), str(e.sink_stream)}
        for e in network.exchangers
    }
    added: dict[str, float] = {}
    for unit in enhanced:
        if unit in drops:
            stream, drop = drops[unit]
            if stream not in served.get(unit, ()):
                return False
            added[stream] = added.get(stream, 0.0) + float(drop)
    return all(
        total <= float(allowances[stream]) + 1e-9
        for stream, total in added.items()
        if stream in allowances
    )


def _structural_steps(parent: _Evaluated, settings: RetrofitSettings, evaluator):
    """The screened new matches, the screened heat pumps (with a
    ``heat_pump_cost``) and, with a ``relocation_cost``, the relocations
    from ``parent``, each as ``(network, move)``."""

    network = parent.state.network
    yield from _heat_pump_steps(parent, settings, evaluator)
    matches = screen_new_matches(
        parent.solved,
        evaluator.ends,
        approach=evaluator.screen_approach,
        limit=settings.new_match_candidates,
    )
    for match in matches:
        exchanger_id = _new_id(network, match)
        move = _Move(
            "new_match",
            exchanger_id,
            match,
            new_splits(network, match),
            repiped=(match.hot, match.cold),
        )
        yield apply_new_match(network, match, exchanger_id), move
    if settings.relocation_cost is None:
        return
    for unit, match in _relocations(
        parent,
        evaluator.ends,
        settings,
        evaluator.existing,
        approach=evaluator.screen_approach,
    ):
        remaining = network.model_copy(
            update={
                "exchangers": tuple(
                    e for e in network.exchangers if e.exchanger_id != unit
                )
            }
        )
        move = _Move(
            "relocation",
            unit,
            match,
            new_splits(remaining, match),
            repiped=_relocation_repiping(network, unit, match),
        )
        yield apply_relocation(network, unit, match), move


def _heat_pump_steps(parent: _Evaluated, settings: RetrofitSettings, evaluator):
    """The ``new_match_candidates`` heat pumps that could relieve the most
    utility in ``parent``, each not yet installed, as ``(network, move)``:
    the evaporator added at its hot stream's end, the condenser at its cold
    stream's."""

    network = parent.state.network
    installed = {str(e.source_stream) for e in network.exchangers}
    for candidate in screen_heat_pumps(
        parent.solved,
        [c for c in evaluator.heat_pumps if condenser_name(c) not in installed],
        limit=settings.new_match_candidates,
    ):
        evaporator = NewUtility(candidate.hot, evaporator_name(candidate), False)
        evaporator_id = _new_utility_id(network, evaporator)
        piped = apply_new_utility(network, evaporator, evaporator_id)
        condenser = NewUtility(candidate.cold, condenser_name(candidate), True)
        condenser_id = _new_utility_id(piped, condenser)
        yield (
            apply_new_utility(piped, condenser, condenser_id),
            _Move(
                "heat_pump",
                evaporator_id,
                utility_id=condenser_id,
                repiped=(candidate.hot, candidate.cold),
                heat_pump=candidate,
            ),
        )


def _heat_pump_links(state: _State, structure, ids) -> tuple:
    """Each heat pump of ``state`` as (evaporator, condenser, COP), the two
    by index among ``structure``'s utility exchangers (``ids`` names each
    slot)."""

    slot = {
        exchanger_id: index
        for (kind, index), exchanger_id in zip(structure.slots, ids)
        if kind == "utility"
    }
    return tuple(
        (slot[m.exchanger_id], slot[m.utility_id], m.heat_pump.cop)
        for m in state.moves
        if m.kind == "heat_pump"
    )


def _relocation_repiping(
    network: HeatExchangerNetwork, unit: str, match: NewMatch
) -> tuple[str, ...]:
    """The streams moving ``unit`` to ``match`` repipes: those it leaves
    and those it joins, less any it stays on in the same stage."""

    old = next(e for e in network.exchangers if e.exchanger_id == unit)
    before = (str(old.source_stream), str(old.sink_stream))
    after = (match.hot, match.cold)
    in_place = not match.new_stage and match.stage == old.stage
    kept = set(before).intersection(after) if in_place else set()
    return tuple(s for s in dict.fromkeys(before + after) if s not in kept)


def _relocations(
    parent: _Evaluated, ends, settings: RetrofitSettings, existing, *, approach
):
    """The relocations from ``parent``, most promising first.

    Each existing recovery unit not yet moved may go to any screened match
    that shares one of its streams (its own pair elsewhere is a repiping).
    A move scores the match's potential less the unit's present duty, the
    heat it gives up, best first. The stepwise search keeps the
    ``relocation_candidates`` best positive scores; the structure search
    keeps up to ``ceiling_relocations`` moves whatever their score, as the
    ceiling ranking judges a move by what its whole structure can reach and
    the score barely tracks that; the cap bounds the ranking on large
    networks. A unit that is the
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
    pool = screen_new_matches(solved, ends, approach=approach, limit=sys.maxsize)
    scored = []
    for match in pool:
        potential = match_potential(solved, ends, match, approach=approach)
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
        cap = max(0, int(settings.ceiling_relocations))
        return [(unit, match) for _score, unit, match in scored[:cap]]
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
        enhanced=frozenset(),
        moves=tuple(
            m for m in state.moves if m.kind not in {"area_addition", "enhancement"}
        ),
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
    return recovery, utilities, state.enlarged, state.enhanced


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
        self.screen_approach = (
            2.0 * float(config.thermal.dt_cont)
            if settings.approach is None
            else float(settings.approach)
        )
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
        # Heat pumps add their condensers and evaporators as utilities,
        # after the network's own (which new utility exchangers draw on).
        self.heat_pumps: list[HeatPumpCandidate] = []
        self.cycle_settings = _cycle_settings(config.hpr)
        if settings.heat_pump_cycle is not None:
            rating = (
                carnot_rating(float(config.hpr.eta_ii_carnot))
                if settings.heat_pump_cycle == "carnot"
                else vapour_compression_rating(
                    list(config.hpr.normalised_refrigerants), **self.cycle_settings
                )
            )
            # A pump whose condenser or evaporator would take a name the
            # problem already uses is left out, so that name keeps resolving
            # to the problem's own stream or utility.
            taken = [
                str(key)
                for axis in (
                    "hot_process_streams",
                    "cold_process_streams",
                    "hot_utilities",
                    "cold_utilities",
                )
                for key in axes[axis]
            ]
            self.heat_pumps = [
                candidate
                for candidate in heat_pump_candidates(
                    self.ends, approach=self.screen_approach, rating=rating
                )
                if not any(
                    key == name or key.endswith("." + name)
                    for name in (condenser_name(candidate), evaporator_name(candidate))
                    for key in taken
                )
            ]
            price_scale, _capital = annual_cost_scales(config.costing)
            self.arrays = with_heat_pump_utilities(
                self.arrays,
                self.heat_pumps,
                float(config.costing.hpr_ele_price) * price_scale,
            )
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
        factors = settings.enhancement_factors or {}
        enhanced = {
            kind: {
                index: float(factors[exchanger_id])
                for (slot_kind, index), exchanger_id in zip(structure.slots, ids)
                if slot_kind == kind and exchanger_id in state.enhanced
            }
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
                enhanced_recovery=enhanced["recovery"],
                enhanced_utilities=enhanced["utility"],
                heat_pumps=_heat_pump_links(state, structure, ids),
            ),
        )
        # A move can leave a structure the model cannot build (streams
        # with no utility exchanger whose loads cannot balance): drop it.
        # The existing network's own error is the user's to see.
        try:
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
        except ValueError:
            if not state.moves:
                raise
            return None
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
            reported = getattr(model, "rating_allowances", None) or {}
            allowances = {
                exchanger_id: reported.get(slot, 1.0)
                for slot, exchanger_id in zip(structure.slots, ids)
            }
            if verify_solution(
                structure, model, self._widened(solved, state.enhanced, allowances)
            ):
                return None
            results = dict(zip(ids, structure.results(model), strict=True))
            widen = 1.0 + float(settings.area_tolerance)
            values = getattr(model, "area_values", None)
            worth = (
                None
                if values is None
                else {
                    exchanger_id: values.get(slot, 0.0)
                    for slot, exchanger_id in zip(structure.slots, ids)
                }
            )
            # Units at their cap, most valuable area first; with duals, only
            # those where more area is worth something.
            binding = tuple(
                sorted(
                    (
                        exchanger_id
                        for exchanger_id in ids
                        if exchanger_id in self.existing
                        and exchanger_id not in state.enlarged
                        and results[exchanger_id].active
                        and max(p.required_area for p in results[exchanger_id].periods)
                        >= _BINDING_SHARE * widen * results[exchanger_id].area
                        and (worth is None or worth[exchanger_id] > _MIN_AREA_VALUE)
                    ),
                    key=lambda exchanger_id: (
                        0.0 if worth is None else -worth[exchanger_id]
                    ),
                )
            )
            metrics = solved.summary_metrics
            # Heat pump condensers and evaporators are not utility use.
            pumped = [
                (
                    self._weighted(results[m.utility_id]),
                    self._weighted(results[m.exchanger_id]),
                )
                for m in state.moves
                if m.kind == "heat_pump"
            ]
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
                hot_utility=float(metrics["hot_utility_load"])
                - sum(condenser for condenser, _evaporator in pumped),
                cold_utility=float(metrics["cold_utility_load"])
                - sum(evaporator for _condenser, evaporator in pumped),
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

    def _weighted(self, result) -> float:
        """``result``'s duty, period-weighted (kW)."""

        weights = [float(w) for w in self.arrays.arrays["period_weights"]]
        total = sum(weights) or 1.0
        return (
            sum(w * p.duty for w, p in zip(weights, result.periods, strict=True))
            / total
        )

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
        exchangers = _structural_exchangers(state.network)
        ids = [exchangers[p].exchanger_id for p in structure.positions]
        try:
            model = FixedStructureModel(
                name="retrofit-ceiling",
                solver=self.solver,
                solver_arrays=self.arrays,
                spec=replace(
                    structure.spec,
                    recovery_areas={},
                    utility_areas={},
                    heat_pumps=_heat_pump_links(state, structure, ids),
                ),
                minimisation_goal="utility cost",
                default_approach=self.settings.approach,
                tol=self.tol,
                solver_options=dict(self.solver_options),
            )
        except ValueError:
            # A structure the model cannot build has no ceiling (see evaluate).
            return None
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
        added area is worth keeping, each an area addition. With
        ``enhancement_factors``, the kept units that have one are also tried
        enhanced, their area sized again the same way; the better of the two
        by net annual benefit is kept."""

        sized = self._sized_with(state, frozenset())
        factors = self.settings.enhancement_factors or {}
        if sized is None or not factors:
            return sized
        # Kept units with a factor, best return per kPa first, while their
        # inserts fit each stream's pressure drop allowance.
        enhance: frozenset[str] = frozenset()
        for unit in sorted(
            (
                m.exchanger_id
                for m in sized.state.moves
                if m.kind == "area_addition" and m.exchanger_id in factors
            ),
            key=lambda unit: -self._insert_return(unit, sized.added.get(unit, 0.0)),
        ):
            if _within_pressure_drop(enhance | {unit}, self.settings, state.network):
                enhance = enhance | {unit}
        if not enhance:
            return sized
        enhanced = self._sized_with(state, enhance)
        if (
            enhanced is not None
            and enhanced.design.net_annual_benefit > sized.design.net_annual_benefit
        ):
            return enhanced
        return sized

    def _insert_return(self, unit: str, added: float) -> float:
        """What an insert on ``unit`` may save per kPa of pressure drop it
        adds: the area addition it can stand in for (fixed cost and the
        ``added`` area's cost) less its own cost. Infinite (first) for an
        insert with no pressure drop; an unprofitable one ranks below every
        profitable one."""

        settings = self.settings
        saving = (
            settings.area_addition_cost
            + self._area_cost(added)
            - settings.enhancement_cost
            - settings.enhancement_area_cost * self.installed[unit]
        )
        drops = settings.enhancement_pressure_drops or {}
        drop = float(drops[unit][1]) if unit in drops else 0.0
        if drop <= 0.0:
            return math.inf if saving > 0.0 else saving
        return saving / drop

    def _sized_with(self, state: _State, enhance: frozenset[str]) -> _Evaluated | None:
        """``state`` with the ``enhance`` units enhanced and its added area
        sized (see ``sized``)."""

        state = replace(
            state,
            enhanced=enhance,
            moves=state.moves
            + tuple(_Move("enhancement", unit) for unit in sorted(enhance)),
        )
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

    def _widened(
        self,
        network: HeatExchangerNetwork,
        enhanced: frozenset[str] = frozenset(),
        allowances: Mapping[str, float] | None = None,
    ) -> HeatExchangerNetwork:
        """``network`` with each existing unit's installed area widened by
        ``area_tolerance`` and by the model's rating allowance for it
        (``allowances``: how much more area exact or slice sizing needs than
        the model's calibrated Chen cap), and an ``enhanced`` unit's area
        times its enhancement factor (verification sizes at the base
        coefficient), as the model rates them."""

        tolerance = float(self.settings.area_tolerance)
        factors = self.settings.enhancement_factors or {}

        def rated(e):
            area = (e.area + tolerance * self.installed[e.exchanger_id]) * (
                (allowances or {}).get(e.exchanger_id, 1.0)
            )
            if e.exchanger_id in enhanced:
                area *= float(factors[e.exchanger_id])
            return e.model_copy(update={"area": area})

        return network.model_copy(
            update={
                "exchangers": tuple(
                    rated(e)
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
        nothing (no added area, or a unit with no duty) or, for a real-fluid
        heat pump, when its machine's cycle cannot be solved at its duty. A
        relocation adds no area. A new or moved match reports ``stage``, its
        stage in the design's network. ``utility`` is the result of the new utility
        exchanger paired with the move, priced as one more new unit; it too
        must carry duty."""

        if not result.active or (utility is not None and not utility.active):
            return None
        settings = self.settings
        splits = settings.split_cost * len(move.splits)
        if move.kind == "relocation":
            added, fixed = 0.0, settings.relocation_cost + splits
        elif move.kind == "heat_pump" and move.heat_pump.refrigerant is not None:
            # The HPR machine capital covers the exchangers too.
            added = 0.0
            try:
                fixed = vapour_compression_capital(
                    move.heat_pump,
                    utility.max_duty,
                    (float(result.added_area), float(utility.added_area)),
                    **self.cycle_settings,
                )
            except (ValueError, RuntimeError) as exc:
                # The cycle can fail at the solved capacity: drop this one.
                logger.debug("Heat pump %s not priced: %s", move.heat_pump, exc)
                return None
        elif move.kind == "heat_pump":
            # The compressor at its capacity; the evaporator and condenser
            # are new exchangers, as any new match is.
            added = float(result.added_area)
            fixed = (
                settings.heat_pump_cost * utility.max_duty + 2 * settings.new_unit_cost
            )
        elif move.kind == "enhancement":
            added = 0.0
            fixed = (
                settings.enhancement_cost
                + settings.enhancement_area_cost * self.installed[move.exchanger_id]
            )
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
        machine = move.heat_pump is not None and move.heat_pump.refrigerant is not None
        if utility is not None and move.kind != "heat_pump":
            fixed += settings.new_unit_cost
        fixed += settings.repiping_cost * len(move.repiped)
        match = move.match
        pump = move.heat_pump
        if pump is not None:
            return HeatExchangerNetworkRetrofitModification(
                kind=move.kind,
                exchanger_id=move.exchanger_id,
                added_area=float(result.added_area),
                fixed_cost=float(fixed),
                area_cost=0.0
                if machine
                else self._area_cost(added) + self._area_cost(utility_area),
                hot_stream=pump.hot,
                cold_stream=pump.cold,
                utility_exchanger_id=move.utility_id,
                utility_area=utility_area,
                repiped_streams=move.repiped,
                heat_pump_cop=pump.cop,
                heat_pump_capacity=utility.max_duty,
                heat_pump_refrigerant=pump.refrigerant,
            )
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
            repiped_streams=move.repiped,
        )

    def _area_cost(self, area: float) -> float:
        """Installed cost ($) of ``area`` m2 of new area."""

        return self.area_coeff * area**self.area_exp if area > 0.0 else 0.0


def _cycle_settings(hpr) -> dict:
    """The HPR service's vapour compression settings: the approach,
    compressor and motor efficiencies and cycle limits (pressures in Pa)."""

    return {
        "dtcont": float(hpr.dt_cont),
        "eta_comp": compressor_efficiency_model(hpr.compressor_model, hpr.eta_comp),
        "eta_motor": float(hpr.eta_motor),
        "limits": CycleLimits(
            max_discharge_superheat=hpr.max_discharge_superheat,
            max_pressure_ratio=hpr.max_pressure_ratio,
            max_discharge_pressure=hpr.max_discharge_pressure * 1e5,
            max_discharge_pressure_co2=hpr.max_discharge_pressure_co2 * 1e5,
            min_suction_pressure=hpr.screen_min_suction_pressure * 1e5,
            min_lift=float(hpr.min_stage_lift),
        ),
    }


def _existing_names(axis_maps, existing: HeatExchangerNetwork) -> dict[str, str]:
    """Each solver stream name mapped to the name ``existing`` uses for it.

    The existing network may name a stream without its zone prefix
    (``H1`` for ``Process A.H1``) unless a stream has that name exactly;
    streams it does not name keep theirs. A bare name that fits streams of
    several zones, and no stream exactly, raises ``ValueError``.
    """

    used = {
        name for e in existing.exchangers for name in (e.source_stream, e.sink_stream)
    }
    keys = [str(key) for axis in axis_maps.values() for key in axis]
    for name in used:
        tails = [key for key in keys if key.endswith("." + name)]
        if name not in keys and len(tails) > 1:
            raise ValueError(
                f"the existing network names stream {name!r}, which could be "
                + " or ".join(sorted(tails))
                + "; name it with its zone."
            )
    names: dict[str, str] = {}
    for key in keys:
        matches = [
            n for n in used if n == key or (key.endswith("." + n) and n not in keys)
        ]
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

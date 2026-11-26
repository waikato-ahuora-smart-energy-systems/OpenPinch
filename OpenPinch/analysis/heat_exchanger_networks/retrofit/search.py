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
import warnings
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

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
from ..heat_pump_cost import network_with_heat_pumps
from .frontier import mark_dominated
from .heat_pump_streams import with_heat_pump_matches
from .loops import (
    HeatRecoveryLoop,
    heat_recovery_loop_candidates,
    loop_cooler_name,
    loop_heater_name,
    screen_heat_recovery_loops,
    with_loop_utilities,
)
from .moves import (
    NewMatch,
    NewUtility,
    apply_new_match,
    apply_new_utility,
    apply_relocation,
    apply_utility_switch,
    match_potential,
    new_splits,
    new_utility_candidates,
    period_mean,
    screen_new_matches,
    utility_switch_candidates,
)
from .processes import stream_process_zones

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
# The search warns when more than this share of its solves failed.
_FAILED_SOLVE_WARNING_SHARE = 0.25
STRATEGIES = ("stepwise", "structure")
# The solver axes whose names a loop utility must not take.
_NAMED_AXES = (
    "hot_process_streams",
    "cold_process_streams",
    "hot_utilities",
    "cold_utilities",
)
# Moves that already pair two utility exchangers take no new end utility.
_UTILITY_PAIRS = ("heat_recovery_loop",)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrofitSettings:
    """Search limits and retrofit fixed costs.

    ``max_modifications`` is the search depth, ``beam_width`` the designs kept
    per depth and ``new_match_candidates`` the screened new matches tried from
    each. ``area_tolerance`` widens every installed area (a fraction) and
    ``approach`` is the minimum approach (K) on every exchanger; ``None``
    gives each exchanger the sum of its two sides' temperature contributions
    and screens new matches and heat pumps at twice ``THERMAL_DT_CONT``,
    both scaled by ``dt_cont_multiplier`` (1 by default; the multiplier, and
    for an evolved network the multiplier times the approach share, that a
    designed existing network records; left at 1 with an ``approach``).
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
    changes. ``heat_pump_cycle`` and ``heat_pump_cost`` are the heat pump
    stream step's (``retrofit_with_heat_pump``): the search itself refuses a
    cycle, its linked heat pump move being retired; it starts from a heat
    pump only as streams (``heat_pump=``).
    ``heat_recovery_loops`` (default on) lets a retrofit whose existing
    network spans several process zones link two of them through a new heat
    recovery loop, the only link it proposes between processes: a loop
    cooler at the end of a hot stream of one process and a loop heater at
    the end of a cold stream of another, an intermediate fluid at one
    temperature whose duties balance in every period. Its two exchangers
    are priced like new exchangers, plus ``heat_recovery_loop_cost`` ($ per
    loop, for its pump and piping run). ``utility_switch_cost`` ($ per
    switch) enables utility switches (None: none): each existing heater or
    cooler, once, served by another utility of its kind, keeping its id,
    stream and installed area; any area it then needs is an area addition
    of its own.

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
    heat_recovery_loops: bool = True
    heat_recovery_loop_cost: float = 0.0
    utility_switch_cost: float | None = None
    dt_cont_multiplier: float = 1.0


@dataclass(frozen=True)
class _Move:
    kind: str
    exchanger_id: str
    match: NewMatch | None = None
    splits: tuple[str, ...] = ()
    utility_id: str | None = None
    repiped: tuple[str, ...] = ()
    loop: HeatRecoveryLoop | None = None
    # The heat pump whose stream this new match serves, when the search
    # started from a heat pump step.
    machine: str | None = None
    # A switched heater's or cooler's hot and cold sides after the switch.
    switch: tuple[str, str] | None = None


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
    problem,
    existing: HeatExchangerNetwork,
    settings: RetrofitSettings,
    *,
    heat_pump=None,
) -> HeatExchangerNetworkRetrofitResult:
    """Search modifications of ``existing``, one at a time.

    Every exchanger of ``existing`` needs its installed ``area``; a
    ``ValueError`` names those without one. ``max_added_area_fraction`` must
    be None or finite and non-negative, a ``heat_pump_cycle`` is refused (the
    linked heat pump move is retired), and every exchanger id and stream in
    ``enhancement_factors``, ``enhancement_pressure_drops`` and
    ``pressure_drop_allowances`` must be in the existing network.
    """

    if settings.strategy not in STRATEGIES:
        raise ValueError(
            f"strategy must be one of {STRATEGIES}; got {settings.strategy!r}."
        )
    if settings.heat_pump_cycle is not None:
        raise ValueError(
            "the retrofit search's linked heat pump move is retired; a heat "
            "pump enters a retrofit as streams (design.retrofit with "
            f"heat_pump_cycle={settings.heat_pump_cycle!r}, or "
            "retrofit_with_heat_pump)."
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
    evaluator = _Evaluator(problem, existing, settings, heat_pump)
    if evaluator.processes is not None:
        warnings.warn(
            "the existing network spans the process zones "
            + ", ".join(sorted(set(evaluator.processes.values())))
            + ": retrofit proposes new matches only within one process"
            + (
                ", linking processes only through a new heat recovery loop,"
                if evaluator.loops_on
                else ""
            )
            + " and does not model the site utility balance, so it prices "
            "utilities unit by unit.",
            UserWarning,
            stacklevel=3,
        )
    base = evaluator.evaluate(
        _State(existing, frozenset(), ())
        if heat_pump is None
        else _heat_pump_root(existing, heat_pump, evaluator)
    )
    if base is None and heat_pump is not None:
        raise ValueError(
            f"the existing network with the heat pump {heat_pump.record.machine!r} "
            "added as streams does not solve: its coolers and heaters cannot "
            "serve the machine's duties at their approaches."
        )
    if base is None:
        raise ValueError(
            "the existing network does not meet its targets at its installed "
            "areas; check the areas or raise the area tolerance."
        )
    frontier = [base.design]
    beam = [base]
    seen = {_signature(base.state, evaluator.existing)}
    structural = settings.strategy == "structure"
    failed: list[int] = []
    # A heat pump step is the first of the modifications allowed.
    taken = 0 if heat_pump is None else 1
    for _depth in range(max(0, int(settings.max_modifications) - taken)):
        children = []
        failed_before = evaluator.failures
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
        if children:
            children.sort(key=lambda child: -child.design.net_annual_benefit)
            beam = children[: max(1, int(settings.beam_width))]
            if structural:
                # Only the depth's best design: each drop costs a solve, and
                # the next depth sizes every kept design's area afresh.
                beam[0] = evaluator.paying(beam[0])
            frontier.append(beam[0].design)
        failed.append(evaluator.failures - failed_before)
        if not children:
            break
    if sum(failed) > _FAILED_SOLVE_WARNING_SHARE * evaluator.evaluations:
        warnings.warn(
            f"{sum(failed)} of {evaluator.evaluations} retrofit candidate solves "
            "failed (did not converge, or the solved network failed its "
            "checks) and were dropped, so the frontier may miss better "
            "designs; result.failed_solves counts them by step. A wider "
            "area_tolerance or other solver options may recover them.",
            UserWarning,
            stacklevel=3,
        )
    return HeatExchangerNetworkRetrofitResult(
        frontier=mark_dominated(frontier),
        evaluated_designs=evaluator.evaluations,
        failed_solves=tuple(failed),
    )


def _heat_pump_root(existing: HeatExchangerNetwork, step, evaluator) -> _State:
    """The state a search with a heat pump step starts from: ``existing``
    with the matches that serve the machine's streams, each a new match.

    The matches are placed from the base design's solved coolers and
    heaters. A match stands off by the search's approach, or with none by
    its process stream's own contribution and the heat pump's.
    """

    config = evaluator.config
    approach = (
        float(config.thermal.dt_cont) + float(config.hpr.dt_cont)
        if evaluator.settings.approach is None
        else float(evaluator.settings.approach)
    )
    # Each process stream's own contribution and the heat pump's, each side
    # apart: a zone may name a hot and a cold stream alike.
    own = evaluator.settings.approach is None
    hot_approaches, cold_approaches = (
        (
            {
                name: float(contribution) + float(config.hpr.dt_cont)
                for name, contribution in getattr(evaluator, side).items()
            }
            if own
            else None
        )
        for side in ("hot_contributions", "cold_contributions")
    )
    # Air carries the ambient contribution, not a process stream's.
    ambient_approach = (
        float(config.hpr.dt_env_cont) + float(config.hpr.dt_cont)
        if evaluator.settings.approach is None
        else float(evaluator.settings.approach)
    )
    network, added = with_heat_pump_matches(
        existing,
        condensers=step.condensers,
        evaporators=step.evaporators,
        approach=approach,
        machine=step.record.machine,
        # The base design's network names streams as the solver does; the
        # matches name them as the existing network does.
        solved=_with_names(step.base.network, evaluator.names),
        ambient_sources=step.ambient_sources,
        ambient_approach=ambient_approach,
        hot_approaches=hot_approaches,
        cold_approaches=cold_approaches,
        # A shared design: each level against each period's network.
        in_periods=step.in_periods,
    )
    matched = {
        name for placed in added for name in (placed.match.hot, placed.match.cold)
    }
    held = (
        f"each stream's contribution plus {float(config.hpr.dt_cont):g} K"
        if own
        else f"{approach:g} K"
    )
    for streams, what, gap in (
        (step.condensers, "no heater starts cold enough for the condenser", held),
        (step.evaporators, "no cooler starts hot enough for the evaporator", held),
        (
            step.ambient_sources,
            "no evaporator is colder than the ambient source",
            f"{ambient_approach:g} K",
        ),
    ):
        for stream in streams:
            if stream.name not in matched:
                raise ValueError(
                    f"{what} {stream.name!r} of heat pump "
                    f"{step.record.machine!r} at an approach of {gap}, "
                    "so nothing in the existing network can serve it."
                )
    # The heat pump's streams are new: only the process stream of each
    # match has piping to change.
    pumped = {
        stream.name
        for stream in step.condensers + step.evaporators + step.ambient_sources
    }
    moves = tuple(
        _Move(
            "new_match",
            placed.exchanger_id,
            match=placed.match,
            splits=placed.splits,
            repiped=tuple(
                stream
                for stream in (placed.match.hot, placed.match.cold)
                if stream not in pumped
            ),
            machine=step.record.machine,
        )
        for placed in added
    )
    return _State(network, frozenset(), moves)


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
        # A switch is one change of its own, as a utility pair is.
        if not settings.new_utility_units or move.kind in (
            *_UTILITY_PAIRS,
            "utility_switch",
        ):
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
    """The screened heat recovery loops, the screened new matches, the
    utility switches (with a ``utility_switch_cost``) and, with a
    ``relocation_cost``, the relocations from ``parent``, each as
    ``(network, move)``."""

    network = parent.state.network
    yield from _loop_steps(parent, settings, evaluator)
    matches = screen_new_matches(
        parent.solved,
        evaluator.ends,
        approach=evaluator.screen_approach,
        limit=settings.new_match_candidates,
        processes=evaluator.processes,
        weights=evaluator.weights,
        period_ends=evaluator.period_ends,
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
    yield from _switch_steps(parent, settings, evaluator)
    if settings.relocation_cost is None:
        return
    for unit, match in _relocations(
        parent,
        evaluator.ends,
        settings,
        evaluator.existing,
        approach=evaluator.screen_approach,
        processes=evaluator.processes,
        weights=evaluator.weights,
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


def _switch_steps(parent: _Evaluated, settings: RetrofitSettings, evaluator):
    """With a ``utility_switch_cost``, each existing heater or cooler not yet
    switched, served by each other utility of its kind, as
    ``(network, move)``."""

    if settings.utility_switch_cost is None:
        return
    network = parent.state.network
    switched = {
        m.exchanger_id for m in parent.state.moves if m.kind == "utility_switch"
    }
    for unit, utility in utility_switch_candidates(
        network,
        units=evaluator.existing - switched,
        hot_utilities=evaluator.utility_axes["hot_utilities"],
        cold_utilities=evaluator.utility_axes["cold_utilities"],
    ):
        moved = apply_utility_switch(network, unit, utility)
        (after,) = (e for e in moved.exchangers if e.exchanger_id == unit)
        yield (
            moved,
            _Move(
                "utility_switch",
                unit,
                switch=(str(after.source_stream), str(after.sink_stream)),
            ),
        )


def _loop_steps(parent: _Evaluated, settings: RetrofitSettings, evaluator):
    """The ``new_match_candidates`` heat recovery loops that could relieve
    the most utility in ``parent``, each not yet installed, as
    ``(network, move)``: the loop cooler added at its hot stream's end, the
    loop heater at its cold stream's."""

    network = parent.state.network
    installed = {str(e.source_stream) for e in network.exchangers}
    for loop in screen_heat_recovery_loops(
        parent.solved,
        [c for c in evaluator.loops if loop_heater_name(c) not in installed],
        limit=settings.new_match_candidates,
        weights=evaluator.weights,
    ):
        cooler = NewUtility(loop.hot, loop_cooler_name(loop), False)
        cooler_id = _new_utility_id(network, cooler)
        piped = apply_new_utility(network, cooler, cooler_id)
        heater = NewUtility(loop.cold, loop_heater_name(loop), True)
        heater_id = _new_utility_id(piped, heater)
        yield (
            apply_new_utility(piped, heater, heater_id),
            _Move(
                "heat_recovery_loop",
                cooler_id,
                utility_id=heater_id,
                repiped=(loop.hot, loop.cold),
                loop=loop,
            ),
        )


def _loop_links(state: _State, structure, ids) -> tuple:
    """Each heat recovery loop of ``state`` as (cooler, heater), the two by
    index among ``structure``'s utility exchangers (``ids`` names each
    slot)."""

    slot = {
        exchanger_id: index
        for (kind, index), exchanger_id in zip(structure.slots, ids)
        if kind == "utility"
    }
    return tuple(
        (slot[m.exchanger_id], slot[m.utility_id])
        for m in state.moves
        if m.kind == "heat_recovery_loop"
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
    parent: _Evaluated,
    ends,
    settings: RetrofitSettings,
    existing,
    *,
    approach,
    processes=None,
    weights=None,
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
    only exchanger on one of its streams stays on that stream. With
    ``processes`` (stream to process zone) only matches within one process
    are screened.
    """

    solved = parent.solved
    moved = {m.exchanger_id for m in parent.state.moves if m.kind == "relocation"}
    duties = {
        e.exchanger_id: (
            e.source_stream,
            e.sink_stream,
            period_mean([state.duty for state in e.period_states], weights),
        )
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
        solved,
        ends,
        approach=approach,
        limit=sys.maxsize,
        processes=processes,
        weights=weights,
    )
    scored = []
    for match in pool:
        potential = match_potential(
            solved, ends, match, approach=approach, weights=weights
        )
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

    def __init__(
        self, problem, existing: HeatExchangerNetwork, settings, heat_pump=None
    ) -> None:
        from ..solver.arrays import (
            nominal_approach_temperature,
            problem_to_solver_arrays,
        )

        config = problem.master_zone.config
        self.config = config
        self.settings = settings
        # The streams' contributions at the multiplier the existing network
        # was designed at.
        nominal = nominal_approach_temperature(config, settings.dt_cont_multiplier)
        # The heat pump step the search starts from, if any.
        self.heat_pump = heat_pump
        self.existing = frozenset(
            e.exchanger_id for e in _structural_exchangers(existing)
        )
        self.installed = {
            e.exchanger_id: float(e.area) for e in _structural_exchangers(existing)
        }
        self.arrays = problem_to_solver_arrays(problem, nominal)
        self.screen_approach = (
            nominal if settings.approach is None else float(settings.approach)
        )
        hens = config.hens
        self.solver = configured_stage_solver(hens, "HENS_SOLVER_EVM")
        self.solver_options = stage_solver_options(hens, "evm", self.solver)
        self.tol = float(hens.solve_tolerance)
        self.max_parallel = supported_workers(max(1, int(hens.max_parallel)))
        self.names = _existing_names(self.arrays.axis_maps, existing)
        # Each process stream's contribution, by its existing name, each
        # side apart: a zone may name a hot and a cold stream alike.
        self.hot_contributions = _side_contributions(
            self.arrays, self.names, "hot_process_streams", "T_h_cont_period"
        )
        self.cold_contributions = _side_contributions(
            self.arrays, self.names, "cold_process_streams", "T_c_cont_period"
        )
        # The streams the screens offer new matches and utility units on. A
        # heat pump step's own streams are left out: its step gave them
        # their matches, and a condenser must not heat its evaporator.
        pumped = (
            ()
            if heat_pump is None
            else [
                s.name
                for s in heat_pump.condensers
                + heat_pump.evaporators
                + heat_pump.ambient_sources
            ]
        )

        def offered(name: str) -> bool:
            return not any(name == p or name.endswith("." + p) for p in pumped)

        self.ends = {
            self.names[name]: ends
            for name, ends in _stream_ends(self.arrays).items()
            if offered(name)
        }
        # Each period's own ends, so the match screens read potential period
        # by period.
        self.period_ends = tuple(
            {self.names[name]: ends for name, ends in period.items() if offered(name)}
            for period in _period_stream_ends(self.arrays)
        )
        # The period weights the screens average a solved network with.
        period_weights = np.asarray(self.arrays.arrays["period_weights"], dtype=float)
        self.weights = tuple(float(w) for w in period_weights / period_weights.sum())
        self.processes = _spanned_processes(problem, existing, self.names)
        axes = self.arrays.axis_maps
        self.utility_axes = {
            "hot_streams": tuple(n for n, e in self.ends.items() if e[0] > e[1]),
            "cold_streams": tuple(n for n, e in self.ends.items() if e[0] <= e[1]),
            "hot_utilities": tuple(self.names[n] for n in axes["hot_utilities"]),
            "cold_utilities": tuple(self.names[n] for n in axes["cold_utilities"]),
        }
        # A site retrofit may link two processes through a heat recovery
        # loop: its cooler and heater join the utilities, free.
        self.loops: list[HeatRecoveryLoop] = []
        self.loops_on = self.processes is not None and settings.heat_recovery_loops
        if self.loops_on:
            taken = [str(key) for axis in _NAMED_AXES for key in axes[axis]]
            self.loops = [
                loop
                for loop in heat_recovery_loop_candidates(
                    self.ends, self.processes, approach=self.screen_approach
                )
                if not any(
                    key == name or key.endswith("." + name)
                    for name in (loop_cooler_name(loop), loop_heater_name(loop))
                    for key in taken
                )
            ]
            self.arrays = with_loop_utilities(self.arrays, self.loops)
        # Savings are measured against the existing network: the first
        # design solved, or with a heat pump step the base search's.
        self.base_utility_cost: float | None = (
            None if heat_pump is None else float(heat_pump.base.utility_cost)
        )
        self.evaluations = 0
        # Solves that did not converge, or whose network failed its checks.
        self.failures = 0
        self._count_lock = threading.Lock()
        costing = config.costing
        self.area_coeff = float(costing.hx_area_coeff)
        self.area_exp = float(costing.hx_area_exp)
        self.recovery_factor = float(
            compute_capital_recovery_factor(costing.discount_rate, costing.service_life)
        )

    def _failed(self) -> None:
        """Count a solve that did not converge or failed its checks."""

        with self._count_lock:
            self.failures += 1

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
                loops=_loop_links(state, structure, ids),
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
            try:
                model.optimise(print_output=False)
            except ValueError:
                # A solved duty the model cannot place on a segmented
                # stream's profile (raised while it reads the solution):
                # the candidate is dropped. For the existing network the
                # error stands.
                if not state.moves:
                    raise
                self._failed()
                return None
            if getattr(model, "mSuccess", 0) != 1:
                self._failed()
                return None
            results = dict(zip(ids, structure.results(model), strict=True))
            solved = _with_owned_areas(
                build_solution_network(request, structure, model, self.arrays),
                results,
                self.installed,
            )
            # The solver accepts loosely converged points: drop a design whose
            # approaches, heat balances or areas fail the duty optimiser's
            # checks, judging installed areas as the model did (widened).
            reported = getattr(model, "rating_allowances", None) or {}
            allowances = {
                exchanger_id: reported.get(slot, 1.0)
                for slot, exchanger_id in zip(structure.slots, ids)
            }
            widened = self._widened(solved, state.enhanced, allowances)
            if verify_solution(structure, model, widened):
                self._failed()
                return None
            solved = _with_rating_allowances(solved, widened)
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
            # Heat pump condensers and evaporators, and heat recovery loops,
            # are not utility use.
            pumped = [
                (
                    self._weighted(results[m.utility_id]),
                    self._weighted(results[m.exchanger_id]),
                )
                for m in state.moves
                if m.kind in _UTILITY_PAIRS
            ]
            utility_cost = float(model.utility_cost_value)
            if self.heat_pump is not None:
                # The machine's electricity, a constant of the step.
                utility_cost += float(self.heat_pump.record.electricity_cost)
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
            if self.heat_pump is not None:
                # The machine itself, ahead of the matches that serve it:
                # its compressor is a constant of the step.
                record = self.heat_pump.record
                solved = network_with_heat_pumps(solved, (record,))
                modifications = (
                    HeatExchangerNetworkRetrofitModification(
                        kind="heat_pump",
                        exchanger_id=record.machine,
                        fixed_cost=float(record.capital_cost),
                        heat_pump_cop=float(record.cop),
                        heat_pump_capacity=float(record.condenser_duty),
                    ),
                    *modifications,
                )
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
                    loops=_loop_links(state, structure, ids),
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
                self._failed()
                return None
            return float(model.utility_cost_value)
        except ValueError:
            # A solved duty the model cannot place on a segmented stream's
            # profile (raised while it reads the solution) has no ceiling.
            self._failed()
            return None
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

    def paying(self, sized: _Evaluated) -> _Evaluated:
        """``sized`` without the area additions that do not pay.

        The sizing solve weighs an addition's area cost against the utility
        it saves but not its fixed cost (``area_addition_cost``), so it grows
        units that do not earn that cost. The design is solved again with
        each kept addition dropped in turn (on the search's threads, as its
        other solves are); the drop that raises net annual benefit most
        stands, and the rest are tried again from there until none does.
        """

        def net(evaluated: _Evaluated) -> float:
            return evaluated.design.net_annual_benefit

        best = sized
        while True:
            state = best.state
            trials = self.map(
                self.evaluate,
                [
                    replace(
                        state,
                        enlarged=state.enlarged - {move.exchanger_id},
                        moves=tuple(m for m in state.moves if m is not move),
                    )
                    for move in state.moves
                    if move.kind == "area_addition"
                ],
            )
            better = max((t for t in trials if t is not None), key=net, default=None)
            if better is None or net(better) <= net(best):
                return best
            best = better

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
        elif move.kind == "utility_switch":
            # Its added area, if any, is an area addition of its own.
            added, fixed = 0.0, settings.utility_switch_cost
        elif move.kind == "heat_recovery_loop":
            # Two new exchangers and the loop's pump and piping run.
            added = float(result.added_area)
            fixed = 2 * settings.new_unit_cost + settings.heat_recovery_loop_cost
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
        if utility is not None and move.kind not in _UTILITY_PAIRS:
            fixed += settings.new_unit_cost
        fixed += settings.repiping_cost * len(move.repiped)
        match = move.match
        loop = move.loop
        if loop is not None:
            return HeatExchangerNetworkRetrofitModification(
                kind=move.kind,
                exchanger_id=move.exchanger_id,
                added_area=added,
                fixed_cost=float(fixed),
                area_cost=self._area_cost(added) + self._area_cost(utility_area),
                hot_stream=loop.hot,
                cold_stream=loop.cold,
                utility_exchanger_id=move.utility_id,
                utility_area=utility_area,
                repiped_streams=move.repiped,
                loop_temperature=loop.temperature - 273.15,
                loop_duty=utility.max_duty,
            )
        hot, cold = (match.hot, match.cold) if match else move.switch or (None, None)
        return HeatExchangerNetworkRetrofitModification(
            kind=move.kind,
            exchanger_id=move.exchanger_id,
            added_area=added,
            fixed_cost=float(fixed),
            area_cost=self._area_cost(added) + self._area_cost(utility_area),
            hot_stream=hot,
            cold_stream=cold,
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


def _spanned_processes(
    problem, existing: HeatExchangerNetwork, names
) -> dict[str, str] | None:
    """Each stream (by its name in ``existing``) mapped to its process
    zone when ``existing``'s streams span several process zones, else
    ``None``."""

    processes = {
        names.get(key, key): process
        for key, process in stream_process_zones(problem.master_zone).items()
    }
    spanned = {
        processes[name]
        for e in _structural_exchangers(existing)
        for name in (e.source_stream, e.sink_stream)
        if name in processes
    }
    return processes if len(spanned) > 1 else None


def _side_contributions(arrays, names, axis: str, values: str) -> dict[str, float]:
    """Each process stream of one side's contribution (K, first period, with
    the zone's dt_cont multiplier), by the name ``names`` maps it to."""

    # Period-major: the first period's row.
    return {
        names.get(str(key), str(key)): float(arrays.arrays[values][0][index])
        for key, index in arrays.axis_maps[axis].items()
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


def _with_owned_areas(
    network: HeatExchangerNetwork, results: Mapping, installed: Mapping[str, float]
) -> HeatExchangerNetwork:
    """``network`` with each existing unit's area the one it has after the
    retrofit: installed plus any added, as the model holds it (``results``,
    by exchanger id). A solved network otherwise gives a unit the area its
    duty needs, which for a unit on a segmented stream is the sum of its
    slices and can be far below what is installed.

    A unit rated within the area tolerance may need a little more by its
    slices than it has. It then keeps the slices' area: an exchanger whose
    area is below its slices' is not a valid one, and could not be read
    back from the design's own dump.
    """

    def owned(exchanger):
        area = float(results[exchanger.exchanger_id].area)
        return max(area, exchanger.segment_design_area or 0.0)

    return network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": owned(e)})
                if e.exchanger_id in installed and e.exchanger_id in results
                else e
                for e in network.exchangers
            )
        }
    )


def _with_rating_allowances(
    network: HeatExchangerNetwork, widened: HeatExchangerNetwork
) -> HeatExchangerNetwork:
    """``network`` recording, per existing unit, how much its reported area
    was widened to rate it (``widened``'s area over its own: the area
    tolerance, the Chen cap's rating allowance and any enhancement), as
    ``rating_area_allowances``, so verification judges it as it was rated;
    the largest is ``summary_metrics["rating_area_allowance"]``."""

    rated = {e.exchanger_id: e.area for e in widened.exchangers}
    allowances = {
        e.exchanger_id: float(rated[e.exchanger_id]) / float(e.area)
        for e in network.exchangers
        if e.exchanger_id is not None
        and e.area
        and rated.get(e.exchanger_id) not in (None, e.area)
    }
    if not allowances:
        return network
    return network.model_copy(
        update={
            "source_metadata": dict(network.source_metadata)
            | {"rating_area_allowances": allowances},
            "summary_metrics": dict(network.summary_metrics)
            | {"rating_area_allowance": max(allowances.values())},
        }
    )


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
    flowrate, over every period.

    The flowrate is the period-weighted mean, so a stream that is off part
    of the time counts for the share it runs. The temperatures are the
    weighted mean over the periods the stream runs (every period for one
    that never does). The screens that read these therefore do not depend on
    which period is listed first. A stream with several segments has no
    single flowrate (a zero placeholder in the arrays): it gets its
    segments' duty over its temperature span.
    """

    weights = np.asarray(arrays.arrays["period_weights"], dtype=float)
    weights = weights / weights.sum()
    ends: dict[str, tuple[float, float, float]] = {}
    for name, supply, target, flows in _stream_rows(arrays):
        running = weights * (flows > 0.0)
        over = running / running.sum() if running.sum() > 0.0 else weights
        ends[name] = (
            float(over @ supply),
            float(over @ target),
            float(weights @ flows),
        )
    return ends


def _period_stream_ends(arrays) -> tuple[dict[str, tuple[float, float, float]], ...]:
    """Each period's own stream ends: every process stream's supply and
    target temperature and heat capacity flowrate in that period (0 for a
    stream that is off; a segmented stream's as in ``_stream_ends``)."""

    rows = list(_stream_rows(arrays))
    return tuple(
        {
            name: (float(supply[n]), float(target[n]), float(flows[n]))
            for name, supply, target, flows in rows
        }
        for n in range(len(arrays.arrays["period_weights"]))
    )


def _stream_rows(arrays):
    """Each process stream's name and, by period, its supply and target
    temperature and heat capacity flowrate (a segmented stream's being its
    segments' duty over its temperature span)."""

    data = arrays.arrays
    for side, axis in (("h", "hot_process_streams"), ("c", "cold_process_streams")):
        for name, index in arrays.axis_maps[axis].items():
            flows = np.asarray(data[f"f_{side}_period"], dtype=float)[:, index]
            # A stream with several segments has a zero placeholder there:
            # its flowrate is its segments' duty over its temperature span.
            prefix = "hot" if side == "h" else "cold"
            duty = np.where(
                data[f"{prefix}_segment_mask_period"][:, index],
                data[f"{prefix}_segment_duty_period"][:, index],
                0.0,
            ).sum(axis=1)
            span = np.abs(
                data[f"T_{side}_in_period"][:, index]
                - data[f"T_{side}_out_period"][:, index]
            )
            segmented = (flows <= 0.0) & (duty > 0.0) & (span > 0.0)
            flows = np.where(segmented, duty / np.where(span > 0.0, span, 1.0), flows)
            yield (
                str(name),
                np.asarray(data[f"T_{side}_in_period"], dtype=float)[:, index],
                np.asarray(data[f"T_{side}_out_period"], dtype=float)[:, index],
                flows,
            )

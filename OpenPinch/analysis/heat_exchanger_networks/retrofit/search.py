"""Retrofit search: one modification at a time on an existing network.

Derived from the network evolution method: each depth adds one
modification (a new match or added area on an existing unit) to every
network in a beam, solves each new structure once with the fixed-structure
model's "retrofit" goal, and keeps the best. The best design at each depth
makes the frontier over the number of modifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ....contracts.synthesis.retrofit import (
    HeatExchangerNetworkRetrofitDesign,
    HeatExchangerNetworkRetrofitModification,
    HeatExchangerNetworkRetrofitResult,
)
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ...economics import compute_capital_recovery_factor
from ..context import ensure_target_results
from ..duty_optimisation import (
    DEFAULT_FEASIBILITY_APPROACH,
    _structural_exchangers,
    build_duty_optimisation_request,
    build_solution_network,
    fixed_network_structure,
)
from ..execution.settings import configured_stage_solver, stage_solver_options
from .frontier import mark_dominated
from .moves import NewMatch, apply_new_match, new_splits, screen_new_matches

# Added area below this (m2) is no modification.
_MIN_ADDED_AREA = 1e-3
# An existing unit needing this share of its widened area is at its cap.
_BINDING_SHARE = 0.99


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
    added area).
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


@dataclass(frozen=True)
class _Move:
    kind: str
    exchanger_id: str
    match: NewMatch | None = None
    splits: tuple[str, ...] = ()


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


def run_retrofit_search(
    problem, existing: HeatExchangerNetwork, settings: RetrofitSettings
) -> HeatExchangerNetworkRetrofitResult:
    """Search modifications of ``existing``, one at a time.

    Every exchanger of ``existing`` needs its installed ``area``; a
    ``ValueError`` names those without one. ``max_added_area_fraction`` must
    be None or finite and non-negative.
    """

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
    seen = {_signature(base.state)}
    for _depth in range(max(0, int(settings.max_modifications))):
        children = []
        for parent in beam:
            for state in _children(parent, evaluator.ends, settings):
                signature = _signature(state)
                if signature in seen:
                    continue
                seen.add(signature)
                evaluated = evaluator.evaluate(state)
                if evaluated is not None:
                    children.append(evaluated)
        if not children:
            break
        children.sort(key=lambda child: -child.design.net_annual_benefit)
        beam = children[: max(1, int(settings.beam_width))]
        frontier.append(beam[0].design)
    return HeatExchangerNetworkRetrofitResult(
        frontier=mark_dominated(frontier), evaluated_designs=evaluator.evaluations
    )


def _children(parent: _Evaluated, ends, settings: RetrofitSettings):
    """Each one-modification step from ``parent``: added area on each
    existing unit at its cap, then the screened new matches."""

    state = parent.state
    for exchanger_id in parent.binding:
        yield replace(
            state,
            enlarged=state.enlarged | {exchanger_id},
            moves=state.moves + (_Move("area_addition", exchanger_id),),
        )
    for match in screen_new_matches(
        parent.solved,
        ends,
        approach=settings.approach,
        limit=settings.new_match_candidates,
    ):
        exchanger_id = _new_id(state.network, match)
        move = _Move("new_match", exchanger_id, match, new_splits(state.network, match))
        yield replace(
            state,
            network=apply_new_match(state.network, match, exchanger_id),
            moves=state.moves + (move,),
        )


def _new_id(network: HeatExchangerNetwork, match: NewMatch) -> str:
    taken = {e.exchanger_id for e in network.exchangers}
    stem = f"new:{match.hot}->{match.cold}"
    exchanger_id, n = stem, 1
    while exchanger_id in taken:
        n += 1
        exchanger_id = f"{stem}:{n}"
    return exchanger_id


def _signature(state: _State):
    """Recovery matches by stage and the enlarged units: one per structure."""

    recovery = frozenset(
        (e.source_stream, e.sink_stream, e.stage)
        for e in state.network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY
    )
    return recovery, state.enlarged


class _Evaluator:
    """Solves retrofit structures on one problem's solver arrays."""

    def __init__(self, problem, existing: HeatExchangerNetwork, settings) -> None:
        from ..solver.arrays import problem_to_solver_arrays

        config = problem.master_zone.config
        self.settings = settings
        self.existing = frozenset(
            e.exchanger_id for e in _structural_exchangers(existing)
        )
        self.arrays = problem_to_solver_arrays(problem, 2.0 * config.thermal.dt_cont)
        hens = config.hens
        self.solver = configured_stage_solver(hens, "HENS_SOLVER_EVM")
        self.solver_options = stage_solver_options(hens, "evm", self.solver)
        self.tol = float(hens.solve_tolerance)
        self.names = _existing_names(self.arrays.axis_maps, existing)
        self.ends = {
            self.names[name]: ends for name, ends in _stream_ends(self.arrays).items()
        }
        self.base_utility_cost: float | None = None
        self.evaluations = 0
        costing = config.costing
        self.area_coeff = float(costing.hx_area_coeff)
        self.area_exp = float(costing.hx_area_exp)
        self.recovery_factor = float(
            compute_capital_recovery_factor(costing.discount_rate, costing.service_life)
        )

    def evaluate(self, state: _State) -> _Evaluated | None:
        """Solve ``state``; ``None`` if the solve fails."""

        from ..models.fixed_structure import FixedStructureModel

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
        model.optimise(print_output=False)
        if getattr(model, "mSuccess", 0) != 1:
            return None
        solved = build_solution_network(request, structure, model, self.arrays)
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
                move, results[move.exchanger_id], stages[move.exchanger_id]
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
        return _Evaluated(state, design, _with_names(solved, self.names), binding)

    def _modification(self, move: _Move, result, stage: int | None):
        """``move`` priced from its unit's added area; ``None`` when it does
        nothing (no added area, or a unit with no duty). A new match reports
        ``stage``, its stage in the design's network."""

        added = float(result.added_area)
        if added < _MIN_ADDED_AREA or not result.active:
            return None
        if move.kind == "new_match":
            fixed = self.settings.new_unit_cost + self.settings.split_cost * len(
                move.splits
            )
        else:
            fixed = self.settings.area_addition_cost
        match = move.match
        return HeatExchangerNetworkRetrofitModification(
            kind=move.kind,
            exchanger_id=move.exchanger_id,
            added_area=added,
            fixed_cost=float(fixed),
            area_cost=self.area_coeff * added**self.area_exp,
            hot_stream=match.hot if match else None,
            cold_stream=match.cold if match else None,
            stage=stage if match else None,
            new_stage=bool(match and match.new_stage),
            new_splits=move.splits,
        )


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

"""Networks for problems with only hot or only cold process streams.

With one side alone there is nothing to recover, so the network is heaters
(or coolers) alone and no solver is needed. Each utility level takes the
duty targeting gives it: the composite curve is cascaded from its pinch end,
the least valuable level first, with targeting's own algorithm. The
composite is then cut where one level's duty ends and the next begins, and
each stream's heat between two cuts goes to that level, so a stream spanning
several levels has utility exchangers in series, the least valuable first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ...contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ...domain.configuration import tol
from ...domain.enums import (
    HeatExchangerKind,
    HeatExchangerNetworkDesignMethod,
    StreamID,
)
from ...domain.heat_exchanger import HeatExchanger
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from ...domain.zone import Zone
from ..targeting.utilities import _calculate_assigned_utility_duties, is_default_utility
from .execution.settings import SynthesisWorkflowSettings
from .prepared_problem import PreparedProblem
from .results.assembly import (
    SynthesisWorkflowResult,
    build_synthesis_result,
    demote_infeasible_outcomes,
)
from .results.record import admit
from .solver.arrays import (
    _ordered_stream_items,
    _ordered_utility_items,
    _period_ids_and_weights,
    _segment_identity,
    _segments_for_solver,
    _utility_solver_target,
    _value,
    annual_cost_scales,
)
from .solver.piecewise import (
    PiecewiseThermalProfile,
    duty_aligned_area_contributions,
    utility_thermal_profile,
)

UTILITY_ONLY_TASK_ID = "hens-task-utility-only"


def is_one_sided(zone: Zone) -> bool:
    """Whether ``zone`` has process streams on one side only."""
    return bool(len(zone.hot_streams)) != bool(len(zone.cold_streams))


@dataclass(frozen=True)
class _Piece:
    """One stream segment in one period: real and shifted ends, in K."""

    identity: str
    t_in: float
    t_out: float
    duty: float
    cp: float
    htc: float
    contribution: float


@dataclass(frozen=True)
class _Side:
    """The present side: heaters (cold streams) or coolers (hot streams)."""

    heating: bool

    @property
    def stream_kind(self) -> str:
        return "cold" if self.heating else "hot"

    def shifted(self, temperature: float, contribution: float) -> float:
        return (
            temperature + contribution if self.heating else temperature - contribution
        )


def utility_only_workflow(
    problem: PreparedProblem, settings: SynthesisWorkflowSettings
) -> SynthesisWorkflowResult:
    """The one network a one-sided problem has, as a synthesis workflow result."""
    start = perf_counter()
    task = HeatExchangerNetworkSynthesisTask(
        task_id=UTILITY_ONLY_TASK_ID,
        run_id=settings.run_id,
        method=HeatExchangerNetworkDesignMethod.PinchDesign,
        dt_cont_multiplier=1.0,
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        period_id=settings.period_id,
    )
    network = utility_only_network(problem).model_copy(
        update={
            "run_id": settings.run_id,
            "task_id": task.task_id,
            "method": task.method,
        }
    )
    _refuse_forbidden_pairs(problem, network)
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=network,
        objective_value=network.total_annual_cost,
        solver_status="optimal",
    )
    admit(outcome)
    outcomes = demote_infeasible_outcomes((outcome,))
    return SynthesisWorkflowResult(
        tasks=(task,),
        outcomes=outcomes,
        accepted_result=build_synthesis_result(settings, (task,), outcomes),
        total_run_time=perf_counter() - start,
    )


def _refuse_forbidden_pairs(
    problem: PreparedProblem, network: HeatExchangerNetwork
) -> None:
    """Raise when ``network`` uses a pair ``HENS_FORBIDDEN_MATCHES`` names.

    A one-sided problem's network is set by its utility targets: each level
    carries the duty targeting gives it, on the streams it reaches. There is
    no other match to move a forbidden pair's duty to, so the design is
    refused rather than returned with it.
    """

    def named(name: str, key: str) -> bool:
        return key == name or key.endswith("." + name)

    used = [
        (hot, cold)
        for hot, cold in problem.master_zone.config.hens.forbidden_matches
        if any(
            named(hot, str(e.source_stream)) and named(cold, str(e.sink_stream))
            for e in network.exchangers
        )
    ]
    if used:
        pairs = ", ".join(f"{hot} to {cold}" for hot, cold in used)
        raise ValueError(
            "HENS_FORBIDDEN_MATCHES cannot be honoured: a problem with only "
            "hot or only cold streams has one network, set by its utility "
            f"targets, and it needs {pairs}. Change the utilities the "
            "targets may use (a utility's maximum heat flow, say) instead."
        )


def utility_only_network(problem: PreparedProblem) -> HeatExchangerNetwork:
    """Heaters or coolers serving every stream at the targeted utility levels."""
    zone = problem.master_zone
    side = _Side(heating=bool(len(zone.cold_streams)))
    streams = _ordered_stream_items(
        problem,
        (zone.cold_streams if side.heating else zone.hot_streams).items(),
    )
    utilities = _ordered_utility_items(
        problem,
        (zone.hot_utilities if side.heating else zone.cold_utilities).items(),
    )
    if any(utility.has_segments for _, utility, _ in utilities):
        raise ValueError(
            "a network for only hot or only cold streams needs unsegmented utilities."
        )
    period_ids, weights = _period_ids_and_weights(zone)
    costing = zone.config.costing
    # Only an explicit "price": "auto" keeps targeting's levels here.
    by_price = zone.config.hens.utility_placement == "price"
    price_scale, capital_scale = annual_cost_scales(costing)

    # states[(utility, stream)][n]: the exchanger's duty, slices and temperatures.
    states: dict[tuple[int, int], dict[int, tuple]] = {}
    utility_cost_by_period = []
    load_by_period = []
    for n in range(len(period_ids)):
        pieces = [_pieces(key, stream, n) for key, stream, _ in streams]
        levels = _level_duties(side, pieces, utilities, n, by_price=by_price)
        bands = _bands(side, pieces, utilities, levels, n)
        load_by_period.append(sum(levels))
        utility_cost_by_period.append(
            sum(
                _value(utilities[u][1].price, "$/MW/h", period_idx=n)
                * price_scale
                * duty
                for u, duty in enumerate(levels)
            )
        )
        for u, (low, high) in bands.items():
            for s, stream_pieces in enumerate(pieces):
                exchanger = _band_exchanger(
                    side, stream_pieces, low, high, utilities[u], zone, n, period_ids
                )
                if exchanger is not None:
                    states.setdefault((u, s), {})[n] = exchanger

    exchangers = [
        _exchanger(side, streams[s][0], utilities[u][0], states[(u, s)], period_ids)
        for u, s in sorted(states, key=lambda pair: (pair[1], pair[0]))
    ]
    weight_sum = float(sum(weights))
    utility_cost = (
        sum(w * c for w, c in zip(weights, utility_cost_by_period)) / weight_sum
    )
    capital_cost = float(costing.hx_unit_cost) * capital_scale * len(
        exchangers
    ) + float(costing.hx_area_coeff) * capital_scale * sum(
        exchanger.area ** float(costing.hx_area_exp) for exchanger in exchangers
    )
    load = sum(w * q for w, q in zip(weights, load_by_period)) / weight_sum
    kind = "hot" if side.heating else "cold"
    return HeatExchangerNetwork(
        exchangers=tuple(exchangers),
        objective_value=utility_cost + capital_cost,
        total_annual_cost=utility_cost + capital_cost,
        utility_cost=utility_cost,
        capital_cost=capital_cost,
        summary_metrics={
            "total_units": len(exchangers),
            "recovery_units": 0,
            "hot_utility_units": len(exchangers) if side.heating else 0,
            "cold_utility_units": 0 if side.heating else len(exchangers),
            "hot_utility_load": load if side.heating else 0.0,
            "cold_utility_load": 0.0 if side.heating else load,
            "recovery_load": 0.0,
            "dt_cont_multiplier": 1.0,
        },
        solver_axis_metadata={
            "axis_maps": {
                f"{side.stream_kind}_process_streams": {
                    key: index for index, (key, _, _) in enumerate(streams)
                },
                f"{kind}_utilities": {
                    key: index for index, (key, _, _) in enumerate(utilities)
                },
            }
        },
        source_metadata={
            "solver_framework": "utility only",
            "solver_non_isothermal_model": True,
            "dt_cont_multiplier": 1.0,
            f"{side.stream_kind}_stream_total_duties": [
                sum(piece.duty for piece in _pieces(key, stream, 0))
                for key, stream, _ in streams
            ],
            f"{side.stream_kind}_stream_total_duties_by_period": [
                [
                    sum(piece.duty for piece in _pieces(key, stream, n))
                    for key, stream, _ in streams
                ]
                for n in range(len(period_ids))
            ],
        },
    )


def _pieces(key: str, stream, n: int) -> list[_Piece]:
    """A stream's segments in period ``n``, from supply to target."""
    pieces = []
    for index, segment in enumerate(_segments_for_solver(stream)):
        duty = _value(segment.heat_flow, "kW", period_idx=n)
        if duty <= tol:
            continue
        t_in = _value(segment.supply_temperature, "K", period_idx=n)
        t_out = _value(segment.target_temperature, "K", period_idx=n)
        pieces.append(
            _Piece(
                identity=_segment_identity(key, index),
                t_in=t_in,
                t_out=t_out,
                duty=duty,
                cp=duty / abs(t_out - t_in),
                htc=_value(
                    segment.heat_transfer_coefficient,
                    "kW/m^2/delta_degC",
                    period_idx=n,
                ),
                contribution=_value(
                    segment.effective_delta_t_contribution,
                    "delta_degC",
                    period_idx=n,
                ),
            )
        )
    return pieces


def _shifted_span(side: _Side, piece: _Piece) -> tuple[float, float]:
    low, high = sorted((piece.t_in, piece.t_out))
    return (
        side.shifted(low, piece.contribution),
        side.shifted(high, piece.contribution),
    )


def _composite(
    side: _Side, pieces: Sequence[Sequence[_Piece]], levels: Sequence[float] = ()
):
    """Shifted temperatures, descending, and the heat from the pinch end to each.

    ``levels`` (shifted, K) add rows, as utility levels add problem-table
    intervals: targeting reads the heat a utility can supply at those rows.
    """
    spans = [(_shifted_span(side, p), p.cp) for row in pieces for p in row]
    temperatures = np.array(
        sorted(
            {t for (low, high), _ in spans for t in (low, high)} | set(levels),
            reverse=True,
        )
    )

    def heat_below(t: float) -> float:
        return sum(cp * max(0.0, min(high, t) - low) for (low, high), cp in spans)

    below = np.array([heat_below(t) for t in temperatures])
    # Heaters fill from the cold end, coolers from the hot end.
    heat = below if side.heating else below[0] - below
    return temperatures, heat


def _level_duties(
    side: _Side, pieces, utilities, n: int, *, by_price: bool = False
) -> list[float]:
    """Each utility's duty in period ``n``, as targeting assigns it.

    By price, the cheapest utility serves first (a default one last), each
    as far as its level reaches: on one side alone each level can take all
    the heat beyond it, so this is the cheapest assignment.
    """
    levels = [
        _value(getattr(utility, name), "K", period_idx=n)
        for _, utility, _ in utilities
        for name in ("shifted_minimum_temperature", "shifted_maximum_temperature")
    ]
    temperatures, heat = _composite(side, pieces, levels)
    # Targeting reads each utility's level in the unit it is stored in.
    reference = utilities[0][1].shifted_maximum_temperature
    offset = float(reference[n]) - _value(reference, "K", period_idx=n)
    order = (
        sorted(
            range(len(utilities)),
            key=lambda u: (
                is_default_utility(utilities[u][1]),
                _value(utilities[u][1].price, "$/MW/h", period_idx=n),
                _assignment_rank(side, utilities[u][1], n),
            ),
        )
        if by_price
        else None
    )
    return list(
        _calculate_assigned_utility_duties(
            T_vals=temperatures + offset,
            H_vals=heat,
            u_ls=[utility for _, utility, _ in utilities],
            pinch_row=len(temperatures) - 1 if side.heating else 0,
            is_hot_ut=side.heating,
            is_real_temperatures=False,
            idx=n,
            order=order,
        )
    )


def _bands(
    side: _Side, pieces, utilities, levels: Sequence[float], n: int
) -> dict[int, tuple]:
    """Each used utility's shifted temperature band on the composite.

    Levels are cut from the pinch end in targeting's order: the coldest hot
    utility (or hottest cold utility) first, a default utility last.
    """
    temperatures, heat = _composite(side, pieces)
    order = sorted(
        (u for u, duty in enumerate(levels) if duty > tol),
        key=lambda u: _assignment_rank(side, utilities[u][1], n),
    )
    # np.interp needs ascending heat: from the pinch end.
    ascending_heat = heat[::-1] if side.heating else heat
    ascending_t = temperatures[::-1] if side.heating else temperatures
    bands = {}
    assigned = 0.0
    for u in order:
        start, end = assigned, assigned + levels[u]
        bounds = np.interp([start, end], ascending_heat, ascending_t)
        bands[u] = (min(bounds), max(bounds))
        assigned = end
    return bands


def _assignment_rank(side: _Side, utility, n: int) -> tuple:
    """Targeting's order: least valuable shifted level first, a default last."""
    low = _value(utility.shifted_minimum_temperature, "K", period_idx=n)
    high = _value(utility.shifted_maximum_temperature, "K", period_idx=n)
    level = (high, low) if side.heating else (-low, -high)
    return (is_default_utility(utility), *level)


def _band_exchanger(side, stream_pieces, low, high, utility_item, zone, n, period_ids):
    """The duty and slices of one stream's exchanger in a shifted band, if any."""
    key, utility, _ = utility_item
    duty = 0.0
    real_low, real_high = None, None
    for piece in stream_pieces:
        span_low, span_high = _shifted_span(side, piece)
        overlap_low, overlap_high = max(span_low, low), min(span_high, high)
        if overlap_high - overlap_low <= tol:
            continue
        duty += piece.cp * (overlap_high - overlap_low)
        offset = piece.contribution if side.heating else -piece.contribution
        pieces_low, pieces_high = overlap_low - offset, overlap_high - offset
        real_low = pieces_low if real_low is None else min(real_low, pieces_low)
        real_high = pieces_high if real_high is None else max(real_high, pieces_high)
    if duty <= tol:
        return None
    utility_in = _value(utility.supply_temperature, "K", period_idx=n)
    utility_out = _utility_solver_target(utility, zone, period_idx=n)
    utility_profile = utility_thermal_profile(
        identity=key,
        inlet_temperature=utility_in,
        outlet_temperature=utility_out,
        duty=duty,
        heat_transfer_coefficient=_value(
            utility.heat_transfer_coefficient, "kW/m^2/delta_degC", period_idx=n
        ),
    )
    stream_profile = _profile(stream_pieces)
    if side.heating:
        slices = duty_aligned_area_contributions(
            utility_profile,
            stream_profile,
            duty=duty,
            hot_inlet_temperature=utility_in,
            cold_inlet_temperature=real_low,
            period=period_ids[n],
        )
        stream_in, stream_out = real_low, real_high
    else:
        slices = duty_aligned_area_contributions(
            stream_profile,
            utility_profile,
            duty=duty,
            hot_inlet_temperature=real_high,
            cold_inlet_temperature=utility_in,
            period=period_ids[n],
        )
        stream_in, stream_out = real_high, real_low
    return duty, slices, utility_in, utility_out, stream_in, stream_out


def _profile(pieces: Sequence[_Piece]) -> PiecewiseThermalProfile:
    return PiecewiseThermalProfile(
        identities=tuple(piece.identity for piece in pieces),
        temperatures_in=np.array([piece.t_in for piece in pieces]),
        temperatures_out=np.array([piece.t_out for piece in pieces]),
        duties=np.array([piece.duty for piece in pieces]),
        heat_capacity_flowrates=np.array([piece.cp for piece in pieces]),
        heat_transfer_coefficients=np.array([piece.htc for piece in pieces]),
        temperature_contributions=np.array([piece.contribution for piece in pieces]),
    )


def _exchanger(side, stream_key, utility_key, by_period, period_ids):
    from ...domain._heat_exchanger.period_state import HeatExchangerPeriodState

    period_states = []
    slices = []
    for n, period_id in enumerate(period_ids):
        if n not in by_period:
            period_states.append(
                HeatExchangerPeriodState(
                    period_id=period_id, period_idx=n, duty=0.0, active=False
                )
            )
            continue
        duty, period_slices, utility_in, utility_out, stream_in, stream_out = by_period[
            n
        ]
        slices.extend(period_slices)
        if side.heating:
            temperatures = dict(
                source_inlet_temperature=utility_in,
                source_outlet_temperature=utility_out,
                sink_inlet_temperature=stream_in,
                sink_outlet_temperature=stream_out,
            )
        else:
            temperatures = dict(
                source_inlet_temperature=stream_in,
                source_outlet_temperature=stream_out,
                sink_inlet_temperature=utility_in,
                sink_outlet_temperature=utility_out,
            )
        period_states.append(
            HeatExchangerPeriodState(
                period_id=period_id, period_idx=n, duty=duty, **temperatures
            )
        )
    if side.heating:
        return HeatExchanger(
            exchanger_id=f"hot-utility:{utility_key}->{stream_key}",
            kind=HeatExchangerKind.HOT_UTILITY,
            source_stream=utility_key,
            sink_stream=stream_key,
            source_stream_role=StreamID.Utility,
            sink_stream_role=StreamID.Process,
            period_states=tuple(period_states),
            segment_area_contributions=tuple(slices),
        )
    return HeatExchanger(
        exchanger_id=f"cold-utility:{stream_key}->{utility_key}",
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream=stream_key,
        sink_stream=utility_key,
        source_stream_role=StreamID.Process,
        sink_stream_role=StreamID.Utility,
        period_states=tuple(period_states),
        segment_area_contributions=tuple(slices),
    )


__all__ = [
    "UTILITY_ONLY_TASK_ID",
    "is_one_sided",
    "utility_only_network",
    "utility_only_workflow",
]

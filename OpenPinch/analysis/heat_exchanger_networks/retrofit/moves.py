"""Retrofit modifications applied to a network structure."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger import HeatExchanger
from ....domain.heat_exchanger_network import HeatExchangerNetwork

HEATER = HeatExchangerKind.HOT_UTILITY
COOLER = HeatExchangerKind.COLD_UTILITY


@dataclass(frozen=True)
class NewMatch:
    """A new recovery match between ``hot`` and ``cold`` at a 1-based stage.

    With ``new_stage`` a stage is inserted at ``stage`` and the stages from
    ``stage`` on move one later; otherwise the match joins stage ``stage``.
    """

    hot: str
    cold: str
    stage: int
    new_stage: bool


@dataclass(frozen=True)
class NewUtility:
    """A new utility exchanger at the end of process stream ``stream``: a
    heater served by hot utility ``utility`` (``heater``) or a cooler served
    by cold utility ``utility``."""

    stream: str
    utility: str
    heater: bool


def new_utility_candidates(
    network: HeatExchangerNetwork,
    *,
    hot_streams,
    cold_streams,
    hot_utilities,
    cold_utilities,
) -> tuple[NewUtility, ...]:
    """New end-of-stream utility exchangers ``network`` could take.

    Each cold stream without an end heater may take one from each hot
    utility, then each hot stream without an end cooler one from each cold
    utility; a heater or cooler placed after a stage does not count.
    """

    ends = {
        (e.kind, e.sink_stream if e.kind is HEATER else e.source_stream)
        for e in network.exchangers
        if e.kind is not HeatExchangerKind.RECOVERY and e.stage is None
    }
    heaters = [
        NewUtility(stream, utility, heater=True)
        for stream in sorted(cold_streams)
        if (HEATER, stream) not in ends
        for utility in hot_utilities
    ]
    coolers = [
        NewUtility(stream, utility, heater=False)
        for stream in sorted(hot_streams)
        if (COOLER, stream) not in ends
        for utility in cold_utilities
    ]
    return tuple(heaters + coolers)


def apply_new_utility(
    network: HeatExchangerNetwork, move: NewUtility, exchanger_id: str
) -> HeatExchangerNetwork:
    """``network`` with the utility exchanger ``move`` added at its stream's
    end as ``exchanger_id``, with no installed area."""

    entry = [(move.stream, move.utility)]
    (new,) = HeatExchangerNetwork.from_structure(
        recovery=[],
        heaters=entry if move.heater else (),
        coolers=() if move.heater else entry,
    ).exchangers
    return network.model_copy(
        update={
            "exchangers": network.exchangers
            + (new.model_copy(update={"exchanger_id": exchanger_id}),),
            "stage_count": stage_count(network),
        }
    )


def stage_count(network: HeatExchangerNetwork) -> int:
    """The network's stage count, or its deepest recovery stage."""

    stages = [
        int(e.stage)
        for e in network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY and e.stage is not None
    ]
    return int(network.stage_count or max(stages, default=1))


def screen_new_matches(
    network: HeatExchangerNetwork,
    ends: Mapping[str, tuple[float, float, float]],
    *,
    approach: float,
    limit: int,
    processes: Mapping[str, str] | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[NewMatch, ...]:
    """The ``limit`` most promising new matches for solved ``network``.

    Every hot and cold stream pair is tried joining each stage and in a new
    stage at each position; a pair already matched in a stage is not
    proposed there again. With ``processes`` (stream to its process zone)
    only pairs of one process are tried; a stream it leaves out joins no
    pair. Moves with positive
    ``match_potential`` are ranked by ``utility_relief``, then potential,
    then fewest new splits, then stage, joining before a new stage, and
    stream names. ``weights`` are the period weights the network's duties
    and temperatures are averaged with (``period_mean``).
    """

    hot = sorted(name for name, (supply, target, _) in ends.items() if supply > target)
    cold = sorted(name for name in ends if name not in hot)
    existing = {
        (e.source_stream, e.sink_stream, e.stage)
        for e in network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY
    }
    count = stage_count(network)
    moves = [
        NewMatch(h, c, stage, new_stage)
        for h in hot
        for c in cold
        if processes is None
        or (h in processes and c in processes and processes[h] == processes[c])
        for new_stage, last in ((False, count), (True, count + 1))
        for stage in range(1, last + 1)
        if new_stage or (h, c, stage) not in existing
    ]
    scored = [
        (
            match_potential(network, ends, move, approach=approach, weights=weights),
            move,
        )
        for move in moves
    ]
    ranked = sorted(
        ((potential, move) for potential, move in scored if potential > 0.0),
        key=lambda item: (
            -utility_relief(network, item[1], item[0], weights=weights),
            -item[0],
            len(new_splits(network, item[1])),
            item[1].stage,
            item[1].new_stage,
            item[1].hot,
            item[1].cold,
        ),
    )
    return tuple(move for _potential, move in ranked[: max(0, int(limit))])


def match_potential(
    network: HeatExchangerNetwork,
    ends: Mapping[str, tuple[float, float, float]],
    move: NewMatch,
    *,
    approach: float,
    weights: Sequence[float] | None = None,
    period_ends: Sequence[Mapping[str, tuple[float, float, float]]] | None = None,
) -> float:
    """Heat ``move`` could recover at its position in solved ``network``.

    ``ends`` maps each process stream to its supply and target temperature
    and heat capacity flowrate. The potential is the smaller heat capacity
    flowrate times the temperature difference, less ``approach``, between the
    hot and the cold stream where the match would sit, read from ``network``
    as ``period_mean`` gives it; zero without overlap. With ``period_ends``
    (the same mapping for each period, matching ``weights``) it is the
    ``period_mean`` of ``match_potentials``, each period's own potential, so
    streams that never run together have none.
    """

    if (
        period_ends is not None
        and weights is not None
        and len(weights) == len(period_ends)
    ):
        return period_mean(
            match_potentials(network, period_ends, move, approach=approach), weights
        )
    hot = _hot_temperature(network, ends, move.hot, move.stage, weights)
    cold_boundary = move.stage - 1 if move.new_stage else move.stage
    cold = _cold_temperature(network, ends, move.cold, cold_boundary, weights)
    capacity = min(ends[move.hot][2], ends[move.cold][2])
    return capacity * max(0.0, hot - cold - approach)


def match_potentials(
    network: HeatExchangerNetwork,
    period_ends: Sequence[Mapping[str, tuple[float, float, float]]],
    move: NewMatch,
    *,
    approach: float,
) -> list[float]:
    """Heat ``move`` could recover in each period of solved ``network``.

    ``period_ends`` holds each period's stream ends. A period's potential is
    the smaller of the two heat capacity flowrates in that period (0 for a
    stream that is off) times the temperature difference, less ``approach``,
    between the streams where the match would sit, read from that period.
    """

    cold_boundary = move.stage - 1 if move.new_stage else move.stage
    return [
        min(ends[move.hot][2], ends[move.cold][2])
        * max(
            0.0,
            _hot_temperature(network, ends, move.hot, move.stage, period=n)
            - _cold_temperature(network, ends, move.cold, cold_boundary, period=n)
            - approach,
        )
        for n, ends in enumerate(period_ends)
    ]


def utility_relief(
    network: HeatExchangerNetwork,
    move: NewMatch,
    potential: float,
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Utility a new match of ``potential`` could cut at first order.

    Heat it moves cuts utility only as far as its hot stream is cooled and
    its cold stream heated in the same period: the least of ``potential``,
    the hot stream's cooler duty and the cold stream's heater duty in each
    period of solved ``network``, then ``period_mean`` over the periods. A
    stream cooled only while the other is not heated relieves nothing.
    """

    return joint_relief(
        utility_duties(network, COOLER, move.hot),
        utility_duties(network, HEATER, move.cold),
        weights,
        cap=potential,
    )


def utility_duties(
    network: HeatExchangerNetwork, kind: HeatExchangerKind, stream: str
) -> list[float]:
    """The duty of ``stream``'s coolers (``kind`` cold utility) or heaters
    (hot utility) in each period of solved ``network``; empty with none."""

    states = [
        e.period_states
        for e in network.exchangers
        if e.kind is kind
        and (e.source_stream if kind is COOLER else e.sink_stream) == stream
    ]
    return [sum(float(state.duty) for state in period) for period in zip(*states)]


def joint_relief(
    cooling: Sequence[float],
    heating: Sequence[float],
    weights: Sequence[float] | None,
    *,
    cap: float = math.inf,
) -> float:
    """Utility a link from a cooled to a heated stream could cut.

    ``cooling`` and ``heating`` hold what each side can give and take in
    each period. The link moves the smaller of the two, at most ``cap``, in
    each period; the result is ``period_mean`` of that, not the smaller of
    the two means, so sides that never run together relieve nothing.
    """

    joint = [min(cap, c, h) for c, h in zip(cooling, heating)]
    return period_mean(joint, weights) if joint else 0.0


def period_mean(values: Sequence[float | None], weights: Sequence[float] | None):
    """A solved quantity over the periods: its weighted mean.

    ``values`` holds one reading per period (``None`` where the network has
    none, as for a temperature in a period its exchanger is off); the mean is
    over the periods that have one. Without ``weights``, or when they do not
    match the readings, it is the first period's reading.
    """

    if weights is None or len(weights) != len(values):
        return values[0]
    known = [(float(w), float(v)) for w, v in zip(weights, values) if v is not None]
    total = sum(w for w, _ in known)
    if total <= 0.0:
        return values[0]
    return sum(w * v for w, v in known) / total


def _reading(states, attribute: str, weights, period: int | None):
    """An exchanger's reading: in ``period``, else ``period_mean``."""

    if period is None:
        return period_mean([getattr(state, attribute) for state in states], weights)
    return getattr(states[period], attribute) if period < len(states) else None


def _hot_temperature(
    network, ends, stream: str, stage: int, weights=None, *, period=None
) -> float:
    """A hot stream's temperature as it enters ``stage`` (1-based).

    The inlet of its first exchanger at or after that point, else its target.
    With ``period`` the reading is that period's, from the first such
    exchanger that has one there.
    """

    def key(exchanger: HeatExchanger) -> tuple[float, int]:
        position = math.inf if exchanger.stage is None else float(exchanger.stage)
        return position, int(exchanger.kind is not HeatExchangerKind.RECOVERY)

    readings = [
        (
            key(e),
            _reading(e.period_states, "source_inlet_temperature", weights, period),
        )
        for e in network.exchangers
        if e.source_stream == stream and key(e) >= (stage, 0)
    ]
    downstream = [item for item in readings if period is None or item[1] is not None]
    return min(downstream)[1] if downstream else ends[stream][1]


def _cold_temperature(
    network, ends, stream: str, stage: int, weights=None, *, period=None
) -> float:
    """A cold stream's temperature as it enters ``stage`` (1-based).

    Cold streams pass the stages from the last to the first. The inlet of
    its first exchanger at or after that point, else its target. With
    ``period`` the reading is that period's, from the first such exchanger
    that has one there.
    """

    def key(exchanger: HeatExchanger) -> tuple[float, int]:
        position = math.inf if exchanger.stage is None else -float(exchanger.stage)
        return position, int(exchanger.kind is not HeatExchangerKind.RECOVERY)

    readings = [
        (
            key(e),
            _reading(e.period_states, "sink_inlet_temperature", weights, period),
        )
        for e in network.exchangers
        if e.sink_stream == stream and key(e) >= (-stage, 0)
    ]
    downstream = [item for item in readings if period is None or item[1] is not None]
    return min(downstream)[1] if downstream else ends[stream][1]


def new_splits(network: HeatExchangerNetwork, move: NewMatch) -> tuple[str, ...]:
    """Streams ``move`` splits: those already matched in the stage it joins."""

    if move.new_stage:
        return ()
    matched = {
        stream
        for e in network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY and e.stage == move.stage
        for stream in (e.source_stream, e.sink_stream)
    }
    return tuple(stream for stream in (move.hot, move.cold) if stream in matched)


def apply_relocation(
    network: HeatExchangerNetwork, exchanger_id: str, move: NewMatch
) -> HeatExchangerNetwork:
    """``network`` with recovery exchanger ``exchanger_id`` moved to ``move``.

    The exchanger leaves its match and position and is re-installed, with
    its id and installed area, as the match ``move`` describes (inserting a
    stage as ``apply_new_match`` does). Its old stage stays, even if empty.
    """

    (old,) = (e for e in network.exchangers if e.exchanger_id == exchanger_id)
    remaining = network.model_copy(
        update={
            "exchangers": tuple(e for e in network.exchangers if e is not old),
            "stage_count": stage_count(network),
        }
    )
    moved = apply_new_match(remaining, move, exchanger_id)
    return moved.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": old.area})
                if e.exchanger_id == exchanger_id
                else e
                for e in moved.exchangers
            )
        }
    )


def apply_new_match(
    network: HeatExchangerNetwork, move: NewMatch, exchanger_id: str
) -> HeatExchangerNetwork:
    """``network`` with the new match ``move`` added as ``exchanger_id``.

    Inserting a stage moves every recovery exchanger, and every utility
    exchanger placed after a stage, from that stage on one stage later.
    """

    count = stage_count(network)
    exchangers = list(network.exchangers)
    if move.new_stage:
        exchangers = [
            e.model_copy(update={"stage": e.stage + 1})
            if e.stage is not None and e.stage >= move.stage
            else e
            for e in exchangers
        ]
        count += 1
    (new,) = HeatExchangerNetwork.from_structure(
        recovery=[(move.hot, move.cold, move.stage)]
    ).exchangers
    exchangers.append(new.model_copy(update={"exchanger_id": exchanger_id}))
    return network.model_copy(
        update={"exchangers": tuple(exchangers), "stage_count": count}
    )

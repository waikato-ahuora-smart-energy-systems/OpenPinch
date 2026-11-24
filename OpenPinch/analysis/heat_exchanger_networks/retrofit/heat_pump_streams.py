"""A heat pump of a fixed size as streams of an existing network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ....contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitDesign
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork, NetworkHeatPump
from .moves import NewMatch, apply_new_match, new_splits, stage_count

# Solved network states carry temperatures in K; streams in degC.
_KELVIN = 273.15
# Kelvin a match may fall short of its approach by rounding alone.
_APPROACH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class HeatPumpStep:
    """A heat pump of a fixed size that a retrofit search starts from.

    The problem the search runs on holds the machine's ``condensers`` (hot
    streams) and ``evaporators`` (cold streams). ``record`` is the machine
    with its electricity and compressor costs, and ``base`` the existing
    network's design in the search without the heat pump: its solved
    network places the matches and its utility cost is what savings are
    measured against.
    """

    record: NetworkHeatPump
    condensers: tuple
    evaporators: tuple
    base: HeatExchangerNetworkRetrofitDesign
    # The hot streams of the air the evaporators draw on, if any.
    ambient_sources: tuple = ()
    # A shared design over several periods: each condenser and evaporator
    # by name, the stream in each period (None where it is off).
    in_periods: Mapping[str, tuple] | None = None


@dataclass(frozen=True)
class PlacedMatch:
    """A match added for a heat pump's stream: the new exchanger's id, the
    move that placed it and the streams it split."""

    exchanger_id: str
    match: NewMatch
    splits: tuple[str, ...]


def with_heat_pump_matches(
    network: HeatExchangerNetwork,
    *,
    condensers,
    evaporators,
    approach: float,
    machine: str = "HP",
    solved: HeatExchangerNetwork | None = None,
    ambient_sources=(),
    ambient_approach: float | None = None,
    hot_approaches: Mapping[str, float] | None = None,
    cold_approaches: Mapping[str, float] | None = None,
    in_periods: Mapping[str, Sequence] | None = None,
) -> tuple[HeatExchangerNetwork, tuple[PlacedMatch, ...]]:
    """``network`` with the matches that serve a heat pump's streams, and
    the ids of the exchangers added.

    ``network`` is solved, or ``solved`` is the same network solved: its
    coolers and heaters carry their process
    side temperatures (the first period's without ``in_periods``). Each of
    ``evaporators`` (cold
    streams, degC) is matched to every hot stream whose cooler starts at
    least ``approach`` K above the evaporator's hot end, in a new stage
    after the network's last, hottest evaporator first. Each of
    ``condensers`` (hot streams) is matched to every cold stream whose
    heater starts at least ``approach`` K below the condenser's cold end,
    in a new stage ahead of the network's first, so the hottest condenser
    ends at stage 1. A stream no cooler or heater can serve gets no match.
    ``hot_approaches`` and ``cold_approaches`` map a hot or a cold process
    stream to its own approach in place of ``approach``; a zone may name a
    hot and a cold stream alike. ``in_periods`` maps a condenser or
    evaporator to the stream in each period (``None`` where it is off),
    and a match is placed only where its approach holds in every period
    the level runs, against the cooler or heater as it starts there.
    """

    def needed(process: str, side) -> float:
        return float((side or {}).get(process, approach))

    def each_period(level) -> tuple:
        return tuple(in_periods[level.name]) if in_periods else (level,)

    count = max((len(each_period(s)) for s in (*condensers, *evaporators)), default=1)
    coolers, heaters = (
        [_starts(solved or network, kind, period) for period in range(count)]
        for kind in (HeatExchangerKind.COLD_UTILITY, HeatExchangerKind.HOT_UTILITY)
    )

    def holds(level, process: str, starts, gap, side) -> bool:
        # Each period both run in (a cooler or heater off in a period has
        # no start there), and there must be one: a match that never runs
        # carries no duty.
        together = [
            (period, stream)
            for period, stream in enumerate(each_period(level))
            if stream is not None and process in starts[period]
        ]
        return bool(together) and all(
            gap(stream, starts[period][process]) >= needed(process, side)
            for period, stream in together
        )

    def candidates(starts) -> list[str]:
        """Every process stream with a cooler or heater running in a period."""
        return list(dict.fromkeys(name for each in starts for name in each))

    added: list[PlacedMatch] = []

    def add(hot: str, cold: str, stage: int, new_stage: bool) -> None:
        nonlocal network
        move = NewMatch(hot, cold, stage, new_stage)
        # ``:2``, ``:3`` and so on when the id is taken, as any new match.
        taken = {e.exchanger_id for e in network.exchangers}
        stem = f"new:{machine}:{hot}->{cold}"
        exchanger_id, n = stem, 1
        while exchanger_id in taken:
            n += 1
            exchanger_id = f"{stem}:{n}"
        splits = new_splits(network, move)
        network = apply_new_match(network, move, exchanger_id)
        added.append(PlacedMatch(exchanger_id, move, splits))

    # Rising, each ahead of the last: the hottest condenser ends at stage 1.
    for condenser in sorted(condensers, key=_cold_end):
        served = sorted(
            cold
            for cold in candidates(heaters)
            if holds(
                condenser,
                cold,
                heaters,
                lambda level, start: _cold_end(level) - start,
                cold_approaches,
            )
        )
        for position, cold in enumerate(served):
            add(condenser.name, cold, 1, position == 0)
    # Falling, each after the last.
    for evaporator in sorted(evaporators, key=_hot_end, reverse=True):
        served = sorted(
            hot
            for hot in candidates(coolers)
            if holds(
                evaporator,
                hot,
                coolers,
                lambda level, start: start - _hot_end(level),
                hot_approaches,
            )
        )
        stage = stage_count(network) + 1
        for position, hot in enumerate(served):
            add(hot, evaporator.name, stage, position == 0)
    # The air an evaporator draws on, after everything else: every
    # evaporator at least the approach below it, at both ends of a
    # counter-current match, takes a share. The target places an
    # evaporator at the limit, so the comparison allows rounding.
    gap = approach if ambient_approach is None else float(ambient_approach)

    def under(air, evaporator) -> bool:
        # In every period both run, with ``in_periods``, and there must be
        # one: a match between streams that never run together carries no
        # duty.
        together = [
            (a, e)
            for a, e in zip(each_period(air), each_period(evaporator))
            if a is not None and e is not None
        ]
        return bool(together) and all(
            min(_hot_end(a) - _hot_end(e), _cold_end(a) - _cold_end(e))
            >= gap - _APPROACH_TOLERANCE
            for a, e in together
        )

    for air in ambient_sources:
        below = sorted(
            evaporator.name for evaporator in evaporators if under(air, evaporator)
        )
        stage = stage_count(network) + 1
        for position, cold in enumerate(below):
            add(air.name, cold, stage, position == 0)
    return network, tuple(added)


def _starts(
    network: HeatExchangerNetwork, kind: HeatExchangerKind, period: int = 0
) -> dict[str, float]:
    """The temperature (degC) at which each process stream enters its
    cooler (``kind`` cold utility) or heater (hot utility) in the period
    of index ``period``."""

    cooler = kind is HeatExchangerKind.COLD_UTILITY
    starts: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is not kind:
            continue
        state = exchanger.period_states[period]
        if cooler:
            name, inlet = exchanger.source_stream, state.source_inlet_temperature
        else:
            name, inlet = exchanger.sink_stream, state.sink_inlet_temperature
        if not state.active or inlet is None:
            # Off in this period: it has no start there.
            continue
        if cooler:
            starts[name] = max(starts.get(name, -_KELVIN), float(inlet) - _KELVIN)
        else:
            starts[name] = min(starts.get(name, float("inf")), float(inlet) - _KELVIN)
    return starts


def _cold_end(stream) -> float:
    return min(
        float(stream.supply_temperature.value), float(stream.target_temperature.value)
    )


def _hot_end(stream) -> float:
    return max(
        float(stream.supply_temperature.value), float(stream.target_temperature.value)
    )

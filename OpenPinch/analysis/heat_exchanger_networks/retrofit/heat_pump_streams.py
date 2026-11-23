"""A heat pump of a fixed size as streams of an existing network."""

from __future__ import annotations

from dataclasses import dataclass

from ....contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitDesign
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork, NetworkHeatPump
from .moves import NewMatch, apply_new_match, new_splits, stage_count

# Solved network states carry temperatures in K; streams in degC.
_KELVIN = 273.15


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
) -> tuple[HeatExchangerNetwork, tuple[PlacedMatch, ...]]:
    """``network`` with the matches that serve a heat pump's streams, and
    the ids of the exchangers added.

    ``network`` is solved, or ``solved`` is the same network solved: its
    coolers and heaters carry their process
    side temperatures (first period). Each of ``evaporators`` (cold
    streams, degC) is matched to every hot stream whose cooler starts at
    least ``approach`` K above the evaporator's hot end, in a new stage
    after the network's last, hottest evaporator first. Each of
    ``condensers`` (hot streams) is matched to every cold stream whose
    heater starts at least ``approach`` K below the condenser's cold end,
    in a new stage ahead of the network's first, so the hottest condenser
    ends at stage 1. A stream no cooler or heater can serve gets no match.
    """

    coolers = _starts(solved or network, HeatExchangerKind.COLD_UTILITY)
    heaters = _starts(solved or network, HeatExchangerKind.HOT_UTILITY)
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
            for cold, start in heaters.items()
            if _cold_end(condenser) - start >= approach
        )
        for position, cold in enumerate(served):
            add(condenser.name, cold, 1, position == 0)
    # Falling, each after the last.
    for evaporator in sorted(evaporators, key=_hot_end, reverse=True):
        served = sorted(
            hot
            for hot, start in coolers.items()
            if start - _hot_end(evaporator) >= approach
        )
        stage = stage_count(network) + 1
        for position, hot in enumerate(served):
            add(hot, evaporator.name, stage, position == 0)
    # The air an evaporator draws on, after everything else: every
    # evaporator colder than it takes a share.
    for air in ambient_sources:
        below = sorted(
            evaporator.name
            for evaporator in evaporators
            if _hot_end(evaporator) < _cold_end(air)
        )
        stage = stage_count(network) + 1
        for position, cold in enumerate(below):
            add(air.name, cold, stage, position == 0)
    return network, tuple(added)


def _starts(network: HeatExchangerNetwork, kind: HeatExchangerKind) -> dict[str, float]:
    """The temperature (degC) at which each process stream enters its
    cooler (``kind`` cold utility) or heater (hot utility)."""

    cooler = kind is HeatExchangerKind.COLD_UTILITY
    starts: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is not kind:
            continue
        state = exchanger.period_states[0]
        if cooler:
            name, inlet = exchanger.source_stream, state.source_inlet_temperature
            starts[name] = max(starts.get(name, -_KELVIN), float(inlet) - _KELVIN)
        else:
            name, inlet = exchanger.sink_stream, state.sink_inlet_temperature
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

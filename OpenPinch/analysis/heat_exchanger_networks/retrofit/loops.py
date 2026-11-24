"""Heat recovery loops for a multi-process retrofit.

A heat recovery loop is an intermediate fluid (hot water, say) that
circulates at one temperature level between processes: a hot stream of one
process heats it in a loop cooler at the stream's end, like a cooler, and a
cold stream of another process takes that heat in a loop heater at its end,
like a heater. It is in effect a new intermediate utility level whose heat
balances: what the loop collects it delivers, in every period, with no loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..solver.arrays import PreparedSolverArrays
from .moves import joint_relief, utility_duties

# The loop circulates over this many K about its temperature, as a utility
# needs a temperature change.
LOOP_SPAN = 1.0


@dataclass(frozen=True)
class HeatRecoveryLoop:
    """A loop from hot stream ``hot`` to cold stream ``cold`` (of another
    process) at ``temperature`` (K). ``capacity`` (kW) is the most it can
    carry there, read from the two streams' ends; unlimited when not given."""

    hot: str
    cold: str
    temperature: float
    capacity: float = math.inf


def heat_recovery_loop_candidates(
    ends: Mapping[str, tuple[float, float, float]],
    processes: Mapping[str, str],
    *,
    approach: float,
) -> list[HeatRecoveryLoop]:
    """A loop for each hot and cold stream pair of different processes
    that a loop can carry heat between.

    ``ends`` maps each process stream to its supply and target temperature
    (K) and heat capacity flowrate; a hot stream's supply is above its
    target. ``processes`` maps streams to their process zone; a stream it
    leaves out joins no loop. The loop's temperature and the heat it can
    carry are ``loop_placement``'s.
    """

    candidates = []
    for hot, hot_ends in ends.items():
        if hot_ends[0] <= hot_ends[1] or hot not in processes:
            continue
        for cold, cold_ends in ends.items():
            if (
                cold_ends[0] > cold_ends[1]
                or cold not in processes
                or processes[cold] == processes[hot]
            ):
                continue
            placement = loop_placement(hot_ends, cold_ends, approach=approach)
            if placement is not None:
                candidates.append(HeatRecoveryLoop(hot, cold, *placement))
    return candidates


def loop_placement(
    hot: tuple[float, float, float],
    cold: tuple[float, float, float],
    *,
    approach: float,
) -> tuple[float, float] | None:
    """The temperature (K) of a loop from a hot stream to a cold one, each
    given as supply, target and heat capacity flowrate, and the heat (kW) it
    can carry there; ``None`` when no loop can carry heat between them.

    Where the targets are ``2 * approach + LOOP_SPAN`` apart the loop runs
    midway between them, each target ``approach`` clear of the loop's
    farther end, and can take all of the hot stream's end cooling and give
    all of the cold stream's end heating. Otherwise it serves part of each,
    the hot stream above the loop and the cold stream below it, with their
    own cooler and heater finishing them, and it runs where the heat it can
    collect and the heat it can deliver, both read from the stream ends,
    are largest together.
    """

    hot_supply, hot_target, hot_flow = hot
    cold_supply, cold_target, cold_flow = cold
    margin = approach + 0.5 * LOOP_SPAN
    if hot_target - cold_target >= 2.0 * margin - 1e-9:
        return 0.5 * (hot_target + cold_target), min(
            hot_flow * (hot_supply - hot_target),
            cold_flow * (cold_target - cold_supply),
        )

    # Each loop exchanger is counter-current: the hot stream leaves the loop
    # cooler ``approach`` above the loop's cool end, and the cold stream
    # leaves the loop heater ``approach`` below its warm end.
    outlet = approach - 0.5 * LOOP_SPAN

    def carried(temperature: float) -> float:
        collected = hot_flow * (hot_supply - max(hot_target, temperature + outlet))
        delivered = cold_flow * (min(cold_target, temperature - outlet) - cold_supply)
        return min(collected, delivered)

    # The collected heat falls and the delivered heat rises with the loop's
    # temperature, each flat once it covers its whole stream, so the most
    # they carry together is where they cross or at an end of a flat.
    points = [hot_target - outlet, cold_target + outlet]
    if hot_flow + cold_flow > 0.0:
        points.append(
            (hot_flow * (hot_supply - outlet) + cold_flow * (cold_supply + outlet))
            / (hot_flow + cold_flow)
        )
    # At their inlets the streams face the loop's other ends: the hot stream
    # enters against the warm end and the cold stream against the cool end.
    # ``margin`` from each supply bounds the loop; the heat carried has one
    # peak, so the best within the bounds is the nearest bound.
    lowest, highest = cold_supply + margin, hot_supply - margin
    if lowest > highest:
        return None
    best = max((min(max(p, lowest), highest) for p in points), key=carried)
    return (best, carried(best)) if carried(best) > 0.0 else None


def screen_heat_recovery_loops(
    network: HeatExchangerNetwork,
    candidates: list[HeatRecoveryLoop],
    *,
    limit: int,
    weights: Sequence[float] | None = None,
) -> list[HeatRecoveryLoop]:
    """The ``limit`` loops that could relieve the most utility in solved
    ``network``, most first, ties broken by names.

    A loop takes at most its hot stream's cooler duty and gives at most its
    cold stream's heater duty, in the same period, and no more than its
    ``capacity``, the heat it can carry at its temperature. Those that
    relieve none are left out. The two duties are paired in each period,
    capped, and then averaged with ``weights`` (``moves.joint_relief``),
    else read in the first period.
    """

    relief = {
        loop: joint_relief(
            utility_duties(network, HeatExchangerKind.COLD_UTILITY, loop.hot),
            utility_duties(network, HeatExchangerKind.HOT_UTILITY, loop.cold),
            weights,
            cap=loop.capacity,
        )
        for loop in candidates
    }
    ranked = sorted(
        (loop for loop in candidates if relief[loop] > 0.0),
        key=lambda loop: (-relief[loop], loop.hot, loop.cold),
    )
    return ranked[: max(0, int(limit))]


def _loop_pair(loop: HeatRecoveryLoop) -> str:
    """``hot->cold``, each name with ``\\`` and ``>`` escaped by a backslash,
    so the first unescaped ``>`` is the separator's and no two pairs share
    an id (hot ``A->B`` with cold ``C`` and hot ``A`` with cold ``B->C``
    differ). Names without either character read as given."""

    def escaped(name: str) -> str:
        return name.replace("\\", "\\\\").replace(">", "\\>")

    return f"{escaped(loop.hot)}->{escaped(loop.cold)}"


def loop_cooler_name(loop: HeatRecoveryLoop) -> str:
    """The cold utility through which a loop collects its hot stream's
    heat."""

    return f"Loop {_loop_pair(loop)}:cooler"


def loop_heater_name(loop: HeatRecoveryLoop) -> str:
    """The hot utility through which a loop gives heat to its cold
    stream."""

    return f"Loop {_loop_pair(loop)}:heater"


def with_loop_utilities(
    arrays: PreparedSolverArrays, loops: list[HeatRecoveryLoop]
) -> PreparedSolverArrays:
    """``arrays`` with each loop's heater added as a hot utility, from the
    loop's warm end to its cool end, and its cooler as a cold utility, the
    other way, after the problem's own. Both are free: the loop's heat comes
    from a process stream. ``arrays`` is not changed."""

    half = 0.5 * LOOP_SPAN
    utilities = []
    for loop in loops:
        warm, cool = loop.temperature + half, loop.temperature - half
        utilities.append(("hot", loop_heater_name(loop), warm, cool, 0.0))
        utilities.append(("cold", loop_cooler_name(loop), cool, warm, 0.0))
    return with_constant_utilities(arrays, utilities)


def with_constant_utilities(
    arrays: PreparedSolverArrays,
    utilities: list[tuple[str, str, float, float, float]],
) -> PreparedSolverArrays:
    """``arrays`` with each of ``utilities`` (side "hot" or "cold", name,
    inlet and outlet temperature in K, price per kW) added on its side,
    after the problem's own, as one segment. Film coefficients, area costs
    and temperature contributions are those of the first utility on each
    side. ``arrays`` is not changed."""

    data = {name: values.copy() for name, values in arrays.arrays.items()}
    axis_maps = {axis: dict(names) for axis, names in arrays.axis_maps.items()}
    streams = {axis: list(names) for axis, names in arrays.stream_identities.items()}
    identities = {
        axis: list(names) for axis, names in arrays.utility_identities.items()
    }
    for side, name, inlet, outlet, price in utilities:
        _append_utility(data, side, inlet, outlet, price, f"{name}.S1")
        axis = f"{side}_utilities"
        segments = f"{side}_utility_segments"
        axis_maps[axis][name] = len(axis_maps[axis])
        axis_maps[segments][f"{name}.S1"] = len(axis_maps[segments])
        identities[axis].append(name)
        streams[segments].append(f"{name}.S1")
    return replace(
        arrays,
        arrays=data,
        axis_maps=axis_maps,
        stream_identities=streams,
        utility_identities=identities,
    )


def _append_utility(data, side: str, inlet, outlet, price, segment: str) -> None:
    """Add a constant-temperature utility on ``side`` to ``data``, copying
    the first utility's other properties (a ``None`` value below)."""

    short = "hu" if side == "hot" else "cu"
    long = f"{side}_utility"
    duty = 1.0
    # Key: (axis the utilities run along, the new utility's value).
    values = {
        f"T_{short}_in_period": (1, inlet),
        f"T_{short}_out_period": (1, outlet),
        f"T_{short}_cont_period": (1, None),
        f"htc_{short}_period": (1, None),
        f"{short}_cost_period": (1, price),
        f"{short}_coeff": (0, None),
        f"{short}_exp": (0, None),
        f"{short}_unit_cost": (0, None),
        f"{long}_segment_count": (0, 1),
        f"{long}_parent_segmented": (0, False),
    }
    # One segment, the first: (its value, the value of the others).
    segments = {
        f"{long}_segment_mask_period": (True, False),
        f"{long}_segment_t_in_period": (inlet, 0.0),
        f"{long}_segment_t_out_period": (outlet, 0.0),
        f"{long}_segment_duty_period": (duty, 0.0),
        f"{long}_segment_cp_period": (duty / abs(inlet - outlet), 0.0),
        f"{long}_segment_htc_period": (None, 0.0),
        f"{long}_segment_price_period": (price, 0.0),
        f"{long}_segment_dt_cont_period": (None, 0.0),
        f"{long}_segment_cumulative_duty_period": (0.0, duty),
    }

    def grow(key: str, axis: int):
        template = np.take(data[key], [0], axis=axis)
        data[key] = np.concatenate([data[key], template], axis=axis)
        return data[key]

    for key, (axis, value) in values.items():
        grown = grow(key, axis)
        if value is not None:
            if axis == 0:
                grown[-1] = value
            else:
                grown[:, -1] = value
    for key, (first, rest) in segments.items():
        grown = grow(key, 1)
        kept = grown[:, -1, 0].copy()
        grown[:, -1, :] = rest
        grown[:, -1, 0] = kept if first is None else first
    identities = grow(f"{long}_segment_identities", 0)
    identities[-1, :] = ""
    identities[-1, 0] = segment


__all__ = [
    "LOOP_SPAN",
    "HeatRecoveryLoop",
    "heat_recovery_loop_candidates",
    "loop_cooler_name",
    "loop_heater_name",
    "loop_placement",
    "screen_heat_recovery_loops",
    "with_loop_utilities",
]

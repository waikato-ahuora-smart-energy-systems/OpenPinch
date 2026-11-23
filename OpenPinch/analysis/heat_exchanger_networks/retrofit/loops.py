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
from dataclasses import dataclass

from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..solver.arrays import PreparedSolverArrays
from .heat_pumps import with_constant_utilities
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

"""Heat recovery loops for a multi-process retrofit.

A heat recovery loop is an intermediate fluid (hot water, say) that
circulates at one temperature level between processes: a hot stream of one
process heats it in a loop cooler at the stream's end, like a cooler, and a
cold stream of another process takes that heat in a loop heater at its end,
like a heater. It is in effect a new intermediate utility level whose heat
balances: what the loop collects it delivers, in every period, with no loss.
"""

from __future__ import annotations

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
    process) at ``temperature`` (K)."""

    hot: str
    cold: str
    temperature: float


def heat_recovery_loop_candidates(
    ends: Mapping[str, tuple[float, float, float]],
    processes: Mapping[str, str],
    *,
    approach: float,
) -> list[HeatRecoveryLoop]:
    """A loop for each hot and cold stream pair of different processes
    whose targets leave room for it.

    ``ends`` maps each process stream to its supply and target temperature
    (K) and heat capacity flowrate; a hot stream's supply is above its
    target. ``processes`` maps streams to their process zone; a stream it
    leaves out joins no loop. The loop runs midway between the hot stream's
    target and the cold stream's, so the hot stream reaches its target at
    least ``approach`` above the loop's warm end and the cold stream its
    target at least ``approach`` below the loop's cool end: the targets must
    be ``2 * approach + LOOP_SPAN`` apart. Then the loop can take all of the
    hot stream's end cooling and give all of the cold stream's end heating.
    """

    candidates = []
    for hot, (hot_supply, hot_target, _flow) in ends.items():
        if hot_supply <= hot_target or hot not in processes:
            continue
        for cold, (cold_supply, cold_target, _cold_flow) in ends.items():
            if (
                cold_supply > cold_target
                or cold not in processes
                or processes[cold] == processes[hot]
                or hot_target - cold_target < 2.0 * approach + LOOP_SPAN - 1e-9
            ):
                continue
            candidates.append(
                HeatRecoveryLoop(hot, cold, 0.5 * (hot_target + cold_target))
            )
    return candidates


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
    cold stream's heater duty, in the same period. Those that relieve none
    are left out. The two are paired in each period and then averaged with
    ``weights`` (``moves.joint_relief``), else read in the first period.
    """

    relief = {
        loop: joint_relief(
            utility_duties(network, HeatExchangerKind.COLD_UTILITY, loop.hot),
            utility_duties(network, HeatExchangerKind.HOT_UTILITY, loop.cold),
            weights,
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
    "screen_heat_recovery_loops",
    "with_loop_utilities",
]

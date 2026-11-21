"""Heat pump moves for retrofit.

A heat pump takes heat from a hot stream in an evaporator, at the end of
the stream like a cooler, and delivers it with the compressor work to a
cold stream in a condenser, at its end like a heater. Its temperatures
are fixed per stream pair, so its coefficient of performance is too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ...heat_pumps.common.capital import (
    CompressorDuty,
    MachineComponents,
    installed_capital,
)
from ...heat_pumps.common.shared import calc_carnot_heat_pump_cop
from ...heat_pumps.cycles.limits import CycleLimits
from ...heat_pumps.cycles.vapour_compression_cycle import VapourCompressionCycle
from ..solver.arrays import PreparedSolverArrays


@dataclass(frozen=True)
class HeatPumpCandidate:
    """A heat pump from hot stream ``hot`` to cold stream ``cold``,
    evaporating and condensing at those temperatures (K), with coefficient
    of performance ``cop`` (condenser heat over work) and, for a real-fluid
    cycle, its ``refrigerant``."""

    hot: str
    cold: str
    evaporating: float
    condensing: float
    cop: float
    refrigerant: str | None = None


# A rating gives a heat pump's (COP, refrigerant) from its evaporating and
# condensing temperatures (K), or None when no cycle can serve them.
Rating = Callable[[float, float], "tuple[float, str | None] | None"]


def carnot_rating(eta_ii: float) -> Rating:
    """The HPR service's Carnot heat pump: heating COP
    ``1 + eta_ii * T_evap / (T_cond - T_evap)``, no refrigerant."""

    def rate(evaporating: float, condensing: float):
        return float(calc_carnot_heat_pump_cop(condensing, evaporating, eta_ii)), None

    return rate


def vapour_compression_rating(
    refrigerants: list[str],
    *,
    dtcont: float,
    eta_comp,
    eta_motor: float,
    limits: CycleLimits,
) -> Rating:
    """The HPR service's single-stage vapour compression cycle, on the
    ``refrigerants`` that serve the temperatures within ``limits`` (no
    cycle penalty), the one with the best heating COP kept."""

    def rate(evaporating: float, condensing: float):
        best = None
        for refrigerant in refrigerants:
            cycle = VapourCompressionCycle()
            try:
                cycle.solve(
                    evaporating - 273.15,
                    condensing - 273.15,
                    dtcont=dtcont,
                    eta_comp=eta_comp,
                    eta_motor=eta_motor,
                    refrigerant=refrigerant,
                    Q_heat=1.0,
                    limits=limits,
                )
            except ValueError:
                continue
            if not cycle.solved or float(np.sum(cycle.penalty)) > 0.0:
                continue
            if best is None or cycle.COP_h > best[0]:
                best = (float(cycle.COP_h), refrigerant)
        return best

    return rate


def vapour_compression_capital(
    candidate: HeatPumpCandidate,
    capacity: float,
    areas: tuple[float, float],
    *,
    dtcont: float,
    eta_comp,
    eta_motor: float,
    limits: CycleLimits,
) -> float:
    """The HPR service's installed capital ($) of ``candidate``'s machine
    delivering ``capacity`` kW: its compressor and motor, sized by the cycle
    at that duty, and its evaporator and condenser ``areas`` (m2)."""

    cycle = VapourCompressionCycle()
    cycle.solve(
        candidate.evaporating - 273.15,
        candidate.condensing - 273.15,
        dtcont=dtcont,
        eta_comp=eta_comp,
        eta_motor=eta_motor,
        refrigerant=candidate.refrigerant,
        Q_heat=float(capacity),
        limits=limits,
    )
    compressor = CompressorDuty(
        candidate.refrigerant, cycle.suction_volume_flow, cycle.shaft_power
    )
    return float(
        installed_capital(
            MachineComponents(compressors=(compressor,), exchanger_areas=tuple(areas))
        )
    )


def heat_pump_candidates(
    ends: Mapping[str, tuple[float, float, float]],
    *,
    approach: float,
    rating: Rating,
) -> list[HeatPumpCandidate]:
    """A heat pump for each hot and cold stream pair that needs lift.

    ``ends`` maps each process stream to its supply and target temperature
    (K) and heat capacity flowrate; a hot stream's supply is above its
    target. The evaporator runs ``approach`` below the hot stream's target
    and the condenser ``approach`` above the cold stream's target, so either
    can take its stream all the way there. ``rating`` gives the COP and
    refrigerant. A pair with no lift (a recovery match can serve it), or
    that ``rating`` cannot serve, is left out.
    """

    candidates = []
    for hot, (hot_supply, hot_target, _flow) in ends.items():
        if hot_supply <= hot_target:
            continue
        evaporating = hot_target - approach
        for cold, (cold_supply, cold_target, _cold_flow) in ends.items():
            condensing = cold_target + approach
            if cold_supply > cold_target or condensing <= evaporating:
                continue
            rated = rating(evaporating, condensing)
            if rated is not None:
                cop, refrigerant = rated
                candidates.append(
                    HeatPumpCandidate(
                        hot, cold, evaporating, condensing, cop, refrigerant
                    )
                )
    return candidates


def screen_heat_pumps(
    network: HeatExchangerNetwork,
    candidates: list[HeatPumpCandidate],
    *,
    limit: int,
) -> list[HeatPumpCandidate]:
    """The ``limit`` heat pumps that could relieve the most utility in
    solved ``network`` (first period), most first, ties broken by COP.

    A heat pump's evaporator can take at most its hot stream's cooler duty,
    so its condenser can deliver that times COP / (COP - 1), and at most
    its cold stream's heater duty. Those that relieve none are left out.
    """

    def duty(kind: HeatExchangerKind, stream: str) -> float:
        return sum(
            e.period_states[0].duty
            for e in network.exchangers
            if e.kind is kind
            and (
                e.source_stream
                if kind is HeatExchangerKind.COLD_UTILITY
                else e.sink_stream
            )
            == stream
        )

    relief = {
        candidate: min(
            duty(HeatExchangerKind.COLD_UTILITY, candidate.hot)
            * candidate.cop
            / (candidate.cop - 1.0),
            duty(HeatExchangerKind.HOT_UTILITY, candidate.cold),
        )
        for candidate in candidates
    }
    ranked = sorted(
        (c for c in candidates if relief[c] > 0.0),
        key=lambda c: (-relief[c], -c.cop),
    )
    return ranked[: max(0, int(limit))]


def condenser_name(candidate: HeatPumpCandidate) -> str:
    """The hot utility a heat pump's condenser draws on."""

    return f"HP {candidate.hot}->{candidate.cold}:condenser"


def evaporator_name(candidate: HeatPumpCandidate) -> str:
    """The cold utility a heat pump's evaporator draws on."""

    return f"HP {candidate.hot}->{candidate.cold}:evaporator"


# Condensing and evaporating each span this many K, as a utility needs a
# temperature change.
_PHASE_CHANGE_SPAN = 1.0


def with_heat_pump_utilities(
    arrays: PreparedSolverArrays,
    candidates: list[HeatPumpCandidate],
    electricity_price: float,
) -> PreparedSolverArrays:
    """``arrays`` with each candidate's condenser added as a hot utility
    and its evaporator as a cold utility, after the problem's own.

    The condenser condenses from its temperature over 1 K and costs
    ``electricity_price`` ($/kW/y of work) over its COP per kW of heat, its
    compressor work; the evaporator evaporates up to its temperature and is
    free. Film coefficients, area costs and temperature contributions are
    those of the first utility on each side. ``arrays`` is not changed.
    """

    utilities = []
    for candidate in candidates:
        top = candidate.condensing
        bottom = candidate.evaporating
        utilities.append(
            (
                "hot",
                condenser_name(candidate),
                top,
                top - _PHASE_CHANGE_SPAN,
                float(electricity_price) / candidate.cop,
            )
        )
        utilities.append(
            (
                "cold",
                evaporator_name(candidate),
                bottom - _PHASE_CHANGE_SPAN,
                bottom,
                0.0,
            )
        )
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

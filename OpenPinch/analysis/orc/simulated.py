"""Simulated (CoolProp) ORC: parallel subcritical units below the pinch."""

from __future__ import annotations

import numpy as np

from .carnot import OrcDesign
from .costing import OrcCosts, orc_costs
from .cycle import CRITICAL_MARGIN, OrcCycleError, critical_temperature, solve_orc_cycle
from .inputs import OrcTargetInputs
from .profile import OrcHeatSource
from .search import (
    OrcSearchResult,
    OrcTargetingError,
    initial_points,
    require_beneficial,
    search_orc,
)

__all__ = [
    "evaluate_simulated_orc",
    "optimise_simulated_orc",
    "simulated_orc_design",
    "simulated_orc_window",
]


def simulated_orc_window(
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
    fluid: str,
) -> tuple[float, float] | None:
    """Return the (hottest, coldest) real evaporating temperatures for ``fluid``.

    The hottest is limited by the pinch (the evaporator must sit below it)
    and by the fluid's critical temperature; the coldest is the condensing
    temperature plus the minimum lift.
    """
    try:
        T_crit = critical_temperature(fluid)
    except ValueError:
        return None
    T_hi = min(
        source.T_pinch - inputs.dt_cont - inputs.dt_phase_change,
        T_crit - CRITICAL_MARGIN,
    )
    T_lo = inputs.T_cond + inputs.min_lift
    if T_lo >= T_hi:
        return None
    return T_hi, T_lo


def simulated_orc_design(
    x: np.ndarray,
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
    fluid: str,
) -> OrcDesign | None:
    """Decode ``x`` in [0, 1] into a simulated ORC design with ``fluid``.

    ``x`` has, per unit, a temperature fraction, then a duty fraction, then a
    superheat fraction. Units are placed hottest first; each takes its duty
    fraction of the most its evaporator profile can take from what the
    hotter units left, so the design never needs extra hot utility.
    """
    window = simulated_orc_window(source, inputs, fluid)
    if window is None:
        return None
    T_hi, T_lo = window
    n = int(inputs.n_stages)
    x = np.clip(np.asarray(x, dtype=float).ravel(), 0.0, 1.0)
    if x.size != 3 * n:
        raise ValueError(f"A {n}-unit simulated ORC needs {3 * n} variables.")

    dt = float(inputs.dt_cont)
    load_cap = float(inputs.load_fraction) * source.surplus
    taken: list[tuple[np.ndarray, np.ndarray, float]] = []
    profiles = []
    T_hot, T_evap, superheat, Q_in, W_net = [], [], [], [], []
    T_prev, total = T_hi, 0.0
    for i in range(n):
        T_e = T_prev - x[i] * (T_prev - T_lo)
        sh_max = min(float(inputs.max_superheat), source.T_pinch - dt - T_e)
        sh = x[2 * n + i] * max(sh_max, 0.0)
        try:
            cycle = solve_orc_cycle(
                fluid,
                T_evap=T_e,
                T_cond=inputs.T_cond,
                superheat=sh,
                eta_turbine=inputs.eta_turbine,
                eta_pump=inputs.eta_pump,
                dt_recuperator=inputs.dt_recuperator,
            )
        except OrcCycleError:
            return None
        T_profile, share = cycle.evaporator_profile(inputs.dt_phase_change)
        T_shifted = T_profile + dt
        capacity = source.max_sink_duty(T_shifted, share, tuple(taken))
        Q = x[n + i] * max(min(capacity, load_cap - total), 0.0)
        taken.append((T_shifted, share, Q))
        profiles.append(
            (tuple(float(t) for t in T_shifted), tuple(float(s) for s in share))
        )
        T_hot.append(float(T_shifted[0]))
        T_evap.append(float(T_e))
        superheat.append(float(sh))
        Q_in.append(float(Q))
        W_net.append(float(Q * cycle.eta_thermal))
        T_prev, total = T_e, total + Q

    return OrcDesign(
        T_evap_shifted=tuple(T_hot),
        T_evap=tuple(T_evap),
        T_cond=float(inputs.T_cond),
        Q_in=tuple(Q_in),
        W_net=tuple(W_net),
        fluid=fluid,
        superheat=tuple(superheat),
        profiles=tuple(profiles),
    )


def evaluate_simulated_orc(
    x: np.ndarray,
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
    fluid: str,
) -> tuple[OrcDesign, OrcCosts] | None:
    """Decode ``x`` for ``fluid`` and cost the design."""
    design = simulated_orc_design(x, source, inputs, fluid)
    if design is None:
        return None
    costs = orc_costs(
        W_net=np.asarray(design.W_net),
        Q_in=design.Q_in_total,
        Q_out=design.Q_out_total,
        inputs=inputs,
    )
    return design, costs


def _seed_from_carnot(
    carnot: OrcSearchResult | None,
    window: tuple[float, float],
    n: int,
) -> tuple[float, ...] | None:
    """Map a Carnot design's temperatures onto this fluid's window."""
    if carnot is None:
        return None
    T_hi, T_lo = window
    fractions, T_prev = [], T_hi
    for T_e in carnot.design.T_evap:
        T_e = min(max(T_e, T_lo), T_prev)
        span = T_prev - T_lo
        fractions.append(0.0 if span <= 0.0 else (T_prev - T_e) / span)
        T_prev = T_e
    return tuple(fractions) + (1.0,) * n + (0.0,) * n


def optimise_simulated_orc(
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
    *,
    carnot: OrcSearchResult | None = None,
) -> OrcSearchResult:
    """Return the cheapest simulated ORC over the configured working fluids.

    Each fluid is searched in turn, all units using it, seeded from the
    Carnot design when one is given. Raises ``OrcTargetingError`` when no
    fluid can work in the temperature window or no design saves money.
    """
    n = int(inputs.n_stages)
    best: OrcSearchResult | None = None
    usable = []
    for fluid in inputs.fluids:
        window = simulated_orc_window(source, inputs, fluid)
        if window is None:
            continue
        usable.append(fluid)
        starts = initial_points(n, extra_per_unit=1)
        seed = _seed_from_carnot(carnot, window, n)
        if seed is not None:
            starts = (seed, *starts)
        result = search_orc(
            evaluate_simulated_orc,
            source,
            inputs,
            n_vars=3 * n,
            starts=starts,
            extra=(fluid,),
        )
        if result is not None and (
            best is None
            or result.costs.total_annualized_cost_change
            < best.costs.total_annualized_cost_change
        ):
            best = result
    if not usable:
        raise OrcTargetingError(
            "No configured ORC working fluid can evaporate between the "
            f"{inputs.T_cond:g} degC condensing temperature plus the minimum "
            "lift and the pinch."
        )
    return require_beneficial(best)

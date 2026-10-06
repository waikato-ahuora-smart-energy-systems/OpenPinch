"""Carnot-based ORC model: parallel units on the GCC below the pinch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inputs import OrcTargetInputs
from .profile import OrcHeatSource

__all__ = [
    "OrcDesign",
    "carnot_orc_design",
    "orc_temperature_window",
]

_KELVIN = 273.15


@dataclass(frozen=True)
class OrcDesign:
    """One ORC design: parallel units, hottest first.

    Temperatures are in degC. ``T_evap_shifted`` is the hot end of each
    evaporator on the problem table's shifted scale; ``T_evap`` is the real
    evaporating temperature (for the Carnot model, at the evaporator's cold
    end). ``fluid`` and ``superheat`` are set by the simulated model.
    """

    T_evap_shifted: tuple[float, ...]
    T_evap: tuple[float, ...]
    T_cond: float
    Q_in: tuple[float, ...]
    W_net: tuple[float, ...]
    fluid: str | None = None
    superheat: tuple[float, ...] = ()
    # Each unit's evaporator as (shifted temperatures falling, share of its
    # heat taken at or above each); empty means a step at T_evap_shifted.
    profiles: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] = ()

    @property
    def Q_in_total(self) -> float:
        return float(sum(self.Q_in))

    @property
    def W_net_total(self) -> float:
        return float(sum(self.W_net))

    @property
    def Q_out(self) -> tuple[float, ...]:
        """Condenser duty of each unit, in kW."""
        return tuple(q - w for q, w in zip(self.Q_in, self.W_net, strict=True))

    @property
    def Q_out_total(self) -> float:
        return self.Q_in_total - self.W_net_total

    @property
    def eta_thermal(self) -> float:
        """Net power over heat taken in (0 when no heat is taken)."""
        return self.W_net_total / self.Q_in_total if self.Q_in_total > 0 else 0.0


def orc_temperature_window(
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
) -> tuple[float, float] | None:
    """Return the (hottest, coldest) shifted evaporator temperatures allowed.

    The hot end is the pinch; the cold end is the condensing temperature plus
    the minimum lift and the ORC's temperature contribution. ``None`` means
    the surplus is too cold for an ORC.
    """
    dt_pc = float(inputs.dt_phase_change)
    T_hi = source.T_pinch
    T_lo = max(inputs.T_evap_shifted_min + dt_pc, source.T_min + dt_pc)
    if T_lo >= T_hi:
        return None
    return T_hi, T_lo


def carnot_orc_design(
    x: np.ndarray,
    source: OrcHeatSource,
    inputs: OrcTargetInputs,
) -> OrcDesign | None:
    """Decode a decision vector in [0, 1] into a Carnot ORC design.

    ``x`` has one temperature fraction then one duty fraction per unit. The
    temperature fractions place each unit's evaporator between the previous
    (hotter) one and the coldest allowed temperature. Each unit takes its
    duty fraction of the heat still available at its temperature, so the
    design never needs extra hot utility.
    """
    window = orc_temperature_window(source, inputs)
    if window is None:
        return None
    T_hi, T_lo = window
    n = int(inputs.n_stages)
    x = np.clip(np.asarray(x, dtype=float).ravel(), 0.0, 1.0)
    if x.size != 2 * n:
        raise ValueError(f"A {n}-unit ORC needs {2 * n} decision variables.")

    dt_pc = float(inputs.dt_phase_change)
    load_cap = float(inputs.load_fraction) * source.capacity(T_lo - dt_pc)
    T_s, Q = [], []
    T_prev, taken = T_hi, 0.0
    for i in range(n):
        T_i = T_prev - x[i] * (T_prev - T_lo)
        available = min(source.capacity(T_i - dt_pc), load_cap) - taken
        Q_i = x[n + i] * max(available, 0.0)
        T_s.append(T_i)
        Q.append(Q_i)
        T_prev, taken = T_i, taken + Q_i

    T_s_arr = np.asarray(T_s)
    Q_arr = np.asarray(Q)
    T_evap = T_s_arr - dt_pc - float(inputs.dt_cont)
    W = carnot_orc_work(T_evap, inputs.T_cond, Q_arr, inputs.eta_ii)
    return OrcDesign(
        T_evap_shifted=tuple(float(t) for t in T_s_arr),
        T_evap=tuple(float(t) for t in T_evap),
        T_cond=float(inputs.T_cond),
        Q_in=tuple(float(q) for q in Q_arr),
        W_net=tuple(float(w) for w in W),
        profiles=tuple(((float(t), float(t - dt_pc)), (0.0, 1.0)) for t in T_s_arr),
    )


def carnot_orc_work(
    T_evap: np.ndarray,
    T_cond: float,
    Q_in: np.ndarray,
    eta_ii: float,
) -> np.ndarray:
    """Net power of each unit: ``eta_ii * (1 - T_cond / T_evap) * Q_in``."""
    T_e = np.asarray(T_evap, dtype=float) + _KELVIN
    T_c = float(T_cond) + _KELVIN
    eta_carnot = np.clip(1.0 - T_c / T_e, 0.0, None)
    return float(eta_ii) * eta_carnot * np.maximum(np.asarray(Q_in, dtype=float), 0.0)

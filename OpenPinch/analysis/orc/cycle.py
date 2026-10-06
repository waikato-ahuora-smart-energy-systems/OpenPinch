"""Subcritical organic Rankine cycle solved with CoolProp.

States (per kg of working fluid):

1. saturated liquid at the condensing pressure
2. pump outlet at the evaporating pressure
3. turbine inlet: saturated vapour at the evaporating temperature plus any
   superheat
4. turbine outlet at the condensing pressure

With a recuperator the turbine exhaust preheats the pump outlet, so the
evaporator starts at ``2r`` and the condenser at ``4r``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "OrcCycle",
    "OrcCycleError",
    "critical_temperature",
    "solve_orc_cycle",
]

_KELVIN = 273.15
# Evaporation stays this far below the critical temperature (K).
CRITICAL_MARGIN = 5.0


class OrcCycleError(ValueError):
    """The cycle cannot be solved for these temperatures and fluid."""


@lru_cache(maxsize=None)
def _state(fluid: str):
    """One CoolProp ``AbstractState`` per fluid and process (not picklable)."""
    from ...domain.fluids import build_coolprop_abstract_state

    return build_coolprop_abstract_state(fluid)


@lru_cache(maxsize=None)
def critical_temperature(fluid: str) -> float:
    """Critical temperature of ``fluid``, in degC."""
    return float(_state(fluid).T_critical()) - _KELVIN


@dataclass(frozen=True)
class OrcCycle:
    """A solved cycle, per kg of working fluid (J/kg) unless stated."""

    fluid: str
    T_evap: float
    T_cond: float
    superheat: float
    h1: float
    h2: float
    h2r: float
    h3: float
    h4: float
    h4r: float
    h_bubble: float
    h_dew: float
    T2r: float
    T3: float
    turbine_exit_quality: float | None

    @property
    def q_in(self) -> float:
        """Heat taken from the process per kg."""
        return self.h3 - self.h2r

    @property
    def w_net(self) -> float:
        """Turbine work less pump work per kg."""
        return (self.h3 - self.h4) - (self.h2 - self.h1)

    @property
    def q_out(self) -> float:
        """Condenser duty per kg."""
        return self.h4r - self.h1

    @property
    def eta_thermal(self) -> float:
        return self.w_net / self.q_in if self.q_in > 0.0 else 0.0

    def evaporator_profile(
        self, dt_phase_change: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Real temperatures (degC, strictly falling) and the share of the
        evaporator heat taken at or above each, from the turbine inlet down to
        the evaporator inlet. Evaporation is spread over ``dt_phase_change``
        below the evaporating temperature.
        """
        q = self.q_in
        dt = max(float(dt_phase_change), 1e-6)
        points = [(self.T_evap, self.h3 - self.h_dew)]
        if self.superheat > 0.0:
            points.insert(0, (self.T3, 0.0))
        points.append((self.T_evap - dt, self.h3 - self.h_bubble))
        if self.T2r < self.T_evap - dt:
            points.append((self.T2r, q))
        else:
            points[-1] = (self.T_evap - dt, q)
        T = np.asarray([p[0] for p in points], dtype=float)
        share = np.clip(np.asarray([p[1] for p in points], dtype=float) / q, 0.0, 1.0)
        return T, share


def _sat(fluid: str, T_C: float, quality: float) -> tuple[float, float, float]:
    import CoolProp

    s = _state(fluid)
    s.update(CoolProp.QT_INPUTS, quality, T_C + _KELVIN)
    return s.p(), s.hmass(), s.smass()


def _h_ps(fluid: str, p: float, s_mass: float) -> tuple[float, float]:
    import CoolProp

    st = _state(fluid)
    st.update(CoolProp.PSmass_INPUTS, p, s_mass)
    return st.hmass(), st.Q()


def _hs_pt(fluid: str, p: float, T_C: float) -> tuple[float, float]:
    import CoolProp

    st = _state(fluid)
    st.update(CoolProp.PT_INPUTS, p, T_C + _KELVIN)
    return st.hmass(), st.smass()


def _h_pt(fluid: str, p: float, T_C: float) -> float:
    return _hs_pt(fluid, p, T_C)[0]


def _t_ph(fluid: str, p: float, h: float) -> float:
    import CoolProp

    st = _state(fluid)
    st.update(CoolProp.HmassP_INPUTS, h, p)
    return st.T() - _KELVIN


def solve_orc_cycle(
    fluid: str,
    *,
    T_evap: float,
    T_cond: float,
    superheat: float = 0.0,
    eta_turbine: float = 0.8,
    eta_pump: float = 0.7,
    dt_recuperator: float | None = None,
) -> OrcCycle:
    """Solve a subcritical ORC between two saturation temperatures (degC).

    ``dt_recuperator`` is the recuperator's minimum approach (K); ``None``
    means no recuperator. Raises ``OrcCycleError`` if the evaporating
    temperature is too close to critical or not above the condensing one.
    """
    if not 0.0 < eta_turbine <= 1.0 or not 0.0 < eta_pump <= 1.0:
        raise OrcCycleError("ORC turbine and pump efficiencies must be in (0, 1].")
    if T_evap <= T_cond:
        raise OrcCycleError("The evaporating temperature must exceed condensing.")
    if T_evap > critical_temperature(fluid) - CRITICAL_MARGIN:
        raise OrcCycleError(
            f"{fluid} cannot evaporate at {T_evap:g} degC in a subcritical ORC."
        )
    superheat = max(float(superheat), 0.0)
    try:
        p_cond, h1, s1 = _sat(fluid, T_cond, 0.0)
        p_evap, h_bubble, _ = _sat(fluid, T_evap, 0.0)
        _, h_dew, s_dew = _sat(fluid, T_evap, 1.0)
        h2s, _ = _h_ps(fluid, p_evap, s1)
        h2 = h1 + (h2s - h1) / eta_pump
        if superheat > 0.0:
            h3, s3 = _hs_pt(fluid, p_evap, T_evap + superheat)
        else:
            h3, s3 = h_dew, s_dew
        h4s, _ = _h_ps(fluid, p_cond, s3)
        h4 = h3 - eta_turbine * (h3 - h4s)
        T2 = _t_ph(fluid, p_evap, h2)
        T4 = _t_ph(fluid, p_cond, h4)
        _, h_cond_dew, _ = _sat(fluid, T_cond, 1.0)
        quality = None if h4 >= h_cond_dew else (h4 - h1) / (h_cond_dew - h1)

        q_rec = 0.0
        if dt_recuperator is not None and T4 - dt_recuperator > T2:
            # Hot side cools to T2 + dT, but not into the condensing dome;
            # cold side warms to T4 - dT, but not into evaporation.
            T_hot_out = max(T2 + dt_recuperator, T_cond)
            h_hot_out = (
                _h_pt(fluid, p_cond, T_hot_out)
                if T_hot_out > T_cond + 1e-6
                else h_cond_dew
            )
            T_cold_out = T4 - dt_recuperator
            h_cold_out = (
                _h_pt(fluid, p_evap, T_cold_out)
                if T_cold_out < T_evap - 1e-6
                else h_bubble
            )
            q_rec = max(min(h4 - h_hot_out, h_cold_out - h2, h_bubble - h2), 0.0)
        h2r = h2 + q_rec
        h4r = h4 - q_rec
        T2r = _t_ph(fluid, p_evap, h2r) if q_rec > 0.0 else T2
    except ValueError as exc:  # CoolProp raises ValueError on bad states
        raise OrcCycleError(f"CoolProp could not solve the {fluid} ORC: {exc}") from exc

    cycle = OrcCycle(
        fluid=fluid,
        T_evap=float(T_evap),
        T_cond=float(T_cond),
        superheat=superheat,
        h1=h1,
        h2=h2,
        h2r=h2r,
        h3=h3,
        h4=h4,
        h4r=h4r,
        h_bubble=h_bubble,
        h_dew=h_dew,
        T2r=float(T2r),
        T3=float(T_evap + superheat),
        turbine_exit_quality=quality,
    )
    if not (cycle.q_in > 0.0 and np.isfinite(cycle.w_net)):
        raise OrcCycleError(f"The {fluid} ORC has no valid heat input.")
    return cycle

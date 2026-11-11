"""Simple vapour-compression heat pump cycle utilities built on CoolProp."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import CoolProp
import numpy as np
from scipy.optimize import brentq

from ....domain._stream.linearisation import (
    build_segmented_stream_from_profile,
    get_piecewise_data_points,
)
from ....domain.configuration import tol
from ....domain.fluids import build_coolprop_abstract_state, critical_point
from ....domain.stream_collection import StreamCollection
from ..common.compressor_efficiency import (
    CompressorConditions,
    evaluate_compressor_model,
)
from .limits import CycleLimits

# Qualities this close to 0 or 1 are saturated liquid or vapour; differences
# below it are CoolProp round-off and must not change the cycle's path.
_SATURATION_QUALITY_TOL = 1e-9

__all__ = ["VapourCompressionCycle"]


# More IHX in a re-solve only moves the applied subcooling this little, by rounding.
SUBCOOLING_SETTLED_K = 1e-6


class VapourCompressionCycle:
    """Single vapour-compression heat pump cycle.

    Supports an optional internal heat exchanger.
    """

    STATECOUNT = 6

    @staticmethod
    def _new_cycle_states() -> list[dict[str, float]]:
        """Return empty storage for the six thermodynamic state points."""
        return [{} for _ in range(VapourCompressionCycle.STATECOUNT)]

    def __init__(self):
        """Initialise an unsolved cycle with default operating assumptions."""
        self._system: dict[str, str] = {"temperature": "K"}
        self._cycle_states = self._new_cycle_states()

        self._dT_superheat: float = 0.0
        self._dT_subcool: float = 0.0
        self._eta_comp: float = 1.0
        self._eta_comp_applied: float = 1.0
        self._eta_motor: float = 1.0
        self._ihx_gas_dt: float = 0.0
        self._injection_ratio: float = 0.0
        self._dtcont: float = 0.0
        # Default value used in piecewise approximation of non-linear T-h profiles.
        self._dt_diff_max: float = 0.5
        self._solved: bool = False

        self._state = None
        self._w_net: Optional[float] = None
        self._q_cond: Optional[float] = None
        self._Q_cond: Optional[float] = None
        self._q_cas_heat: Optional[float] = None
        self._Q_cas_heat: Optional[float] = None
        self._q_heat: Optional[float] = None
        self._Q_heat: Optional[float] = None
        self._q_evap: Optional[float] = None
        self._Q_evap: Optional[float] = None
        self._q_cas_cool: Optional[float] = None
        self._Q_cas_cool: Optional[float] = None
        self._q_cool: Optional[float] = None
        self._Q_cool: Optional[float] = None

        self._m_dot: Optional[float] = None
        self._work: Optional[float] = None
        self._penalty: Optional[float] = None
        self._refrigerant: Optional[str] = "water"
        self._T_evap: Optional[float] = None
        self._T_cond: Optional[float] = None
        self.temperature_unit = "K"

    @staticmethod
    def _get_fluid_state(value: str | Any):
        """Build a CoolProp abstract state from a refrigerant spec."""
        return build_coolprop_abstract_state(value)

    @property
    def system(self) -> dict[str, str]:
        """Unit metadata associated with stored cycle-state values."""
        return self._system

    @property
    def state(self):
        """Underlying CoolProp fluid state used during cycle calculations."""
        return self._state

    @state.setter
    def state(self, value) -> None:
        """Set the working-fluid state object and refresh critical properties."""
        self._state = self._get_fluid_state(value)
        self._solved = False
        if isinstance(value, str):
            self._t_crit, self._p_crit, self._d_crit = critical_point(value)
        else:
            self._p_crit = self._state.keyed_output(CoolProp.iP_critical)
            self._t_crit = self._state.keyed_output(CoolProp.iT_critical)
            self._d_crit = self._state.keyed_output(CoolProp.irhomass_critical)

    @property
    def cycle_states(self) -> list[dict[str, float]]:
        """Container holding the six solved cycle states."""
        return self._cycle_states

    @cycle_states.setter
    def cycle_states(self, value: list[dict[str, float]]) -> None:
        """Replace the full state container with a validated six-point cycle."""
        if len(value) != self.STATECOUNT:
            raise ValueError(f"Expected exactly {self.STATECOUNT} state points.")
        self._cycle_states = value

    @property
    def state_points(self) -> list[dict[str, float]]:
        """State points around the cycle."""
        return self._cycle_states

    @property
    def Hs(self) -> Sequence[float]:
        """Specific enthalpies for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["H"] for i in range(self.STATECOUNT)]

    @property
    def Ss(self) -> Sequence[float]:
        """Specific entropies for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["S"] for i in range(self.STATECOUNT)]

    @property
    def Ts(self) -> Sequence[float]:
        """Temperatures for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["T"] for i in range(self.STATECOUNT)]

    @property
    def Ps(self) -> Sequence[float]:
        """Pressures for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["P"] for i in range(self.STATECOUNT)]

    @property
    def q_evap(self) -> Optional[float]:
        """Specific evaporator duty."""
        return self._q_evap

    @property
    def Q_evap(self) -> Optional[float]:
        """Total evaporator duty."""
        return self._Q_evap

    @property
    def q_cas_cool(self) -> Optional[float]:
        """Specific cooling passed to a lower cascade stage."""
        return self._q_cas_cool

    @property
    def Q_cas_cool(self) -> Optional[float]:
        """Total cooling passed to a lower cascade stage."""
        return self._Q_cas_cool

    @property
    def q_cool(self) -> Optional[float]:
        """Specific cooling delivered to the process."""
        return self._q_cool

    @property
    def Q_cool(self) -> Optional[float]:
        """Total cooling delivered to the process."""
        return self._Q_cool

    @property
    def q_cond(self) -> Optional[float]:
        """Specific condenser duty."""
        return self._q_cond

    @property
    def Q_cond(self) -> Optional[float]:
        """Total condenser duty."""
        return self._Q_cond

    @property
    def q_cas_heat(self) -> Optional[float]:
        """Specific heat passed to an upper cascade stage."""
        return self._q_cas_heat

    @property
    def Q_cas_heat(self) -> Optional[float]:
        """Total heat passed to an upper cascade stage."""
        return self._Q_cas_heat

    @property
    def q_heat(self) -> Optional[float]:
        """Specific heat delivered to the process."""
        return self._q_heat

    @property
    def Q_heat(self) -> Optional[float]:
        """Total heat delivered to the process."""
        return self._Q_heat

    @property
    def w_net(self) -> Optional[float]:
        """Specific compressor work input."""
        return self._w_net

    @property
    def work(self) -> Optional[float]:
        """Total compressor work input."""
        return self._work

    @property
    def penalty(self) -> Optional[float]:
        """Total penalty for excessive subcooling."""
        return self._penalty

    @property
    def m_dot(self) -> Optional[float]:
        """Working fluid mass flow rate."""
        return self._m_dot

    @property
    def dtcont(self) -> Optional[float]:
        """Minimum temperature approach carried into derived stream profiles."""
        return self._dtcont

    @dtcont.setter
    def dtcont(self, value: float):
        """Set the minimum approach used when building derived stream profiles."""
        self._dtcont = value

    @property
    def COP_h(self) -> Optional[float]:
        """Heating coefficient of performance based on process heat duty."""
        self._require_solution()
        if abs(self._w_net) <= 1e-9:
            raise ZeroDivisionError(
                "COP_h is undefined when net specific work is zero."
            )
        return self._q_cond / self._w_net

    @property
    def COP_r(self) -> Optional[float]:
        """Cooling coefficient of performance based on process cooling duty."""
        self._require_solution()
        if abs(self._w_net) <= 1e-9:
            raise ZeroDivisionError(
                "COP_r is undefined when net specific work is zero."
            )
        return self._q_evap / self._w_net

    @property
    def dt_diff_max(self) -> Optional[float]:
        """Maximum piecewise temperature error for derived stream profiles."""
        return self._dt_diff_max

    @property
    def refrigerant(self) -> Optional[str]:
        """Refrigerant name used for the solved cycle."""
        return self._refrigerant

    @property
    def T_evap(self) -> Optional[float]:
        """Evaporating temperature in degrees Celsius."""
        if self.temperature_unit == "C":
            return self._T_evap - 273.15 if self._T_evap is not None else None
        return self._T_evap

    @property
    def T_evap_sat_vap(self) -> Optional[float]:
        """Saturated vapour temperature at evaporating pressure in degrees Celsius."""
        if self.temperature_unit == "C":
            return (
                self._T_evap_sat_vap - 273.15
                if self._T_evap_sat_vap is not None
                else None
            )
        return self._T_evap_sat_vap

    @property
    def T_cond(self) -> Optional[float]:
        """Condensing temperature in degrees Celsius."""
        if self.temperature_unit == "C":
            return self._T_cond - 273.15 if self._T_cond is not None else None
        return self._T_cond

    @property
    def T_cond_sat_liq(self) -> Optional[float]:
        """Saturated liquid temperature at condensing pressure in degrees Celsius."""
        if self.temperature_unit == "C":
            return (
                self._T_cond_sat_liq - 273.15
                if self._T_cond_sat_liq is not None
                else None
            )
        return self._T_cond_sat_liq

    @property
    def dT_superheat(self) -> float:
        """Applied compressor-inlet superheat."""
        return self._dT_superheat

    @property
    def dT_subcool(self) -> float:
        """Applied condenser-outlet subcooling."""
        return self._dT_subcool

    @property
    def eta_comp(self) -> float:
        """Isentropic compressor efficiency applied in the last solve."""
        return self._eta_comp_applied

    @property
    def eta_motor(self) -> float:
        """Compressor motor efficiency applied in the last solve."""
        return self._eta_motor

    @property
    def dT_ihx_gas_side(self) -> float:
        """Gas-side temperature change across the internal heat exchanger."""
        return self._ihx_gas_dt

    @property
    def liquid_injection_ratio(self) -> float:
        """Condensate injected into the discharge per kg compressed."""
        return self._injection_ratio

    @property
    def solved(self) -> bool:
        """Flag if the cycle has been solved or not."""
        return self._solved

    def _validate_solve_inputs(
        self,
        refrigerant: str = None,
    ) -> bool:
        if refrigerant is not None:
            self.state = refrigerant
        if self._state is None:
            raise ValueError("A fluid must be specified before solving the cycle.")
        return True

    def _get_P_sat_from_T(
        self,
        T: float,
        Q: float = None,
    ):
        s = self._get_fluid_state(self._refrigerant)
        if (T > self._t_crit - 1) or (Q is None):
            s.update(CoolProp.DmassT_INPUTS, self._d_crit, T)
        else:
            s.update(CoolProp.QT_INPUTS, Q, T)
        return s.p()

    def _compute_state_from_pressure_temperature(
        self,
        P: float,
        T: float,
        *,
        phase: str = 1.0,
    ) -> CoolProp.AbstractState:
        s = self._get_fluid_state(self._refrigerant)
        try:
            s.update(CoolProp.PT_INPUTS, P, T)
        except ValueError:
            s.update(CoolProp.PQ_INPUTS, P, phase)  # Close to saturated liquid/vapour
        return s

    def _solve_single_phase_state_from_pressure_target(
        self,
        P: float,
        target: float,
        property_name: str,
        T_low: float,
        T_high: float,
        *,
        expand: str = "high",
    ) -> CoolProp.AbstractState:
        """Solve a single-phase state from pressure and target property."""
        epsilon = 1e-6
        T_low = max(T_low, epsilon)
        T_high = max(T_high, T_low + 1.0)

        def residual(T: float) -> float:
            state = self._get_fluid_state(self._refrigerant)
            state.update(CoolProp.PT_INPUTS, P, T)
            return getattr(state, property_name)() - target

        low_res = residual(T_low)
        high_res = residual(T_high)
        step = max(T_high - T_low, 10.0)

        for _ in range(25):
            if np.isclose(low_res, 0.0):
                return self._compute_state_from_pressure_temperature(P=P, T=T_low)
            if np.isclose(high_res, 0.0):
                return self._compute_state_from_pressure_temperature(P=P, T=T_high)
            if low_res * high_res < 0.0:
                break

            if expand == "low":
                T_low = max(T_low - step, epsilon)
                low_res = residual(T_low)
            else:
                T_high += step
                high_res = residual(T_high)

            step *= 1.5
        else:
            raise ValueError(
                "Could not bracket a temperature solution for "
                f"{property_name} at pressure {P}."
            )

        T = brentq(residual, T_low, T_high)
        return self._compute_state_from_pressure_temperature(P=P, T=T)

    def _compute_state_from_pressure_quality(
        self,
        P: float,
        Q: float,
    ) -> CoolProp.AbstractState:
        s = self._get_fluid_state(self._refrigerant)
        s.update(CoolProp.PQ_INPUTS, P, Q)
        return s

    def _isobar_points(
        self,
        P: float,
        h_start: float,
        h_end: float,
        num: int,
    ) -> list[list[float]]:
        """Return ``num`` [h, T] points from ``h_start`` to ``h_end`` on isobar ``P``.

        Pure fluids step in enthalpy. A HEOS mixture's pressure-enthalpy flash
        costs tens of milliseconds, so a mixture steps in quality inside the
        two-phase region and in temperature outside it; both give exact points
        on the same isobar. Segment ends keep their exact enthalpies so that
        adjacent profile blocks still share their boundary point.
        """
        if len(self._get_fluid_state(self._refrigerant).fluid_names()) == 1:
            return [
                [h, float(self._compute_state_from_pressure_enthalpy(P=P, h=h).T())]
                for h in np.linspace(h_start, h_end, num)
            ]

        h_liquid = self._compute_state_from_pressure_quality(P, 0.0).hmass()
        h_vapour = self._compute_state_from_pressure_quality(P, 1.0).hmass()
        start, end = (
            self._isobar_end_state(P, h, h_liquid, h_vapour) for h in (h_start, h_end)
        )
        if all(h_liquid <= h <= h_vapour for h in (h_start, h_end)):
            states = [
                self._compute_state_from_pressure_quality(P, Q)
                for Q in np.linspace(start[2], end[2], num)[1:-1]
            ]
        else:
            states = [
                self._compute_state_from_pressure_temperature(P, T)
                for T in np.linspace(start[1], end[1], num)[1:-1]
            ]
        interior = [[float(s.hmass()), float(s.T())] for s in states]
        return [[start[0], start[1]], *interior, [end[0], end[1]]]

    def _isobar_end_state(
        self,
        P: float,
        h: float,
        h_liquid: float,
        h_vapour: float,
    ) -> tuple[float, float, float | None]:
        """Return (h, T, quality) of a profile segment end on a mixture isobar.

        Saturation ends and solved cycle states are already known; only an end
        strictly inside the two-phase region needs a flash, for its quality.
        """
        if h == h_vapour:
            return h, float(self._compute_state_from_pressure_quality(P, 1.0).T()), 1.0
        if h == h_liquid:
            return h, float(self._compute_state_from_pressure_quality(P, 0.0).T()), 0.0
        if not h_liquid < h < h_vapour:
            for point in self._cycle_states:
                if point and point["P"] == P and point["H"] == h:
                    return h, point["T"], None
        state = self._compute_state_from_pressure_enthalpy(P=P, h=h)
        quality = float(state.Q()) if h_liquid < h < h_vapour else None
        return h, float(state.T()), quality

    def _compressor_conditions(
        self, suction: CoolProp.AbstractState, P_out: float
    ) -> CompressorConditions:
        P_in = float(suction.p())
        evap_dew = self._compute_state_from_pressure_quality(P_in, 1.0)
        if P_out >= suction.p_critical():
            # A transcritical discharge has no dew point; the dew line ends at
            # the critical point.
            T_cond_dew, rho_cond_dew = suction.T_critical(), suction.rhomass_critical()
        else:
            cond_dew = self._compute_state_from_pressure_quality(P_out, 1.0)
            T_cond_dew, rho_cond_dew = cond_dew.T(), cond_dew.rhomass()
        return CompressorConditions(
            refrigerant=self._refrigerant,
            P_suction=P_in / 1000.0,
            P_discharge=float(P_out) / 1000.0,
            T_suction=self._convert_K_to_C(float(suction.T())),
            T_evap_dew=self._convert_K_to_C(float(evap_dew.T())),
            T_cond_dew=self._convert_K_to_C(float(T_cond_dew)),
            rho_suction=float(suction.rhomass()),
            rho_evap_dew=float(evap_dew.rhomass()),
            rho_cond_dew=float(rho_cond_dew),
        )

    def _compute_compressor_outlet_state(
        self,
        h_in: float,
        s_in: float,
        P_out: float,
        suction: CoolProp.AbstractState | None = None,
    ) -> CoolProp.AbstractState:
        s = self._get_fluid_state(self._refrigerant)
        try:
            s.update(CoolProp.PSmass_INPUTS, P_out, s_in)
        except ValueError:
            sat_vapour = self._compute_state_from_pressure_quality(P_out, 1.0)
            s = self._solve_single_phase_state_from_pressure_target(
                P=P_out,
                target=s_in,
                property_name="smass",
                T_low=sat_vapour.T() + 1e-6,
                T_high=sat_vapour.T() + 200.0,
                expand="high",
            )

        h_out_isentropic = s.hmass()
        eta = self._eta_comp
        if callable(eta):
            eta = evaluate_compressor_model(
                eta, self._compressor_conditions(suction, P_out)
            )
        self._eta_comp_applied = eta
        h_out = h_in + (h_out_isentropic - h_in) / eta
        return self._compute_state_from_pressure_enthalpy(P=P_out, h=h_out)

    def _compute_state_from_pressure_enthalpy(
        self,
        P: float,
        h: float,
    ):
        s = self._get_fluid_state(self._refrigerant)
        try:
            s.update(CoolProp.HmassP_INPUTS, h, P)
            return s
        except ValueError:
            sat_liquid = self._compute_state_from_pressure_quality(P, 0.0)
            sat_vapour = self._compute_state_from_pressure_quality(P, 1.0)
            h_liquid = sat_liquid.hmass()
            h_vapour = sat_vapour.hmass()

            # Snap states within round-off of saturation, on either side, onto
            # it: CoolProp builds (native, WebAssembly) differ in the last digits
            # of h_liquid and h_vapour, which must not change the state or the
            # branch taken.
            quality = (h - h_liquid) / (h_vapour - h_liquid)
            if -_SATURATION_QUALITY_TOL <= quality <= 1.0 + _SATURATION_QUALITY_TOL:
                if quality > 1.0 - _SATURATION_QUALITY_TOL:
                    quality = 1.0
                elif quality < _SATURATION_QUALITY_TOL:
                    quality = 0.0
                return self._compute_state_from_pressure_quality(P=P, Q=quality)

            if h < h_liquid:
                return self._solve_single_phase_state_from_pressure_target(
                    P=P,
                    target=h,
                    property_name="hmass",
                    T_low=max(sat_liquid.T() - 200.0, 1e-6),
                    T_high=sat_liquid.T() - 1e-6,
                    expand="low",
                )

            return self._solve_single_phase_state_from_pressure_target(
                P=P,
                target=h,
                property_name="hmass",
                T_low=sat_vapour.T() + 1e-6,
                T_high=sat_vapour.T() + 200.0,
                expand="high",
            )

    def _convert_C_to_K(
        self,
        T: float,
    ):
        return T + 273.15

    def _convert_K_to_C(
        self,
        T: float,
    ):
        return T - 273.15

    def _save_cycle_state(
        self,
        state: CoolProp.AbstractState,
        i: int,
    ):
        self._cycle_states[i] = {
            "H": float(state.hmass()),
            "S": float(state.smass()),
            "P": float(state.p()),
            "T": float(state.T()),
        }

    def solve(
        self,
        T_evap: float,
        T_cond: float,
        *,
        dtcont: float,
        dT_superheat: float = 0.0,
        dT_subcool: float = 0.0,
        eta_comp: float = 0.7,
        eta_motor: float = 1.0,
        refrigerant: str = "water",
        dT_ihx_gas_side: float = 10.0,
        dt_ihx: float = 0.0,
        Q_heat: float = None,
        Q_cas_heat: float = 0.0,
        Q_cool: float = None,
        Q_cas_cool: float = 0.0,
        is_heat_pump: bool = True,
        limits: CycleLimits | None = None,
    ) -> float:
        """
        Solve the heat pump cycle for the provided operating point.

        Parameters
        ----------
        T_evap : float
            Dew-point (saturated vapour) temperature in the evaporator [deg C].
        T_cond : float
            Bubble-point (saturated liquid) temperature in the condenser [deg C].
        dtcont : float
            Minimum temperature approach used by HPR targeting [K].
        dT_superheat : float, optional
            Degree of superheating of the suction gas, supplied by the process [K].
        dT_subcool : float, optional
            Degree of subcooling after the condenser, heat delivered to the process [K].
        eta_comp : float, optional
            Isentropic efficiency of the compressor [-].
        eta_motor : float, optional
            Efficiency of the compressor motor [-]. Motor losses raise the
            electrical work and leave the cycle duties unchanged.
        refrigerant : str, optional
            Cycle refrigerant; supports multi-component fluids.
        dT_ihx_gas_side : float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        dt_ihx : float, optional
            Minimum approach at the internal heat exchanger's hot end: the
            suction gas leaves it at least this far below the entering liquid [K].
        Q_heat : float, optional
            Heat delivered to the process [W]. Used for heat-pump and cascade
            configurations only.
        Q_cas_heat : float, optional
            Extra condenser heat transferred to the next cascade cycle [W].
            Used only for cascade heat pump configurations.
        Q_cool : float, optional
            Cooling delivered to the process [W]. Used for refrigeration and
            cascade configurations only.
        Q_cas_cool : float, optional
            Extra evaporator cooling transferred to the next cascade cycle [W].
            Used only for cascade refrigeration configurations.
        is_heat_pump : bool, optional
            Flag to indicate if the cycle is in heat pump or refrigeration mode.
        limits : CycleLimits, optional
            Practical limits: liquid injection to a maximum discharge superheat,
            and penalties for the pressure ratio, discharge and suction
            pressures and the lift. A two-phase discharge is always penalised.

        Returns
        -------
        float
            Compressor electrical power for the solved operating point [W].
        """
        self._solved = False
        self.temperature_unit = "K"
        self._validate_solve_inputs(refrigerant)
        self._penalty = []
        self._cycle_states = self._new_cycle_states()

        self._refrigerant = refrigerant
        self._T_evap = self._convert_C_to_K(T_evap)
        self._T_cond = self._convert_C_to_K(T_cond)
        self._dtcont = float(dtcont)
        self._dT_superheat = dT_superheat
        self._dT_subcool = dT_subcool

        if is_heat_pump:
            self._Q_heat = np.float64(1.0) if Q_heat is None else Q_heat
            self._Q_cas_heat = Q_cas_heat if Q_cas_heat is not None else 0.0
            self._Q_cool = Q_cool
            self._Q_cas_cool = 0.0
            self._Q_cond = self._Q_heat + self._Q_cas_heat
        else:  # is refrigeration
            self._Q_heat = Q_heat
            self._Q_cas_heat = 0.0
            self._Q_cool = np.float64(1.0) if Q_cool is None else Q_cool
            self._Q_cas_cool = Q_cas_cool if Q_cas_cool is not None else 0.0
            self._Q_evap = self._Q_cool + self._Q_cas_cool

        self._eta_comp = eta_comp
        if not 0.0 < float(eta_motor) <= 1.0:
            raise ValueError("eta_motor must be in the interval (0, 1].")
        self._eta_motor = float(eta_motor)
        # The suction gas leaves the IHX at least dt_ihx below the liquid
        # entering it, which has left the condenser dT_subcool below T_cond.
        self._ihx_gas_dt = max(
            min(
                dT_ihx_gas_side,
                self._T_cond
                - self._T_evap
                - self._dT_subcool
                - self._dT_superheat
                - float(dt_ihx),
            ),
            0.0,
        )

        # CoolProp extrapolates saturation below Tmin without error.
        if self._T_evap < self._state.Tmin():
            raise ValueError(
                "Evaporating temperature is below the refrigerant's minimum "
                "temperature."
            )

        P_lo = self._get_P_sat_from_T(self._T_evap, Q=1.0)
        P_hi = self._get_P_sat_from_T(self._T_cond, Q=0.0)

        if P_lo > P_hi:
            raise ValueError("Evaporator pressure must be below condenser pressure.")

        """Solve a basic four-state cycle from temperatures and pressures."""
        # 0 - Evaporator outlet / IHX inlet
        self._T_evap_sat_vap = self._compute_state_from_pressure_quality(P_lo, 1.0).T()
        state0 = self._compute_state_from_pressure_temperature(
            P=P_lo,
            T=self._T_evap_sat_vap + self._dT_superheat,
            phase=1,
        )
        self._save_cycle_state(state0, 0)
        h_ihx_in = state0.hmass()

        if dT_ihx_gas_side > self._ihx_gas_dt:
            # Penalise insufficient superheat in internal heat exchange
            h0_tar = self._compute_state_from_pressure_temperature(
                P=P_lo,
                T=self._T_evap_sat_vap + dT_ihx_gas_side,
                phase=1,
            ).hmass()
            self._penalty.append(h0_tar - h_ihx_in)

        # IHX outlet / compressor inlet
        state_ihx_out = self._compute_state_from_pressure_temperature(
            P=P_lo,
            T=self._T_evap_sat_vap + self._dT_superheat + self._ihx_gas_dt,
            phase=1,
        )
        h_ihx_out = state_ihx_out.hmass()
        dh_ihx = h_ihx_out - h_ihx_in

        # 1 - Compressor discharge (real), compressing from the IHX outlet state
        state1 = self._compute_compressor_outlet_state(
            h_in=h_ihx_out,
            s_in=state_ihx_out.smass(),
            P_out=P_hi,
            suction=state_ihx_out,
        )
        self._save_cycle_state(state1, 1)

        # A compressor cannot discharge two-phase: penalise gas short of its
        # dew point (a dry fluid compressed from saturated vapour, say).
        if P_hi < self._p_crit:
            h_dew_hi = self._compute_state_from_pressure_quality(P_hi, 1.0).hmass()
            if state1.hmass() < h_dew_hi:
                self._penalty.append(h_dew_hi - state1.hmass())

        # 2 - Condenser outlet / IHX inlet (source)
        h_cond_out_min = (
            self._compute_state_from_pressure_quality(
                P=P_lo,
                Q=0,
            ).hmass()
            + dh_ihx
        )  # Find the limit to subcooling of the suction gas
        # Saturated liquid at the condenser pressure, or for a transcritical
        # stage the gas at T_cond on the supercritical isobar.
        condensing_reference = self._condensing_reference_state(P_hi)
        self._T_cond_sat_liq = condensing_reference.T()
        T_cond_out_start = (
            min(self._T_cond_sat_liq, self._t_crit - 0.1)
            if P_hi < self._p_crit
            else self._T_cond_sat_liq
        )
        h_cond_out_tar = self._compute_state_from_pressure_temperature(
            P=P_hi,
            T=T_cond_out_start - self._dT_subcool,
            phase=0,
        ).hmass()
        h_cond_out_cascade = (
            self._coldest_outlet_serving_cascade_heat(P_hi, state1.hmass())
            if is_heat_pump
            else self._coldest_outlet_serving_cascade_heat_given_cooling(
                P_hi, h_ihx_out
            )
        )
        h_cond_out = max(h_cond_out_min, h_cond_out_tar, h_cond_out_cascade)
        state2 = self._compute_state_from_pressure_enthalpy(
            P=P_hi,
            h=h_cond_out,
        )
        # Actual subcooling possible; differences within round-off of the
        # condensing reference are none.
        dT_subcool_applied = self._T_cond_sat_liq - state2.T()
        self._dT_subcool = (
            dT_subcool_applied if dT_subcool_applied > SUBCOOLING_SETTLED_K else 0.0
        )
        self._save_cycle_state(state2, 2)
        self._injection_ratio = self._liquid_injection_ratio(state1, state2, limits)

        if (
            dT_ihx_gas_side > self._ihx_gas_dt
            and h_cond_out_cascade > max(h_cond_out_min, h_cond_out_tar)
            and dT_subcool - self._dT_subcool > SUBCOOLING_SETTLED_K
        ):
            # The IHX was limited by subcooling this stage cannot have.
            return self.solve(
                T_evap=T_evap,
                T_cond=T_cond,
                dtcont=dtcont,
                dT_superheat=dT_superheat,
                dT_subcool=float(self._dT_subcool),
                eta_comp=eta_comp,
                eta_motor=eta_motor,
                refrigerant=refrigerant,
                dT_ihx_gas_side=dT_ihx_gas_side,
                dt_ihx=dt_ihx,
                Q_heat=Q_heat,
                Q_cas_heat=Q_cas_heat,
                Q_cool=Q_cool,
                Q_cas_cool=Q_cas_cool,
                is_heat_pump=is_heat_pump,
                limits=limits,
            )

        if h_cond_out_min > h_cond_out_tar:
            # Penalise excessive subcooling beyond the maximum allowed
            self._penalty.append(h_cond_out_min - h_cond_out_tar)

        if self._cycle_states[1]["H"] <= self._cycle_states[2]["H"]:
            raise ValueError(
                "Condenser cannot have a negative or zero enthalpy change."
            )

        # 3 - Expansion valve outlet / evaporator inlet
        state3 = self._compute_state_from_pressure_enthalpy(
            P=P_lo,
            h=state2.hmass() - dh_ihx,
        )
        self._save_cycle_state(state3, 3)

        self._q_evap = state0.hmass() - state3.hmass()
        self._q_cond = state1.hmass() - state2.hmass()
        self._penalty.extend(
            self._q_cond * violation
            for violation in self._limit_violations(P_lo, P_hi, limits)
        )
        self._w_net = (self._q_cond - self._q_evap) / self._eta_motor
        if is_heat_pump:
            self._m_dot = self._Q_cond / self._q_cond
            self._Q_evap = self._m_dot * self._q_evap
        else:  # is refrigeration
            self._m_dot = self._Q_evap / self._q_evap
            self._Q_cond = self._m_dot * self._q_cond

        self._work = self._m_dot * self._w_net

        # Extra states for cascade heat exchangers
        # Condenser side - state 4
        # Evaporator side - state 5
        if is_heat_pump:
            # 4 - Hot side of cascade heat exchanger outlet (otherwise state 4 == 1)
            self._q_cas_heat = (
                self._Q_cas_heat / self._m_dot if self._m_dot > 0.0 else 0.0
            )
            self._q_heat = self._q_cond - self._q_cas_heat
            state4 = self._compute_state_from_pressure_enthalpy(
                P=state2.p(),
                h=state2.hmass() + self._q_heat,
            )
            self._save_cycle_state(state4, 4)

            # 5 - Hot side of cascade heat exchanger outlet (otherwise state 5 == 3)
            if self._Q_cool is None or np.isnan(self._Q_cool):
                self._Q_cool = self._Q_evap
            elif self._Q_cool > self._Q_evap:
                self._Q_cool = self._Q_evap

            self._q_cool = (
                self._Q_cool / self._m_dot if self._m_dot > 0 else self._q_evap
            )

            state5 = self._compute_state_from_pressure_enthalpy(
                P=state3.p(),
                h=state3.hmass() + self._q_cool,
            )
            self._save_cycle_state(state5, 5)
            self._q_cas_cool = self._q_evap - self._q_cool
            self._Q_cas_cool = self._Q_evap - self._Q_cool

        else:  # is refrigeration
            # 5 - Hot side of cascade heat exchanger outlet (otherwise state 5 == 3)
            self._q_cas_cool = (
                self._Q_cas_cool / self._m_dot if self._m_dot > 0.0 else 0.0
            )
            self._q_cool = self._q_evap - self._q_cas_cool
            state5 = self._compute_state_from_pressure_enthalpy(
                P=state3.p(),
                h=state3.hmass() + self._q_cool,
            )
            self._save_cycle_state(state5, 5)

            # 4 - Hot side of cascade heat exchanger outlet (otherwise state 4 == 1)
            if self._Q_heat is None or np.isnan(self._Q_heat):
                self._Q_heat = self._Q_cond
            elif self._Q_heat > self._Q_cond:
                self._Q_heat = self._Q_cond
            self._q_heat = (
                self._Q_heat / self._m_dot if self._m_dot > 0 else self._q_cond
            )
            state4 = self._compute_state_from_pressure_enthalpy(
                P=state2.p(),
                h=state2.hmass() + self._q_heat,
            )
            self._save_cycle_state(state4, 4)
            self._q_cas_heat = self._q_cond - self._q_heat
            self._Q_cas_heat = self._Q_cond - self._Q_heat

        # Finish analysis
        self._penalty = np.asarray(self._penalty) * self._m_dot
        self.temperature_unit = "C"
        if not np.isfinite(float(self._work)) or float(self._work) < 0.0:
            return self._work
        self._solved = True
        return self._work

    def build_stream_collection(
        self,
        include_cond: bool = False,
        include_evap: bool = False,
        is_process_stream: bool = False,
        dtcont: float = 0.0,
        dt_diff_max: float = 0.5,
    ) -> StreamCollection:
        """Approximate condenser and evaporator duties as piecewise stream segments."""
        self._require_solution()
        self._dtcont = dtcont
        self._dt_diff_max = dt_diff_max
        streams = StreamCollection()

        def _build_streams(profile: np.ndarray, is_condenser: bool = True):
            sc = StreamCollection()
            if self._m_dot <= 0.0 or np.ptp(np.asarray(profile)[:, 0]) <= 0.0:
                return sc
            name = "Condenser" if is_condenser else "Evaporator"
            is_hot_profile = bool(profile[0, 1] > profile[-1, 1])
            if np.isclose(profile[0, 1], profile[-1, 1]):
                is_hot_profile = is_condenser
            sc.add(
                build_segmented_stream_from_profile(
                    name=name,
                    profile=profile,
                    heat_scale=self._m_dot,
                    is_hot_stream=is_hot_profile,
                    is_process_stream=is_process_stream,
                    # HPR_DT_CONT is applied once, to the background profiles;
                    # cycle streams carry none of their own, as Carnot streams.
                    delta_t_contribution=0.0,
                )
            )
            return sc

        if include_cond and abs(float(self._Q_heat)) > tol:
            streams += _build_streams(
                self._build_condenser_profile(),
                is_condenser=True,
            )
        if include_evap and abs(float(self._Q_cool)) > tol:
            streams += _build_streams(
                self._build_evaporator_profile(),
                is_condenser=False,
            )
        return streams

    def _coldest_outlet_serving_cascade_heat(self, P_hi: float, h_in: float) -> float:
        """Lowest condenser outlet enthalpy that keeps cascade heat above saturation.

        The cascade exchanger takes the hottest of the condenser's heat and the
        process duty the coldest. The exchanger's evaporator is colder than
        condensation by only the cascade gap, so subcooled liquid cannot heat it:
        subcooling heat must all fit in the process duty.
        """
        if self._Q_cas_heat <= tol:
            return -np.inf
        h_sat_liq = self._condensing_reference_state(P_hi).hmass()
        q_subcool_max = (h_in - h_sat_liq) * max(self._Q_heat, 0.0) / self._Q_cas_heat
        return h_sat_liq - q_subcool_max

    def _coldest_outlet_serving_cascade_heat_given_cooling(
        self, P_hi: float, h_comp_in: float
    ) -> float:
        """The same subcooling limit when evaporator duty sets the flow.

        Refrigeration fixes the evaporator duty, so the outlet also moves the
        flow. With ``q_evap = h_comp_in - h_out``, requiring the subcooling heat
        to fit in the process duty is linear in ``h_out``.
        """
        if self._Q_heat is None or np.isnan(self._Q_heat) or self._Q_evap <= tol:
            return -np.inf
        share = max(self._Q_heat, 0.0) / self._Q_evap
        if share >= 1.0:
            return -np.inf
        h_sat_liq = self._condensing_reference_state(P_hi).hmass()
        return (h_sat_liq - share * h_comp_in) / (1.0 - share)

    def _condensing_reference_state(self, P_hi: float) -> CoolProp.AbstractState:
        """Where condensing ends and subcooling starts on the high-side isobar.

        Below the critical pressure this is the saturated liquid. A
        transcritical stage has no saturation: its gas cooler is measured
        from ``T_cond``, so subcooling is the glide below it.
        """
        if P_hi < self._p_crit:
            return self._compute_state_from_pressure_quality(P=P_hi, Q=0)
        return self._compute_state_from_pressure_temperature(P=P_hi, T=self._T_cond)

    def _limit_violations(
        self, P_lo: float, P_hi: float, limits: CycleLimits | None
    ) -> list[float]:
        """Relative violations of the stage's pressure and lift limits."""
        if limits is None:
            return []
        violations = []
        if limits.max_pressure_ratio is not None:
            violations.append(P_hi / P_lo / limits.max_pressure_ratio - 1.0)
        max_discharge = (
            limits.max_discharge_pressure_co2
            if limits.max_discharge_pressure_co2 is not None
            and self._get_fluid_state(self._refrigerant).fluid_names()
            == ["CarbonDioxide"]
            else limits.max_discharge_pressure
        )
        if max_discharge is not None:
            violations.append(P_hi / max_discharge - 1.0)
        if limits.min_suction_pressure is not None:
            violations.append(limits.min_suction_pressure / P_lo - 1.0)
        if limits.min_lift > 0.0:
            lift = self._T_cond_sat_liq - self._T_evap_sat_vap
            violations.append(1.0 - lift / limits.min_lift)
        return [violation for violation in violations if violation > 0.0]

    def _liquid_injection_ratio(
        self,
        discharge: CoolProp.AbstractState,
        condensate: CoolProp.AbstractState,
        limits: CycleLimits | None,
    ) -> float:
        """Condensate injected per kg compressed to hold the discharge superheat.

        Condensate at the condenser outlet is mixed into the discharge gas
        until the gas is ``max_discharge_superheat`` above its dew point. The
        mixing is adiabatic, so the condenser rejects the same heat and the
        compressor does the same work.
        """
        if limits is None or limits.max_discharge_superheat is None:
            return 0.0
        P_hi = discharge.p()
        if P_hi >= self._p_crit:
            return 0.0
        T_cap = self._compute_state_from_pressure_quality(P_hi, 1.0).T() + float(
            limits.max_discharge_superheat
        )
        if discharge.T() <= T_cap:
            return 0.0
        h_cap = self._compute_state_from_pressure_temperature(
            P=P_hi, T=T_cap, phase=1
        ).hmass()
        return (discharge.hmass() - h_cap) / (h_cap - condensate.hmass())

    def _build_condenser_profile(self) -> np.ndarray:
        """
        Construct a four-point condenser T-h polyline in cycle units.

        Returns
        -------
        np.ndarray
            Array of shape (n,2): [enthalpy, temperature].
            Enthalpy always in J/kg, Temperature in K.
        """
        # Ensure the cycle has been solved
        self._require_solution()

        # Read temperatures and enthalpies from the solved cycle
        H = list(self.Hs)  # [H0, H1, H2, H3] in J/kg
        # With liquid injection, (1 + r) kg flows through the condenser per kg
        # compressed, so the profile is traced in condenser-flow enthalpy and
        # stretched back to per kg compressed at the end.
        stretch = 1.0 + self._injection_ratio
        H[4] = H[2] + (H[4] - H[2]) / stretch

        # Use saturation points from the state if needed
        p_high = self.Ps[4]
        t_h_curve_points = []

        if p_high < self._p_crit and H[4] > H[2]:
            # Saturated vapor at condenser pressure
            h_sat_vapor = self._compute_state_from_pressure_quality(p_high, 1.0).hmass()

            # Saturated liquid at condenser pressure
            h_sat_liquid = self._compute_state_from_pressure_quality(
                p_high, 0.0
            ).hmass()

            if H[4] > h_sat_vapor:
                t_h_curve_points.extend(
                    self._isobar_points(p_high, H[4], max(h_sat_vapor, H[2]), 21)
                )

            if H[4] > h_sat_liquid and not (H[2] > h_sat_vapor):
                for point in self._isobar_points(
                    p_high, min(h_sat_vapor, H[4]), max(h_sat_liquid, H[2]), 21
                ):
                    if not _repeats_last_point(t_h_curve_points, point[0]):
                        t_h_curve_points.append(point)

            if H[2] < h_sat_liquid:
                for point in self._isobar_points(
                    p_high, min(h_sat_liquid, H[4]), H[2], 21
                ):
                    if not _repeats_last_point(t_h_curve_points, point[0]):
                        t_h_curve_points.append(point)

        else:
            # Determine supercritical gas cooler profile
            for h in np.linspace(H[4], H[2], 61):
                state = self._compute_state_from_pressure_enthalpy(P=p_high, h=h)
                t_h_curve_points.append([h, float(state.T())])

        t_h_curve_points = np.array(t_h_curve_points)
        t_h_curve_points[:, 0] = H[2] + (t_h_curve_points[:, 0] - H[2]) * stretch

        # Convert temperature to °C if not SI
        t_h_curve_points[:, 1] -= 273.15

        self._condenser_th_curve = t_h_curve_points

        # Calculate a piecewise-linear approximation of the condenser profile.
        condenser_profile = get_piecewise_data_points(
            curve=t_h_curve_points,
            is_hot_stream=True,
            dt_diff_max=self._dt_diff_max,
        )
        self._condenser_profile = condenser_profile
        return condenser_profile

    def _build_evaporator_profile(self) -> np.ndarray:
        """
        Construct a three-point evaporator T-h polyline in cycle units.

        Returns
        -------
        np.ndarray
            Array of shape (3,2): [enthalpy, temperature].
            Enthalpy always in J/kg. Temperature in °C.
        """
        # Ensure the cycle has been solved
        self._require_solution()

        # Read temperatures and enthalpies from the solved cycle
        H = self.Hs  # [H0, H1, H2, H3] in J/kg

        # Evaporator pressure
        p_low = self.Ps[0]
        t_h_curve_points = []

        if p_low < self._p_crit and H[5] > H[3]:
            # Saturated vapor at condenser pressure
            h_sat_vapor = self._compute_state_from_pressure_quality(p_low, 1.0).hmass()

            # Saturated liquid at condenser pressure
            h_sat_liquid = self._compute_state_from_pressure_quality(p_low, 0.0).hmass()

            if H[3] < h_sat_liquid:
                t_h_curve_points.extend(
                    self._isobar_points(p_low, H[3], min(h_sat_liquid, H[5]), 21)
                )

            if H[3] < h_sat_vapor and not (H[5] < h_sat_liquid):
                for point in self._isobar_points(
                    p_low, max(h_sat_liquid, H[3]), min(h_sat_vapor, H[5]), 21
                ):
                    if not _repeats_last_point(t_h_curve_points, point[0]):
                        t_h_curve_points.append(point)

            if H[5] > h_sat_vapor:
                for point in self._isobar_points(
                    p_low, max(h_sat_vapor, H[3]), H[5], 21
                ):
                    if not _repeats_last_point(t_h_curve_points, point[0]):
                        t_h_curve_points.append(point)

        else:
            # Determine supercritical gas heater profile
            for h in np.linspace(H[3], H[5], 61):
                state = self._compute_state_from_pressure_enthalpy(P=p_low, h=h)
                t_h_curve_points.append([h, float(state.T())])

        t_h_curve_points = np.array(t_h_curve_points)

        # Convert temperature to °C if not SI
        t_h_curve_points[:, 1] -= 273.15

        self._evaporator_th_curve = t_h_curve_points

        # Calculate a piecewise-linear approximation of the evaporator profile.
        evaporator_profile = get_piecewise_data_points(
            curve=t_h_curve_points,
            is_hot_stream=False,
            dt_diff_max=self._dt_diff_max,
        )
        self._evaporator_profile = evaporator_profile
        return evaporator_profile

    def _require_solution(self) -> None:
        if not self._solved:
            raise RuntimeError("Solve the cycle before accessing results.")


def _repeats_last_point(points: list, h: float) -> bool:
    """Return whether ``h`` is the enthalpy of the last profile point.

    Adjacent phase blocks share their boundary point; only the repeat is
    skipped, so a block whose neighbour did not run keeps its first point.
    """
    return bool(points) and h == points[-1][0]

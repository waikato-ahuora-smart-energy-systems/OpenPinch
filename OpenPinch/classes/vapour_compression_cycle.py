"""Simple vapour-compression heat pump cycle utilities built on CoolProp."""

# from __future__ import annotations

from typing import Optional, Sequence

import CoolProp
from CoolProp.Plots.Common import PropertyDict, SIunits, process_fluid_state
from CoolProp.Plots.SimpleCycles import StateContainer
import numpy as np

from .stream import Stream
from .stream_collection import StreamCollection
from ..utils.stream_linearisation import get_piecewise_data_points


__all__ = ["VapourCompressionCycle"]


class VapourCompressionCycle:
    """Single vapour-compression heat pump cycle with optional internal heat exchange."""

    STATECOUNT = 6

    def __init__(self):
        """Initialise an unsolved cycle with default operating assumptions."""
        self._system = SIunits()
        self._cycle_states = StateContainer(unit_system=self._system)

        self._dT_superheat: float = 0.0
        self._dT_subcool: float = 0.0
        self._eta_comp: float = 1.0
        self._ihx_gas_dt: float = 0.0
        self._dtcont: float = 0.0
        self._dt_diff_max: float = 0.5  # Default value, used in piecewise approximation of non linear T-h profiles
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
        self._refrigerant: Optional[str] = None
        self._T_evap: Optional[float] = None
        self._T_cond: Optional[float] = None

    @property
    def system(self) -> PropertyDict:
        """CoolProp unit-system definition used by stored state points."""
        return self._system

    @property
    def state(self):
        """Underlying CoolProp fluid state used during cycle calculations."""
        return self._state

    @state.setter
    def state(self, value) -> None:
        self._state = process_fluid_state(value)
        self._solved = False
        self._p_crit = self._state.keyed_output(CoolProp.iP_critical)
        self._t_crit = self._state.keyed_output(CoolProp.iT_critical)
        self._d_crit = self._state.keyed_output(CoolProp.irhomass_critical)

    @property
    def cycle_states(self) -> StateContainer:
        """Container holding the six solved cycle states."""
        return self._cycle_states

    @cycle_states.setter
    def cycle_states(self, value: StateContainer) -> None:
        if len(value) != self.STATECOUNT:
            raise ValueError(f"Expected exactly {self.STATECOUNT} state points.")
        value.units = self._system
        self._cycle_states = value
        self._solved = True

    @property
    def states(self) -> StateContainer:
        """Expose state container for compatibility with plotting API."""
        return self._cycle_states

    @property
    def state_points(self) -> StateContainer:
        """State points around the cycle."""
        return self._cycle_states

    @property
    def Hs(self) -> Sequence[float]:
        """Specific enthalpies for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i, "H"] for i in self._cycle_states]

    @property
    def Ss(self) -> Sequence[float]:
        """Specific entropies for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i, "S"] for i in self._cycle_states]

    @property
    def Ts(self) -> Sequence[float]:
        """Temperatures for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i, "T"] for i in self._cycle_states]

    @property
    def Ps(self) -> Sequence[float]:
        """Pressures for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i, "P"] for i in self._cycle_states]

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
        return self._T_evap

    @property
    def T_cond(self) -> Optional[float]:
        """Condensing temperature in degrees Celsius."""
        return self._T_cond

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
        """Isentropic compressor efficiency."""
        return self._eta_comp

    @property
    def dt_ihx_gas_side(self) -> float:
        """Gas-side temperature change across the internal heat exchanger."""
        return self._ihx_gas_dt

    @property
    def solved(self) -> bool:
        """Flag if the cycle has been solved or not."""
        return self._solved

    def _validate_solve_inputs(
        self,
        refrigerant: str = None,
    ) -> bool:
        if refrigerant is not None:
            self.state = process_fluid_state(refrigerant)
        if self._state is None:
            raise ValueError("A fluid must be specified before solving the cycle.")
        return True

    def _get_P_sat_from_T(
        self,
        T: float,
        Q: float = 1.0,
    ):
        if T > self._t_crit - 1:
            self._state.update(CoolProp.DmassT_INPUTS, self._d_crit, T)
        else:
            self._state.update(CoolProp.QT_INPUTS, Q, T)
        return self._state.p()

    def _compute_state_from_pressure_temperature(
        self,
        P: float,
        T: float,
        *,
        phase: str = 1.0,
    ) -> CoolProp.AbstractState:
        try:
            self._state.update(CoolProp.PT_INPUTS, P, T)
        except:
            self._state.update(
                CoolProp.PQ_INPUTS, P, phase
            )  # Close to saturated liquid/vapour
        return self._state

    def _compute_state_from_pressure_quality(
        self,
        P: float,
        Q: float,
    ) -> CoolProp.AbstractState:
        self._state.update(CoolProp.PQ_INPUTS, P, Q)
        return self._state

    def _compute_compressor_outlet_state(
        self, h_in: float, s_in: float, P_out: float
    ) -> CoolProp.AbstractState:
        self._state.update(CoolProp.PSmass_INPUTS, P_out, s_in)
        h_out_isentropic = self._state.hmass()
        h_out = h_in + (h_out_isentropic - h_in) / self._eta_comp
        self._state.update(CoolProp.HmassP_INPUTS, h_out, P_out)
        return self._state

    def _compute_state_from_pressure_enthalpy(
        self,
        P: float,
        h: float,
    ):
        self._state.update(CoolProp.HmassP_INPUTS, h, P)
        return self._state

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
        i: int,
    ):
        self._cycle_states[i, "H"] = float(self._state.hmass())
        self._cycle_states[i, "S"] = float(self._state.smass())
        self._cycle_states[i, "P"] = float(self._state.p())
        self._cycle_states[i, "T"] = float(self._state.T())

    def solve(
        self,
        T_evap: float,
        T_cond: float,
        *,
        dT_superheat: float = 0.0,
        dT_subcool: float = 0.0,
        eta_comp: float = 0.7,
        refrigerant: str = "water",
        dt_ihx_gas_side: float = 10.0,
        Q_heat: float = None,
        Q_cas_heat: float = 0.0,
        Q_cool: float = None,
        Q_cas_cool: float = 0.0,
        is_heat_pump: bool = True,
    ) -> float:
        """
        Solve the heat pump cycle for the provided operating point.

        Parameters
        ----------
        T_evap : float
            Liquid saturation temperature in the evaporator [deg C].
        T_cond : float
            Gas saturation temperature in the condenser [deg C].
        dT_superheat : float, optional
            Degree of superheating of the suction gas, supplied by the process [K].
        dT_subcool : float, optional
            Degree of subcooling after the condenser, heat delivered to the process [K].
        eta_comp : float, optional
            Isentropic efficiency of the compressor [-].
        refrigerant : str, optional
            Cycle refrigerant; supports multi-component fluids.
        dt_ihx_gas_side : float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : float, optional
            Heat delivered to the process [W].
        Q_cas_heat : float, optional
            Extra condenser heat transferred to the next cascade cycle [W].
            Used only for cascade configurations.
        Q_cool : float, optional
            Cooling delivered to the process [W]; remaining cooling is supplied by
            a lower cascade cycle in cascade configurations.

        Returns
        -------
        float
            Compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self._validate_solve_inputs(
            refrigerant=refrigerant,
        )

        self._refrigerant = refrigerant
        self._T_evap = T_evap
        self._T_cond = T_cond
        self._dT_superheat = dT_superheat
        self._dT_subcool = dT_subcool

        # TODO: Check if this code block could be written more clearly
        self._Q_heat = np.float64(1.0) if (is_heat_pump == True and Q_heat == None) else Q_heat
        self._Q_cas_heat = Q_cas_heat if is_heat_pump else None
        self._Q_cool = np.float64(1.0) if (is_heat_pump == False and Q_cool == None) else Q_cool
        self._Q_cas_cool = Q_cas_cool if not (is_heat_pump) else None

        if is_heat_pump:
            self._Q_cond = self._Q_heat + self._Q_cas_heat
        else:
            self._Q_evap = self._Q_cool + self._Q_cas_cool
        # End block
        
        self._eta_comp = eta_comp
        self._ihx_gas_dt = max(
            min(
                dt_ihx_gas_side,
                T_cond - T_evap - dT_subcool - dT_superheat - self._dtcont * 2,
            ),
            0.0,
        )

        T_evap = self._convert_C_to_K(T_evap)
        T_cond = self._convert_C_to_K(T_cond)
        P_lo = self._get_P_sat_from_T(T_evap)
        P_hi = self._get_P_sat_from_T(T_cond)

        if P_lo > P_hi:
            raise ValueError("Evaporator pressure must be below condenser pressure.")

        """Solve a basic four-state cycle given inlet/outlet temperatures and pressures."""
        # 0 - Evaporator outlet / IHX inlet
        self._compute_state_from_pressure_temperature(
            P=P_lo,
            T=T_evap + dT_superheat,
            phase=1,
        )
        self._save_cycle_state(0)
        hc_ihx_in = self._state.hmass()

        # IHX outlet / compressor inlet
        self._compute_state_from_pressure_temperature(
            P=P_lo,
            T=T_evap + dT_superheat + self._ihx_gas_dt,
            phase=1,
        )
        dh_ihx = self._state.hmass() - hc_ihx_in

        # 1 - Compressor discharge (real)
        self._compute_compressor_outlet_state(
            h_in=self._state.hmass(), s_in=self._state.smass(), P_out=P_hi
        )
        self._save_cycle_state(1)

        # 2 - Condenser outlet / IHX inlet (source)
        self._compute_state_from_pressure_quality(
            P=P_lo,
            Q=0,
        )
        h_cond_out_min = (
            self._state.hmass() + dh_ihx
        )  # Find the limit to subcooling the condensate
        self._compute_state_from_pressure_temperature(
            P=P_hi,
            T=T_cond - dT_subcool,
            phase=0,
        )
        h_cond_out_tar = self._state.hmass()
        self._compute_state_from_pressure_enthalpy(
            P=P_hi, h=max(h_cond_out_min, h_cond_out_tar)
        )
        dh_penalty = max(h_cond_out_min - h_cond_out_tar, 0.0)
        self._dT_subcool = T_cond - self._state.T()
        self._save_cycle_state(2)
        if self._cycle_states[1, "H"] <= self._cycle_states[2, "H"]:
            raise ValueError(
                "Condenser cannot have a negative or zero enthalpy change."
            )

        # 3 - Expansion valve outlet / evaporator inlet
        self._compute_state_from_pressure_enthalpy(
            P=P_lo,
            h=self._state.hmass() - dh_ihx,
        )
        self._save_cycle_state(3)

        self._m_dot = self._Q_cond / (
            self._cycle_states[1, "H"] - self._cycle_states[2, "H"]
        )
        self._q_evap = self._cycle_states[0, "H"] - self._cycle_states[3, "H"]
        self._Q_evap = self._m_dot * self._q_evap
        self._q_cond = self._cycle_states[1, "H"] - self._cycle_states[2, "H"]
        self._w_net = self._q_cond - self._q_evap
        self._work = self._m_dot * self._w_net
        self._penalty = self._m_dot * dh_penalty

        # Extra states for cascade heat exchangers
        # Condenser side - state 4
        # Evaporator side - state 5

        # 4 - Hot side of cascade heat exchanger outlet (if it exists, Q_cas_heat > 0)
        h_4 = self._cycle_states[1, "H"]
        if self._Q_cond > 0:
            h_4 -= (
                (self._cycle_states[1, "H"] - self._cycle_states[2, "H"])
                * self._Q_cas_heat
                / self._Q_cond
            )
        self._compute_state_from_pressure_enthalpy(
            P=self._cycle_states[1, "P"],
            h=h_4,
        )
        self._save_cycle_state(4)
        self._q_cas_heat = self._cycle_states[1, "H"] - self._cycle_states[4, "H"]
        self._q_heat = self._cycle_states[4, "H"] - self._cycle_states[2, "H"]

        # 5 - Hot side of cascade heat exchanger outlet (if it exists, Q_cas_heat > 0)
        if self._Q_cool == None or np.isnan(self._Q_cool):
            self._Q_cool = self._Q_evap
        elif self._Q_cool > self._Q_evap:
            self._Q_cool = self._Q_evap

        if self._m_dot > 0:
            self._q_cool = self._Q_cool / self._m_dot
        else:
            self._q_cool = self._cycle_states[0, "H"] - self._cycle_states[3, "H"]

        self._compute_state_from_pressure_enthalpy(
            P=self._cycle_states[3, "P"],
            h=self._cycle_states[3, "H"] + self._q_cool,
        )
        self._save_cycle_state(5)
        self._q_cas_cool = self._q_evap - self._q_cool
        self._Q_cas_cool = self._Q_evap - self._Q_cool

        # Finish analysis
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
            for i in range(len(profile) - 1):
                h1, T1 = profile[i]
                h2, T2 = profile[i + 1]

                if abs(T1 - T2) < 0.01:
                    name = f"Heater_{i + 1}" if is_condenser else f"Cooler_{i + 1}"
                    T2 = T1 - 0.01 if is_condenser else T1 + 0.01
                else:
                    name = f"Heater_{i + 1}" if T1 > T2 else f"Cooler_{i + 1}"

                s = Stream(
                    name=name,
                    t_supply=T1,
                    t_target=T2,
                    heat_flow=self._m_dot * abs(h1 - h2),
                    is_process_stream=is_process_stream,
                    dt_cont=self._dtcont,
                )
                sc.add(s)
            return sc

        if include_cond:
            streams += _build_streams(
                self._build_condenser_profile(),
                is_condenser=True,
            )
        if include_evap:
            streams += _build_streams(
                self._build_evaporator_profile(),
                is_condenser=False,
            )
        return streams

    def _build_condenser_profile(self) -> np.ndarray:
        """
        Construct a four-point condenser T-h polyline in the VapourCompressionCycle's unit system.

        Returns
        -------
        np.ndarray
            Array of shape (n,2): [enthalpy, temperature].
            Enthalpy always in J/kg, Temperature in K.
        """
        # Ensure the cycle has been solved
        self._require_solution()

        # Read temperatures and enthalpies from the solved cycle
        H = self.Hs  # [H0, H1, H2, H3] in J/kg

        # Use saturation points from the state if needed
        p_high = self.Ps[4]
        t_h_curve_points = []

        if p_high < self._p_crit and H[4] > H[2]:
            # Saturated vapor at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_high, 1.0)
            h_sat_vapor = self._state.hmass()

            # Saturated liquid at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_high, 0.0)
            h_sat_liquid = self._state.hmass()

            if H[4] > h_sat_vapor:
                for h in np.linspace(H[4], max(h_sat_vapor, H[2]), 21):
                    self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                    t_h_curve_points.append([h, float(self._state.T())])

            if H[4] > h_sat_liquid and not (H[2] > h_sat_vapor):
                for h in np.linspace(
                    min(h_sat_vapor, H[4]), max(h_sat_liquid, H[2]), 21
                ):
                    if h != h_sat_vapor or H[4] < h_sat_vapor:
                        self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                        t_h_curve_points.append([h, float(self._state.T())])

            if H[2] < h_sat_liquid:
                for h in np.linspace(min(h_sat_liquid, H[4]), H[2], 21):
                    if h != h_sat_liquid or H[4] < h_sat_liquid:
                        self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                        t_h_curve_points.append([h, float(self._state.T())])

        else:
            # Determine supercritical gas cooler profile
            for h in np.linspace(H[4], H[2], 61):
                self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                t_h_curve_points.append([h, float(self._state.T())])

        t_h_curve_points = np.array(t_h_curve_points)

        # Convert temperature to °C if not SI
        t_h_curve_points[:, 1] -= 273.15

        self._condenser_th_curve = t_h_curve_points

        # Calculate a piece-wise linear approximation of the condenser or gas cooler profile
        condenser_profile = get_piecewise_data_points(
            curve=t_h_curve_points,
            is_hot_stream=True,
            dt_diff_max=self._dt_diff_max,
        )
        self._condenser_profile = condenser_profile
        return condenser_profile

    def _build_evaporator_profile(self) -> np.ndarray:
        """
        Construct a three-point evaporator T-h polyline in the VapourCompressionCycle's unit system.

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
            self._state.update(CoolProp.PQ_INPUTS, p_low, 1.0)
            h_sat_vapor = self._state.hmass()

            # Saturated liquid at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_low, 0.0)
            h_sat_liquid = self._state.hmass()

            if H[3] < h_sat_liquid:
                for h in np.linspace(H[3], min(h_sat_liquid, H[5]), 21):
                    self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                    t_h_curve_points.append([h, float(self._state.T())])

            if H[3] < h_sat_vapor and not (H[5] < h_sat_liquid):
                for h in np.linspace(
                    max(h_sat_liquid, H[3]), min(h_sat_vapor, H[5]), 21
                ):
                    if h != h_sat_liquid or H[3] > h_sat_liquid:
                        self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                        t_h_curve_points.append([h, float(self._state.T())])

            if H[5] > h_sat_vapor:
                for h in np.linspace(max(h_sat_vapor, H[3]), H[5], 21):
                    if h != h_sat_vapor or H[3] > h_sat_vapor:
                        self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                        t_h_curve_points.append([h, float(self._state.T())])

        else:
            # Determine supercritical gas heater profile
            for h in np.linspace(H[3], H[5], 61):
                self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                t_h_curve_points.append([h, float(self._state.T())])

        t_h_curve_points = np.array(t_h_curve_points)

        # Convert temperature to °C if not SI
        t_h_curve_points[:, 1] -= 273.15

        self._evaporator_th_curve = t_h_curve_points

        # Calculate a piece-wise linear approximation of the evaporator or gas heater profile
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

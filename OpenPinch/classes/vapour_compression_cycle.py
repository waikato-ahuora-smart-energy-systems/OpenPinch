"""Simple vapour-compression heat pump cycle utilities built on CoolProp."""

# from __future__ import annotations

from typing import List, Optional, Sequence

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
        self._refrigerant: Optional[str] = "water"
        self._T_evap: Optional[float] = None
        self._T_cond: Optional[float] = None
        self.temperature_unit = "K"

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
        if self.temperature_unit == "C":
            return self._T_evap - 273.15 if self._T_evap is not None else None
        return self._T_evap

    @property
    def T_evap_sat_vap(self) -> Optional[float]:
        """Saturated vapour temperature at evaporating pressure in degrees Celsius."""
        if self.temperature_unit == "C":
            return self._T_evap_sat_vap - 273.15 if self._T_evap_sat_vap is not None else None
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
            return self._T_cond_sat_liq - 273.15 if self._T_cond_sat_liq is not None else None
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
        """Isentropic compressor efficiency."""
        return self._eta_comp

    @property
    def dT_ihx_gas_side(self) -> float:
        """Gas-side temperature change across the internal heat exchanger."""
        return self._ihx_gas_dt

    @property
    def solved(self) -> bool:
        """Flag if the cycle has been solved or not."""
        return self._solved

    @property
    def penalty(self) -> List[float] | np.ndarray:
        """List of penalties for cycle optimisation."""
        return self._penalty

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
        Q: float = None,
    ):
        s = process_fluid_state(self._refrigerant)
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
        s = process_fluid_state(self._refrigerant)
        try:
            s.update(CoolProp.PT_INPUTS, P, T)
        except:
            s.update(
                CoolProp.PQ_INPUTS, P, phase
            )  # Close to saturated liquid/vapour
        return s

    def _compute_state_from_pressure_quality(
        self,
        P: float,
        Q: float,
    ) -> CoolProp.AbstractState:
        s = process_fluid_state(self._refrigerant)
        s.update(CoolProp.PQ_INPUTS, P, Q)
        return s

    def _compute_compressor_outlet_state(
        self, h_in: float, s_in: float, P_out: float
    ) -> CoolProp.AbstractState:
        s = process_fluid_state(self._refrigerant)
        s.update(CoolProp.PSmass_INPUTS, P_out, s_in)
        h_out_isentropic = s.hmass()
        h_out = h_in + (h_out_isentropic - h_in) / self._eta_comp
        s.update(CoolProp.HmassP_INPUTS, h_out, P_out)
        return s

    def _compute_state_from_pressure_enthalpy(
        self,
        P: float,
        h: float,
    ):
        s = process_fluid_state(self._refrigerant)
        s.update(CoolProp.HmassP_INPUTS, h, P)
        return s

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
        self._cycle_states[i, "H"] = float(state.hmass())
        self._cycle_states[i, "S"] = float(state.smass())
        self._cycle_states[i, "P"] = float(state.p())
        self._cycle_states[i, "T"] = float(state.T())

    def solve(
        self,
        T_evap: float,
        T_cond: float,
        *,
        dT_superheat: float = 0.0,
        dT_subcool: float = 0.0,
        eta_comp: float = 0.7,
        refrigerant: str = "water",
        dT_ihx_gas_side: float = 10.0,
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
        dT_ihx_gas_side : float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : float, optional
            Heat delivered to the process [W]. Heat pump and cascade configurations only.
        Q_cas_heat : float, optional
            Extra condenser heat transferred to the next cascade cycle [W].
            Used only for cascade heat pump configurations.
        Q_cool : float, optional
            Cooling delivered to the process [W]. Refrigeration and cascade configurations only.
        Q_cas_cool : float, optional
            Extra evaporator cooling transferred to the next cascade cycle [W].
            Used only for cascade refrigeration configurations.
        is_heat_pump : bool, optional
            Flag to indicate if the cycle is in heat pump or refrigeration mode.

        Returns
        -------
        float
            Compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self.temperature_unit = "K"
        self._validate_solve_inputs(refrigerant)
        self._penalty = []

        self._refrigerant = refrigerant
        self._T_evap = self._convert_C_to_K(T_evap)
        self._T_cond = self._convert_C_to_K(T_cond)
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
        self._ihx_gas_dt = max(
            min(
                dT_ihx_gas_side,
                self._T_cond - self._T_evap - self._dT_subcool - self._dT_superheat - self._dtcont * 2,
            ),
            0.0,
        )

        P_lo = self._get_P_sat_from_T(self._T_evap)
        P_hi = self._get_P_sat_from_T(self._T_cond)

        if P_lo > P_hi:
            raise ValueError("Evaporator pressure must be below condenser pressure.")

        """Solve a basic four-state cycle given inlet/outlet temperatures and pressures."""
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
                P=P_lo, T=self._T_evap_sat_vap + dT_ihx_gas_side, phase=1,
            ).hmass()
            self._penalty.append(h0_tar - h_ihx_in)

        # IHX outlet / compressor inlet
        h_ihx_out = self._compute_state_from_pressure_temperature(
            P=P_lo, T=self._T_evap_sat_vap + self._dT_superheat + self._ihx_gas_dt, phase=1,
        ).hmass()
        dh_ihx = h_ihx_out - h_ihx_in

        # 1 - Compressor discharge (real)
        state1 = self._compute_compressor_outlet_state(
            h_in=h_ihx_out, s_in=state0.smass(), P_out=P_hi
        )
        self._save_cycle_state(state1, 1)

        """self._penalty.append(
            max(self._compute_state_from_pressure_quality(P_hi, 0.0).hmass() - state1.hmass(), 0.0) # Penalise wet compression
        )  """      

        # 2 - Condenser outlet / IHX inlet (source)
        h_cond_out_min = self._compute_state_from_pressure_quality(
            P=P_lo, Q=0,
        ).hmass() + dh_ihx  # Find the limit to subcooling of the suction gas
        self._T_cond_sat_liq = self._compute_state_from_pressure_quality(
            P=P_hi, Q=0,
        ).T()  # Saturated liquid temperature at the condenser pressure        
        h_cond_out_tar = self._compute_state_from_pressure_temperature(
            P=P_hi, T=min(self._T_cond_sat_liq, self._t_crit - 0.1) - self._dT_subcool, phase=0,
        ).hmass()
        h_cond_out = max(h_cond_out_min, h_cond_out_tar)
        state2 = self._compute_state_from_pressure_enthalpy(
            P=P_hi, h=h_cond_out,
        )
        self._dT_subcool = self._T_cond_sat_liq - state2.T() # Actual subcooling possible
        self._save_cycle_state(state2, 2)

        if h_cond_out_min > h_cond_out_tar:
            # Penalise excessive subcooling beyond the maximum allowed
            self._penalty.append(h_cond_out_min - h_cond_out_tar)

        if self._cycle_states[1, "H"] <= self._cycle_states[2, "H"]:
            raise ValueError(
                "Condenser cannot have a negative or zero enthalpy change."
            )

        # 3 - Expansion valve outlet / evaporator inlet
        state3 = self._compute_state_from_pressure_enthalpy(
            P=P_lo, h=state2.hmass() - dh_ihx,
        )
        self._save_cycle_state(state3, 3)

        self._q_evap = state0.hmass() - state3.hmass()
        self._q_cond = state1.hmass() - state2.hmass()
        self._w_net = self._q_cond - self._q_evap
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
            self._q_cas_heat = self._Q_cas_heat / self._m_dot if self._m_dot > 0.0 else 0.0
            self._q_heat = self._q_cond - self._q_cas_heat
            state4 = self._compute_state_from_pressure_enthalpy(
                P=state2.p(), h=state2.hmass() + self._q_heat,
            )
            self._save_cycle_state(state4, 4)

            # 5 - Hot side of cascade heat exchanger outlet (otherwise state 5 == 3)
            if self._Q_cool == None or np.isnan(self._Q_cool):
                self._Q_cool = self._Q_evap
            elif self._Q_cool > self._Q_evap:
                self._Q_cool = self._Q_evap

            self._q_cool = self._Q_cool / self._m_dot if self._m_dot > 0 else self._q_evap

            state5 = self._compute_state_from_pressure_enthalpy(
                P=state3.p(), h=state3.hmass() + self._q_cool,
            )
            self._save_cycle_state(state5, 5)
            self._q_cas_cool = self._q_evap - self._q_cool
            self._Q_cas_cool = self._Q_evap - self._Q_cool

        else:  # is refrigeration
            # 5 - Hot side of cascade heat exchanger outlet (otherwise state 5 == 3)
            self._q_cas_cool = self._Q_cas_cool / self._m_dot if self._m_dot > 0.0 else 0.0
            self._q_cool = self._q_evap - self._q_cas_cool
            state5 = self._compute_state_from_pressure_enthalpy(
                P=state3.p(), h=state3.hmass() + self._q_cool,
            )
            self._save_cycle_state(state5, 5)

            # 4 - Hot side of cascade heat exchanger outlet (otherwise state 4 == 1)
            if self._Q_heat == None or np.isnan(self._Q_heat):
                self._Q_heat = self._Q_cond
            elif self._Q_heat > self._Q_cond:
                self._Q_heat = self._Q_cond
            self._q_heat = self._Q_heat / self._m_dot if self._m_dot > 0 else self._q_cond
            state4 = self._compute_state_from_pressure_enthalpy(
                P=state2.p(), h=state2.hmass() + self._q_heat,
            )
            self._save_cycle_state(state4, 4)

        # Finish analysis
        self._penalty = np.asarray(self._penalty) * self._m_dot
        self.temperature_unit = "C"
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

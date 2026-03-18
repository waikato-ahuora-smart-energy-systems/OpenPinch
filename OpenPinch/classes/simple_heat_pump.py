"""Simple Heat Pump cycle utilities that wrap CoolProp."""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Tuple

import CoolProp
from CoolProp.Plots.Common import EURunits, KSIunits, PropertyDict, SIunits, process_fluid_state
from CoolProp.Plots.SimpleCycles import StateContainer
import numpy as np

from .stream import Stream
from .stream_collection import StreamCollection
from ..utils.stream_linearisation import get_piecewise_data_points


__all__ = ['SimpleHeatPumpCycle']

class SimpleHeatPumpCycle:
    """Compute a simple vapor compression heat pump cycle.
    """

    STATECOUNT = 4

    def __init__(self):
        self._system = SIunits()
        self._cycle_states = StateContainer(unit_system=self._system)
        self._state = None
        self._w_net: Optional[float] = None
        self._q_evap: Optional[float] = None
        self._q_cond: Optional[float] = None
        self._Q_cond: Optional[float] = None
        self._Q_evap: Optional[float] = None
        self._m_dot: Optional[float] = None
        self._solved: bool = False
        self._dtcont: float = 0.0
        self._dt_diff_max: float = 0.5 # Default value, used in piecewise approximation of non linear T-h profiles
        self._refrigerant: Optional[str] = None
        self._T_evap: Optional[float] = None
        self._T_cond: Optional[float] = None
        self._dT_superheat: float = 0.0
        self._dT_subcool: float = 0.0
        self._eta_comp: float = 1.0
        self._ihx_gas_dt: float = 0.0

    @property
    def system(self) -> PropertyDict:
        return self._system

    @property
    def state(self):
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
        return self._cycle_states

    @cycle_states.setter
    def cycle_states(self, value: StateContainer) -> None:
        if len(value) != self.STATECOUNT:
            raise ValueError(f'Expected exactly {self.STATECOUNT} state points.')
        value.units = self._system
        self._cycle_states = value
        self._solved = True

    @property
    def states(self) -> StateContainer:
        """Expose state container for compatibility with plotting API."""
        return self._cycle_states

    @property
    def state_points(self) -> StateContainer:
        return self._cycle_states

    @property
    def Hs(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, 'H'] for i in self._cycle_states]

    @property
    def Ss(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, 'S'] for i in self._cycle_states]

    @property
    def Ts(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, CoolProp.iT] for i in self._cycle_states]

    @property
    def Ps(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, CoolProp.iP] for i in self._cycle_states]

    @property
    def q_evap(self) -> Optional[float]:
        return self._q_evap

    @property
    def Q_evap(self) -> Optional[float]:
        return self._Q_evap

    @property
    def q_cond(self) -> Optional[float]:
        return self._q_cond

    @property
    def Q_cond(self) -> Optional[float]:
        return self._Q_cond
                
    @property
    def w_net(self) -> Optional[float]:
        return self._w_net

    @property
    def work(self) -> Optional[float]:
        return self._work

    @property
    def dtcont(self) -> Optional[float]:
        return self._dtcont
    
    @dtcont.setter
    def dtcont(self, value: float):
        self._dtcont = value  

    @property
    def COP_h(self) -> Optional[float]:
        self._require_solution()
        return self._q_cond / self._w_net

    @property
    def COP_r(self) -> Optional[float]:
        self._require_solution()
        return self._q_evap / self._w_net

    @property
    def dt_diff_max(self) -> PropertyDict:
        return self._dt_diff_max

    @dt_diff_max.setter
    def dt_diff_max(self, value: float) -> None:
        self._dt_diff_max = value

    @property
    def refrigerant(self) -> Optional[str]:
        return self._refrigerant

    @property
    def T_evap(self) -> Optional[float]:
        return self._T_evap

    @property
    def T_cond(self) -> Optional[float]:
        return self._T_cond

    @property
    def dT_superheat(self) -> float:
        return self._dT_superheat

    @property
    def dT_subcool(self) -> float:
        return self._dT_subcool

    @property
    def eta_comp(self) -> float:
        return self._eta_comp

    @property
    def ihx_gas_dt(self) -> float:
        return self._ihx_gas_dt


    def _validate_solve_inputs(
        self,
        refrigerant: str = None,             
    ) -> bool:
        if refrigerant is not None:
            self.state = process_fluid_state(refrigerant)
        if self._state is None:
            raise ValueError('A fluid must be specified before solving the cycle.') 
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
        p: float,
        T: float,
        *,
        phase: str = 1.0,
    ) -> CoolProp.AbstractState:
        try:
            self._state.update(CoolProp.PT_INPUTS, p, T)          
        except:
            self._state.update(CoolProp.PQ_INPUTS, p, phase) # Close to saturated liquid/vapour  
        return self._state
    

    def _compute_evaporator_outlet_state(
        self,
        p: float, 
        T: float,        
    ) -> CoolProp.AbstractState:
        return self._compute_state_from_pressure_temperature(p, T, 1.0)
    

    def _compute_compressor_outlet_state(
        self, 
        h_in: float, 
        s_in: float, 
        p_out: float
    ) -> CoolProp.AbstractState:
        self._state.update(CoolProp.PSmass_INPUTS, p_out, s_in)
        h_out_isentropic = self._state.hmass()
        h_out = h_in + (h_out_isentropic - h_in) / self._eta_comp
        self._state.update(CoolProp.HmassP_INPUTS, h_out, p_out)
        return self._state
    

    def _compute_condenser_outlet_state(
        self, 
        p: float, 
        T: float, 
        dT_sc: float,
    ) -> CoolProp.AbstractState:
        try:
            self._state.update(CoolProp.PT_INPUTS, p, T)
        except:
            self._state.update(CoolProp.PQ_INPUTS, p, 0.0) # Close to saturated liquid
        
        if self._state.hmass() > self._cycle_states[0, 'H']:
            self._state.update(CoolProp.HmassP_INPUTS, self._cycle_states[0, 'H'], p)
            T = self._state.T() - dT_sc
            self._state.update(CoolProp.PT_INPUTS, p, T)
        return self._state
    

    def _compute_state_from_pressure_enthalpy(
        self, 
        p: float, 
        h: float,
    ):
        self._state.update(CoolProp.HmassP_INPUTS, h, p)
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
        self._cycle_states[i, 'H'] = float(self._state.hmass())
        self._cycle_states[i, 'S'] = float(self._state.smass())
        self._cycle_states[i, CoolProp.iP] = float(self._state.p())
        self._cycle_states[i, CoolProp.iT] = float(self._state.T())


    def _get_metrics(self):
        self._q_evap = (self._cycle_states[0, 'H'] - self._cycle_states[3, 'H']) / 1000
        self._q_cond = (self._cycle_states[1, 'H'] - self._cycle_states[2, 'H']) / 1000
        self._w_net = self._q_cond - self._q_evap

        self._m_dot = self._Q_cond / self._q_cond
        self._Q_evap = self._m_dot * self._q_evap
        self._work = self._m_dot * self._w_net


    def solve(
        self,
        Te: float,
        Tc: float,
        *,
        dT_sh: float = 0.0,
        dT_sc: float = 0.0,
        eta_comp: float = 0.7,
        refrigerant: str = "water",
        ihx_gas_dt: float = 40.0,
        Q_h_total: float = 1.0,
    ) -> float:
        """Solve the cycle using saturation temperatures, superheat, and subcooling."""
        self._validate_solve_inputs(
            refrigerant=refrigerant,
        )

        self._refrigerant = refrigerant
        self._T_evap = Te
        self._T_cond = Tc
        self._dT_superheat = dT_sh
        self._dT_subcool = dT_sc   
        self._Q_cond = Q_h_total
        self._eta_comp = eta_comp
        self._ihx_gas_dt = min(ihx_gas_dt, Tc - Te - dT_sc - dT_sh - 5)

        Te = self._convert_C_to_K(Te)
        Tc = self._convert_C_to_K(Tc)
        p0 = self._get_P_sat_from_T(Te)
        p2 = self._get_P_sat_from_T(Tc)

        if p0 > p2:
            raise ValueError('Evaporator pressure must be below condenser pressure.')   

        """Solve a basic four-state cycle given inlet/outlet temperatures and pressures."""
        # 0 - Evaporator outlet / IHX inlet
        self._compute_state_from_pressure_temperature(
            p=p0, 
            T=Te + dT_sh,
            phase=1,
        )
        self._save_cycle_state(0)
        hc_ihx_in = self._state.hmass()

        # IHX outlet / compressor inlet
        self._compute_state_from_pressure_temperature(
            p=p0, 
            T=Te + dT_sh + self._ihx_gas_dt,
            phase=1,
        )      
        dh_ihx = self._state.hmass() - hc_ihx_in

        # 1 - Compressor discharge (real)
        self._compute_compressor_outlet_state(
            h_in=self._state.hmass(),
            s_in=self._state.smass(),
            p_out=p2
        )
        self._save_cycle_state(1)

        # Condenser outlet / IHX inlet (source)
        self._compute_state_from_pressure_temperature(
            p=p2,
            T=Tc - dT_sc,
            phase=0,
        )
        self._save_cycle_state(2)

        # Expansion valve outlet / evaporator inlet
        self._compute_state_from_pressure_enthalpy(
            p=p0,
            h=self._state.hmass() - dh_ihx,
        )
        self._save_cycle_state(3)

        self._get_metrics()

        self._solved = True
        return self._work


    def build_stream_collection(self, include_cond: bool = False, include_evap: bool = False, is_process_stream: bool = False) -> Tuple[StreamCollection, StreamCollection]:

        def _build_streams(profile: np.ndarray, is_hot: bool): 

            if is_hot:
                self._m_dot = self._Q_cond / abs(profile[0,0] - profile[-1,0])
            sc = StreamCollection()
            for i in range(len(profile) - 1):
                h1, T1 = profile[i]
                h2, T2 = profile[i+1]

                name = f"Condenser_{i+1}" if is_hot else f"Evaporator_{i+1}"

                if abs(T1 - T2) < 0.001: # Catch phase change
                    if is_hot:
                        t_target = T2 - 0.01
                    else:
                        t_target = T2 + 0.01
                else:
                    t_target = T2

                s = Stream(
                    name=name,
                    t_supply=T1,
                    t_target=t_target,
                    heat_flow=self._m_dot*abs(h1 - h2),  # or m_dot * (h1 - h2), depending on your model
                    is_process_stream=False,
                    dt_cont=self._dtcont,
                )

                sc.add(s)
            return sc
        
        streams = StreamCollection()
        if include_cond:
            temp = self._build_condenser_profile()
            streams += _build_streams(temp, True)
        if include_evap:
            temp = self._build_evaporator_profile()
            streams += _build_streams(temp, False)            
        return streams


    def _build_condenser_profile(self) -> np.ndarray:
        """
        Construct a four-point condenser T-h polyline in the SimpleHeatPumpCycle's unit system.
    
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
        p_high = self.Ps[1]
        t_h_curve_points = []

        if p_high < self._p_crit:    
            # Saturated vapor at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_high, 1.0)
            h_sat_vapor = self._state.hmass()
        
            # Saturated liquid at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_high, 0.0)
            h_sat_liquid = self._state.hmass()

            for h in np.linspace(H[1], h_sat_vapor, 21):
                self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                t_h_curve_points.append([h, float(self._state.T())])
            
            for h in np.linspace(h_sat_vapor, h_sat_liquid, 21):
                if h != h_sat_vapor:
                    self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                    t_h_curve_points.append([h, float(self._state.T())])            
            
            for h in np.linspace(h_sat_liquid, H[2], 21):
                if h != h_sat_liquid:
                    self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                    t_h_curve_points.append([h, float(self._state.T())])

        else:
            # Determine supercritical gas cooler profile
            for h in np.linspace(H[1], H[2], 61):
                self._state.update(CoolProp.HmassP_INPUTS, h, p_high)
                t_h_curve_points.append([h, float(self._state.T())])
        
        t_h_curve_points = np.array(t_h_curve_points)

        # Convert temperature to °C if not SI
        t_h_curve_points[:, 1] -= 273.15

        # Calculate a piece-wise linear approximation of the condenser or gas cooler profile
        condenser_profile = get_piecewise_data_points(
            curve=t_h_curve_points, 
            is_hot_stream=True, 
            dt_diff_max=self._dt_diff_max,
        )
        return condenser_profile


    def _build_evaporator_profile(self) -> np.ndarray:
        """
        Construct a three-point evaporator T-h polyline in the SimpleHeatPumpCycle's unit system.
    
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

        if p_low < self._p_crit:    
            # Saturated vapor at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_low, 1.0)
            h_sat_vapor = self._state.hmass()
        
            # Saturated liquid at condenser pressure
            self._state.update(CoolProp.PQ_INPUTS, p_low, 0.0)
            h_sat_liquid = self._state.hmass()

            if H[3] < h_sat_liquid:
                for h in np.linspace(H[3], h_sat_liquid, 21):
                    self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                    t_h_curve_points.append([h, float(self._state.T())])
            
            if H[3] < h_sat_vapor:
                for h in np.linspace(max(h_sat_liquid, H[3]), h_sat_vapor, 21):
                    if h != h_sat_liquid or H[3] > h_sat_liquid:
                        self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                        t_h_curve_points.append([h, float(self._state.T())])            
            if H[0] > h_sat_vapor:
                for h in np.linspace(max(h_sat_vapor, H[3]), H[0], 21):
                    if h != h_sat_vapor or H[3] > h_sat_vapor:
                        self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                        t_h_curve_points.append([h, float(self._state.T())])

        else:
            # Determine supercritical gas heater profile
            for h in np.linspace(H[3], H[0], 61):
                self._state.update(CoolProp.HmassP_INPUTS, h, p_low)
                t_h_curve_points.append([h, float(self._state.T())])
        
        t_h_curve_points = np.array(t_h_curve_points)

        # Convert temperature to °C if not SI
        t_h_curve_points[:, 1] -= 273.15

        # Calculate a piece-wise linear approximation of the evaporator or gas heater profile
        evaporator_profile = get_piecewise_data_points(
            curve=t_h_curve_points, 
            is_hot_stream=False, 
            dt_diff_max=self._dt_diff_max,
        )
        return evaporator_profile


    def _require_solution(self) -> None:
        if not self._solved:
            raise RuntimeError('Solve the cycle before accessing results.')

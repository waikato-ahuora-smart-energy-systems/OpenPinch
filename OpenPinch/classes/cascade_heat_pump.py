"""Simple Heat Pump cycle utilities that wrap CoolProp."""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np

from .stream_collection import StreamCollection
from .simple_heat_pump import SimpleHeatPumpCycle


__all__ = ['CascadeHeatPumpCycle']

class CascadeHeatPumpCycle:
    """
    Compute a cascade of simple vapor compression heat pump cycles
    with or without an internal heat exchanger.
    """

    def __init__(self):
        self._subcycles = []

        self._T_evap = []
        self._T_cond = []
        self._dT_superheat = []
        self._dT_subcool = []
        self._eta_comp = []
        self._refrigerant = []
        self._ihx_gas_dt = []
        self._Q_heat = []
        self._Q_cas_heat = []
        self._Q_cool = []

        self._dtcont: float = 0.0
        self._dt_diff_max: float = 0.5 # Default value, used in piecewise approximation of non linear T-h profiles      
        self._solved: bool = False


        ### check
        # self._w_net: Optional[float] = None
        # self._q_cond: Optional[float] = None
        # self._Q_cond: Optional[float] = None
        # self._q_cas_heat: Optional[float] = None
        # self._Q_cas_heat: Optional[float] = None
        # self._q_heat: Optional[float] = None
        
        # self._q_evap: Optional[float] = None
        # self._Q_evap: Optional[float] = None
        # self._q_cas_cool: Optional[float] = None
        # self._Q_cas_cool: Optional[float] = None
        # self._q_cool: Optional[float] = None
        # self._Q_cool: Optional[float] = None

        # self._m_dot: Optional[float] = None
        # self._work: Optional[float] = None


    @property
    def Q_evap(self) -> Optional[float]:
        return sum([cycle.Q_evap for cycle in self._subcycles])

    @property
    def Q_cas_cool(self) -> Optional[float]:
        return sum([cycle.Q_cas_cool for cycle in self._subcycles])
     
    @property
    def Q_cool(self) -> Optional[float]:
        return sum([cycle.Q_cool for cycle in self._subcycles])
    
    @property
    def Q_cond(self) -> Optional[float]:
        return sum([cycle.Q_cond for cycle in self._subcycles])

    @property
    def Q_cas_heat(self) -> Optional[float]:
        return sum([cycle.Q_cas_heat for cycle in self._subcycles])
 
    @property
    def Q_heat(self) -> Optional[float]:
        return sum([cycle.Q_heat for cycle in self._subcycles])

    @property
    def w_net(self) -> Optional[float]:
        return sum([cycle.w_net for cycle in self._subcycles])

    @property
    def dtcont(self) -> Optional[float]:
        return self._dtcont

    @property
    def COP_h(self) -> Optional[float]:
        self._require_solution()
        return self.Q_heat / self.w_net

    @property
    def COP_r(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_cool for cycle in self._subcycles]) / sum([cycle.w_net for cycle in self._subcycles])

    @property
    def dt_diff_max(self) -> Optional[float]:
        return self._dt_diff_max

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

    @property
    def dt_cascade_hx(self) -> float:
        return self._dt_cascade_hx
    

    def solve(
        self,
        T_evap: List[float] | float,
        T_cond: List[float], 
        *,
        dT_superheat: List[float] | float = 0.0,
        dT_subcool: List[float] | float = 0.0,
        eta_comp: List[float] | float = 0.7,
        refrigerant: List[str] | str = "water",
        ihx_gas_dt: List[float] | float = 10.0,
        Q_heat: List[float] | float = 1.0,
        Q_cas_heat: List[float] | float = 0.0,
        Q_cool: List[float] | float = None,
        dt_cascade_hx: float = 1.0,
    ) -> float:
        """
        Solve the heat-pump cycle for the provided operating point.

        Parameters
        ----------
        T_evap : List[float] 
            Liquid saturation temperature in the evaporator [deg C].
        T_cond : List[float]
            Gas saturation temperature in the condenser [deg C].
        dT_superheat : List[float], optional
            Degree of superheating of the suction gas, supplied by the process [K].
        dT_subcool : List[float], optional
            Degree of subcooling after the condenser, heat delivered to the process [K].
        eta_comp : List[float], optional
            Isentropic efficiency of the compressor [-].
        refrigerant : List[str], optional
            Cycle refrigerant; supports multi-component fluids.
        ihx_gas_dt : List[float], optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : List[float], optional
            Heat delivered to the process [W].
        Q_cas_heat : List[float], optional
            Extra condenser heat transferred to the next cascade cycle [W].
            Used only for cascade configurations.
        Q_cool : List[float], optional
            Cooling delivered to the process [W]; remaining cooling is supplied by
            a lower cascade cycle in cascade configurations.

        Returns
        -------
        float
            Compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self._dt_cascade_hx = dt_cascade_hx

        if isinstance(T_cond, float):
            T_cond = [T_cond]
        if isinstance(T_evap, float):
            T_evap = [T_evap]
        if min(T_cond) < max(T_evap):
            raise ValueError("Invalid condenser and evaporator temperatures.")

        T_cond = np.array(T_cond).sort()[::-1]
        T_evap = np.array(T_evap).sort()[::-1]
        n_heat = T_cond.size
        n_cool = T_evap.size

        T_cond_all = np.concatenate([
            T_cond,
            T_evap[:-1] + self._dt_cascade_hx,
        ]).sort()

        T_evap_all = np.concatenate([
            T_cond[1:] - self._dt_cascade_hx,
            T_evap,
        ]).sort()        

        if T_evap_all.size != T_cond_all.size:
            raise ValueError("Mismatched number of heat pumps in the cascade.")

        self._num_cycles = T_evap_all.size



        

        for i in range(self._num_cycles):
            hp = SimpleHeatPumpCycle()
            hp.solve(
                T_evap=T_evap_all[i],
                T_cond=T_cond_all[i],
                # TODO: prior to the for loop, construct lists of the correct length to match the number of simples heat pumps. Analyse the sitution to figure out how this should be done. 
                dT_subcool=dT_subcool[i] if isinstance(dT_subcool, list) else dT_subcool,
                eta_comp=eta_comp[i] if isinstance(eta_comp, list) else eta_comp,
                refrigerant=refrigerant[i] if isinstance(refrigerant, list) else refrigerant,
                ihx_gas_dt=ihx_gas_dt[i] if isinstance(ihx_gas_dt, list) else ihx_gas_dt,
                Q_heat=Q_heat[i] if isinstance(Q_heat, list) else Q_heat,
                Q_cas_heat=Q_cas_heat[i] if isinstance(Q_cas_heat, list) else Q_cas_heat,
                Q_cool=Q_cool[i] if isinstance(Q_cool, list) else Q_cool,     
            )
            self._subcycles.append(hp)

        # Finish analysis
        self._solved = True
        return self.w_net


    def build_stream_collection(
        self, 
        include_cond: bool = False, 
        include_evap: bool = False, 
        is_process_stream: bool = False,
        dtcont: float = 0.0,
        dt_diff_max: float = 0.5,
    ) -> StreamCollection:
        
        self._require_solution()
        self._dtcont = dtcont
        self._dt_diff_max = dt_diff_max
        streams = StreamCollection()

        for cycle in self._subcycles:
            streams += cycle.build_stream_collection(
                include_cond=include_cond,
                include_evap=include_evap,
                is_process_stream=is_process_stream,
            )          
        return streams


    def _require_solution(self) -> None:
        if not self._solved:
            raise RuntimeError('Solve the cycle before accessing results.')

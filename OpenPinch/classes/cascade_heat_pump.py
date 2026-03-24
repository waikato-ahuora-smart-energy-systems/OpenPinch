"""Simple Heat Pump cycle utilities that wrap CoolProp."""

from __future__ import annotations

from typing import Optional, List
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
        self._T_evap = None
        self._T_cond = None
        self._dT_superheat = None
        self._dT_subcool = None
        self._eta_comp = None
        self._refrigerant = None
        self._ihx_gas_dt = None
        self._Q_heat = None
        self._Q_cas_heat = None
        self._Q_cool = None
        self._num_cycles = 1
        self._dtcont: float = 0.0
        self._dt_diff_max: float = 0.5 # Default value, used in piecewise approximation of non linear T-h profiles      
        self._solved: bool = False

    @property
    def Q_evap(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_evap for cycle in self._subcycles])
    
    @property
    def Q_evap_arr(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.Q_evap for cycle in self._subcycles])

    @property
    def Q_cas_cool(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_cas_cool for cycle in self._subcycles])
     
    @property
    def Q_cas_cool_arr(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.Q_cas_cool for cycle in self._subcycles])
        
    @property
    def Q_cool(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_cool for cycle in self._subcycles])

    @property
    def Q_cool_arr(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.Q_cool for cycle in self._subcycles])
    
    @property
    def Q_cond(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_cond for cycle in self._subcycles])
    
    @property
    def Q_cond(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.Q_cond for cycle in self._subcycles])

    @property
    def Q_cas_heat(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_cas_heat for cycle in self._subcycles])
 
    @property
    def Q_cas_heat_arr(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.Q_cas_heat for cycle in self._subcycles])
    
    @property
    def Q_heat(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.Q_heat for cycle in self._subcycles])

    @property
    def Q_heat_arr(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.Q_heat for cycle in self._subcycles])

    @property
    def work(self) -> Optional[float]:
        self._require_solution()
        return sum([cycle.work for cycle in self._subcycles])

    @property
    def work_arr(self) -> Optional[float]:
        self._require_solution()
        return np.array([cycle.work for cycle in self._subcycles])

    @property
    def dtcont(self) -> Optional[float]:
        return self._dtcont

    @property
    def COP_h(self) -> Optional[float]:
        self._require_solution()
        return self.Q_heat / self.work

    @property
    def COP_r(self) -> Optional[float]:
        self._require_solution()
        return self.Q_cool / self.work
    
    @property
    def COP_o(self) -> Optional[float]:
        self._require_solution()
        return (self.Q_heat + self.Q_cool) / self.work

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
    
    @property
    def num_cycles(self) -> int:
        return self._num_cycles
    
    @property
    def subcycles(self) -> int:
        return self._subcycles    

    def _validate_cascade_hp_input_array(
        self,
        arr: np.ndarray, 
        n_heat: int, 
        n_cool: int, 
        is_cond_attri: bool = True,
        is_fixable: bool = True,
    ):
        arr = np.asarray(0.0 if arr is None else arr, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("Incompatible input to solving a cascade heat pump.")

        n_cycles = n_heat + n_cool - 1
        if arr.size == n_cycles:
            arr_new = arr
        elif arr.size == 1 and is_fixable:
            arr_new = np.full(n_cycles, arr.item(), dtype=float)
        elif arr.size == n_cool and is_cond_attri is False:
            arr_new = np.concatenate([
                np.zeros(n_heat-1),
                arr,
            ])
        elif arr.size == n_heat and is_cond_attri is True:
            arr_new = np.concatenate([
                arr,
                np.zeros(n_cool-1),
            ])
        else:
            raise ValueError("Incompatible input to solving a cascade heat pump.")
        return arr_new    


    def solve(
        self,
        T_evap: np.ndarray,
        T_cond: np.ndarray, 
        *,
        dT_superheat: np.ndarray = 0.0,
        dT_subcool: np.ndarray = 0.0,
        eta_comp: float = 0.7,
        refrigerant: List[str] | str = "water",
        ihx_gas_dt: np.ndarray | float = 10.0,
        Q_heat: np.ndarray = 1.0,
        Q_cool: np.ndarray = None,
        dt_cascade_hx: float = 1.0,
    ) -> float:
        """
        Solve the heat-pump cycle for the provided operating point.

        Parameters
        ----------
        T_evap : np.ndarray
            Liquid saturation temperature in the evaporator [deg C].
        T_cond : np.ndarray
            Gas saturation temperature in the condenser [deg C].
        dT_superheat : np.ndarray, optional
            Degree of superheating of the suction gas, supplied by the process [K].
        dT_subcool : np.ndarray, optional
            Degree of subcooling after the condenser, heat delivered to the process [K].
        eta_comp : float, optional
            Isentropic efficiency of the compressor [-].
        refrigerant : List[str], optional
            Cycle refrigerant; supports multi-component fluids.
        ihx_gas_dt : np.ndarray | float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : np.ndarray, optional
            Heat delivered to the process [W].
        Q_cool : np.ndarray, optional
            Cooling delivered to the process [W]; remaining cooling is supplied by
            a lower cascade cycle in cascade configurations.

        Returns
        -------
        float
            Compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self._subcycles = []
        self._dt_cascade_hx = dt_cascade_hx

        T_cond = np.asarray(T_cond, dtype=float)
        T_evap = np.asarray(T_evap, dtype=float)

        if T_cond.min() < T_evap.max() + self._dt_cascade_hx:
            raise ValueError("Invalid condenser and evaporator temperatures.")

        T_cond = np.sort(T_cond)[::-1]
        T_evap = np.sort(T_evap)[::-1]

        T_cond_all = np.sort(np.concatenate([
            T_cond,
            T_evap[:-1] + self._dt_cascade_hx,
        ]))[::-1]

        T_evap_all = np.sort(np.concatenate([
            T_cond[1:] - self._dt_cascade_hx,
            T_evap,
        ]))[::-1]

        self._num_cycles = T_evap_all.size
        n_heat = T_cond.size
        n_cool = T_evap.size

        dT_superheat_all = self._validate_cascade_hp_input_array(dT_superheat, n_heat, n_cool, False, True)
        dT_subcool_all = self._validate_cascade_hp_input_array(dT_subcool, n_heat, n_cool, True, True)
        Q_heat_all = self._validate_cascade_hp_input_array(Q_heat, n_heat, n_cool, True, False)
        Q_cool_all = self._validate_cascade_hp_input_array(Q_cool, n_heat, n_cool, False, False)

        if isinstance(refrigerant, list):
            if len(refrigerant) == self._num_cycles:
                refrigerant_all = refrigerant
            elif len(refrigerant) == 1:
                refrigerant_all = refrigerant * self._num_cycles
            else:
                raise ValueError(f"Number of refrigerants must match the number of heat pumps, {self._num_cycles}.")
        else:
            refrigerant_all = [refrigerant] * self._num_cycles

        if np.isscalar(ihx_gas_dt):
            ihx_gas_dt_all = np.full(self._num_cycles, ihx_gas_dt, dtype=float)
        else:
            ihx_gas_dt_all = np.asarray(ihx_gas_dt, dtype=float)
            if ihx_gas_dt_all.size != self._num_cycles:
                raise ValueError("ihx_gas_dt must match the number of heat pumps.")

        Q_cas_heat = 0.0
        for i in range(self._num_cycles):
            hp = SimpleHeatPumpCycle()
            hp.solve(
                T_evap=T_evap_all[i],
                T_cond=T_cond_all[i],
                dT_superheat=dT_superheat_all[i],
                dT_subcool=dT_subcool_all[i],
                eta_comp=eta_comp,
                refrigerant=refrigerant_all[i],
                ihx_gas_dt=ihx_gas_dt_all[i],
                Q_heat=Q_heat_all[i],
                Q_cas_heat=Q_cas_heat,
                Q_cool=Q_cool_all[i],
            )
            self._subcycles.append(hp)
            Q_cas_heat = hp.Q_cas_cool

        # Finish analysis
        self._solved = True
        return self.work


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

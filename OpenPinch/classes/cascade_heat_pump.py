"""Cascade heat pump network assembled from multiple simple subcycles."""

from __future__ import annotations

from typing import Optional, List
import numpy as np

from .stream_collection import StreamCollection
from .simple_heat_pump import SimpleHeatPumpCycle


__all__ = ['CascadeHeatPumpCycle']

class CascadeHeatPumpCycle:
    """Cascade of vapour-compression heat pumps coupled through cascade exchangers."""

    def __init__(self):
        """Initialise an unsolved cascade with no configured subcycles."""
        self._subcycles = []
        self._num_cycles = 1
        self._dtcont: float = 0.0
        self._dt_diff_max: float = 0.5 # Default value, used in piecewise approximation of non linear T-h profiles      
        self._solved: bool = False
        self._max_work: float = 0.0

    @property
    def Q_evap(self) -> Optional[float]:
        """Total evaporator duty across all subcycles."""
        self._require_solution()
        return sum(cycle.Q_evap for cycle in self._subcycles)
    
    @property
    def Q_evap_arr(self) -> Optional[float]:
        """Per-subcycle evaporator duties."""
        self._require_solution()
        return np.array([cycle.Q_evap for cycle in self._subcycles])

    @property
    def Q_cas_cool(self) -> Optional[float]:
        """Total cooling handed off to lower cascade stages."""
        self._require_solution()
        return sum(cycle.Q_cas_cool for cycle in self._subcycles)
     
    @property
    def Q_cas_cool_arr(self) -> Optional[float]:
        """Per-subcycle cooling handed to lower cascade stages."""
        self._require_solution()
        return np.array([cycle.Q_cas_cool for cycle in self._subcycles])
        
    @property
    def Q_cool(self) -> Optional[float]:
        """Total cooling delivered to the process."""
        self._require_solution()
        return sum(cycle.Q_cool for cycle in self._subcycles)

    @property
    def Q_cool_arr(self) -> Optional[float]:
        """Per-subcycle cooling delivered to the process."""
        self._require_solution()
        return np.array([cycle.Q_cool for cycle in self._subcycles])
    
    @property
    def Q_cond(self) -> Optional[float]:
        """Total condenser duty across all subcycles."""
        self._require_solution()
        return sum(cycle.Q_cond for cycle in self._subcycles)
    
    @property
    def Q_cond_arr(self) -> Optional[float]:
        """Per-subcycle condenser duties."""
        self._require_solution()
        return np.array([cycle.Q_cond for cycle in self._subcycles])

    @property
    def Q_cas_heat(self) -> Optional[float]:
        """Total heat supplied to upper cascade stages."""
        self._require_solution()
        return sum(cycle.Q_cas_heat for cycle in self._subcycles)
 
    @property
    def Q_cas_heat_arr(self) -> Optional[float]:
        """Per-subcycle heat supplied to upper cascade stages."""
        self._require_solution()
        return np.array([cycle.Q_cas_heat for cycle in self._subcycles])
    
    @property
    def Q_heat(self) -> Optional[float]:
        """Total heat delivered to the process."""
        self._require_solution()
        return sum(cycle.Q_heat for cycle in self._subcycles)

    @property
    def Q_heat_arr(self) -> Optional[float]:
        """Per-subcycle heat delivered to the process."""
        self._require_solution()
        return np.array([cycle.Q_heat for cycle in self._subcycles])

    @property
    def work(self) -> Optional[float]:
        """Total compressor work, or the infeasibility penalty while unsolved."""
        if self.solved:
            return sum(cycle.work for cycle in self._subcycles)
        else:
            return self._max_work

    @property
    def work_arr(self) -> Optional[float]:
        """Per-subcycle compressor work."""
        self._require_solution()
        return np.array([cycle.work for cycle in self._subcycles])

    @property
    def penalty(self) -> Optional[float]:
        """Total penalty for excessive subcooling."""
        if self.solved:
            return sum(cycle.penalty if cycle.solved else 0 for cycle in self._subcycles)
        else:
            return 0
    
    @property
    def dtcont(self) -> Optional[float]:
        """Minimum temperature approach propagated to derived stream profiles."""
        return self._dtcont

    @property
    def COP_h(self) -> Optional[float]:
        """Heating coefficient of performance for the full cascade."""
        self._require_solution()
        if abs(self.work) <= 1e-9:
            raise ZeroDivisionError("COP_h is undefined when net work is zero.")
        return self.Q_heat / self.work

    @property
    def COP_r(self) -> Optional[float]:
        """Cooling coefficient of performance for the full cascade."""
        self._require_solution()
        if abs(self.work) <= 1e-9:
            raise ZeroDivisionError("COP_r is undefined when net work is zero.")
        return self.Q_cool / self.work
    
    @property
    def COP_o(self) -> Optional[float]:
        """Overall coefficient of performance based on heating plus cooling."""
        self._require_solution()
        if abs(self.work) <= 1e-9:
            raise ZeroDivisionError("COP_o is undefined when net work is zero.")
        return (self.Q_heat + self.Q_cool) / self.work

    @property
    def dt_diff_max(self) -> Optional[float]:
        """Maximum piecewise temperature error for derived stream profiles."""
        return self._dt_diff_max

    @property
    def refrigerant(self) -> np.ndarray:
        """Refrigerant assigned to each solved subcycle."""
        self._require_solution()
        return np.array([cycle.refrigerant for cycle in self._subcycles])

    @property
    def T_evap(self) -> np.ndarray:
        """Evaporating temperatures for each solved subcycle."""
        self._require_solution()
        return np.array([cycle.T_evap for cycle in self._subcycles])

    @property
    def T_cond(self) -> np.ndarray:
        """Condensing temperatures for each solved subcycle."""
        self._require_solution()
        return np.array([cycle.T_cond for cycle in self._subcycles])

    @property
    def dT_superheat(self) -> np.ndarray:
        """Applied superheat for each solved subcycle."""
        self._require_solution()
        return np.array([cycle.dT_superheat for cycle in self._subcycles])

    @property
    def dT_subcool(self) -> np.ndarray:
        """Applied subcooling for each solved subcycle."""
        self._require_solution()
        return np.array([cycle.dT_subcool for cycle in self._subcycles])

    @property
    def eta_comp(self) -> np.ndarray:
        """Compressor efficiency used for each solved subcycle."""
        self._require_solution()
        return np.array([cycle.eta_comp for cycle in self._subcycles])

    @property
    def dt_ihx_gas_side(self) -> np.ndarray:
        """Internal heat exchanger gas-side delta-T for each subcycle."""
        self._require_solution()
        return np.array([cycle.dt_ihx_gas_side for cycle in self._subcycles])

    @property
    def dt_cascade_hx(self) -> float:
        """Minimum approach temperature enforced between neighbouring stages."""
        self._require_solution()
        return self._dt_cascade_hx
    
    @property
    def num_cycles(self) -> int:
        """Number of simple heat pump subcycles in the cascade."""
        return self._num_cycles
    
    @property
    def subcycles(self) -> List[SimpleHeatPumpCycle]:
        """Solved simple heat pump subcycles that make up the cascade."""
        return self._subcycles    

    @property
    def solved(self) -> bool:
        """Whether the cascade has been solved successfully."""
        return self._solved   

    def _as_1d_numeric_array(
        self,
        values,
        *,
        default: float = 0.0,
    ) -> np.ndarray:
        if values is None:
            values = default
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError("Input must be numeric, None, or np.nan.") from e

        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("Incompatible input to solving a cascade heat pump.")
        if np.isnan(arr).all():
            arr = np.array([default], dtype=float)
        return arr


    def _normalize_dT_superheat(
        self,
        dT_superheat: np.ndarray,
        n_heat: int,
        n_cool: int,
    ) -> np.ndarray:
        arr = self._as_1d_numeric_array(dT_superheat, default=0.0)
        n_cycles = n_heat + n_cool - 1
        if arr.size == n_cycles:
            return arr
        if arr.size == 1:
            return np.full(n_cycles, arr.item(), dtype=float)
        if arr.size == n_cool:
            return np.concatenate([np.zeros(n_heat - 1), arr])
        raise ValueError("Incompatible dT_superheat input to solving a cascade heat pump.")


    def _normalize_dT_subcool(
        self,
        dT_subcool: np.ndarray,
        n_heat: int,
        n_cool: int,
    ) -> np.ndarray:
        arr = self._as_1d_numeric_array(dT_subcool, default=0.0)
        n_cycles = n_heat + n_cool - 1
        if arr.size == n_cycles:
            return arr
        if arr.size == 1:
            return np.full(n_cycles, arr.item(), dtype=float)
        if arr.size == n_heat:
            return np.concatenate([arr, np.zeros(n_cool - 1)])
        raise ValueError("Incompatible dT_subcool input to solving a cascade heat pump.")


    def _normalize_Q_heat(
        self,
        Q_heat: np.ndarray,
        n_heat: int,
        n_cool: int,
    ) -> np.ndarray:
        if Q_heat is None:
            arr = np.array([0.0], dtype=float)
        else:
            # Preserve NaNs here so we can apply the stage-specific default rule below.
            arr = self._as_1d_numeric_array(Q_heat, default=np.nan)
        n_cycles = n_heat + n_cool - 1
        if arr.size == n_cycles:
            arr_out = arr
        elif arr.size == 1:
            arr_out = np.full(n_cycles, arr.item(), dtype=float)
        elif arr.size == n_heat:
            arr_out = np.concatenate([arr, np.zeros(n_cool - 1)])
        else:
            raise ValueError("Incompatible Q_heat input to solving a cascade heat pump.")

        # For Q_heat only: NaN defaults to 1 at the first stage and 0 otherwise.
        nan_mask = np.isnan(arr_out)
        if np.any(nan_mask):
            nan_defaults = np.zeros_like(arr_out)
            nan_defaults[0] = 1.0
            arr_out = np.where(nan_mask, nan_defaults, arr_out)

        return arr_out


    def _normalize_Q_cool(
        self,
        Q_cool: np.ndarray,
        n_heat: int,
        n_cool: int,
    ) -> np.ndarray:
        n_cycles = n_heat + n_cool - 1

        if Q_cool is None:
            return np.array([0.0] * (n_cycles - 1) + [None], dtype=object)

        arr = np.asarray(Q_cool, dtype=object)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("Incompatible Q_cool input to solving a cascade heat pump.")

        if arr.size == 1:
            v = arr[0]
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                arr = np.array([0.0] * (n_cycles - 1) + [None], dtype=object)
            else:
                arr = np.full(n_cycles, float(v), dtype=object)
        elif arr.size == n_cycles:
            arr = arr.copy()
        elif arr.size == n_cool:
            arr = np.concatenate([np.zeros(n_heat - 1, dtype=object), arr]).astype(object)
        else:
            raise ValueError("Incompatible Q_cool input to solving a cascade heat pump.")

        for i in range(n_cycles - 1):
            v = arr[i]
            if v is None:
                raise ValueError("Only the last Q_cool value may be None or np.nan.")
            try:
                v_float = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError("Q_cool values must be numeric, None, or np.nan.") from e
            if np.isnan(v_float):
                raise ValueError("Only the last Q_cool value may be None or np.nan.")
            arr[i] = v_float

        last = arr[-1]
        if last is None:
            arr[-1] = None
        else:
            try:
                last_float = float(last)
            except (TypeError, ValueError) as e:
                raise ValueError("Q_cool values must be numeric, None, or np.nan.") from e
            arr[-1] = None if np.isnan(last_float) else last_float

        return arr
    

    def _validate_T_cond_and_evap(self, T_cond: np.ndarray, T_evap: np.ndarray) -> float:
        return (
            np.min([T_cond.min() - T_evap.max() + self._dt_cascade_hx, 0.0])
            + 
            np.min([(T_cond - np.roll(T_cond, 1))[:-1].sum(), 0.0]) 
            +
            np.min([(T_evap - np.roll(T_evap, 1))[:-1].sum(), 0.0])
        ) * -1


    def solve(
        self,
        T_evap: np.ndarray,
        T_cond: np.ndarray, 
        *,
        dT_superheat: np.ndarray = 0.0,
        dT_subcool: np.ndarray = 0.0,
        eta_comp: float = 0.7,
        refrigerant: List[str] | str = "water",
        dt_ihx_gas_side: np.ndarray | float = 10.0,
        Q_heat: np.ndarray = np.array([1]),
        Q_cool: np.ndarray | float | None = None,
        dt_cascade_hx: float = 1.0,
    ) -> float:
        """
        Solve the heat pump cycle for the provided operating point.

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
        dt_ihx_gas_side : np.ndarray | float, optional
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

        self._max_work = Q_heat.sum()
        inf = self._validate_T_cond_and_evap(T_cond, T_evap)
        if inf > 0.0:
            self._max_work *= (inf + 1)
            return self._max_work

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

        dT_superheat_all = self._normalize_dT_superheat(dT_superheat, n_heat, n_cool)
        dT_subcool_all = self._normalize_dT_subcool(dT_subcool, n_heat, n_cool)
        Q_heat_all = self._normalize_Q_heat(Q_heat, n_heat, n_cool)
        Q_cool_all = self._normalize_Q_cool(Q_cool, n_heat, n_cool)

        if isinstance(refrigerant, list):
            if len(refrigerant) == self._num_cycles:
                refrigerant_all = refrigerant
            elif len(refrigerant) == 1:
                refrigerant_all = refrigerant * self._num_cycles
            else:
                raise ValueError(f"Number of refrigerants must match the number of heat pumps, {self._num_cycles}.")
        else:
            refrigerant_all = [refrigerant] * self._num_cycles

        if np.isscalar(dt_ihx_gas_side):
            ihx_gas_dt_all = np.full(self._num_cycles, dt_ihx_gas_side, dtype=float)
        else:
            ihx_gas_dt_all = np.asarray(dt_ihx_gas_side, dtype=float)
            if ihx_gas_dt_all.size != self._num_cycles:
                raise ValueError("dt_ihx_gas_side must match the number of heat pumps.")

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
                dt_ihx_gas_side=ihx_gas_dt_all[i],
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
        """Combine piecewise stream approximations from every solved subcycle."""
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

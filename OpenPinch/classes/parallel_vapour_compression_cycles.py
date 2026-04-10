"""Parallel heat pump network assembled from multiple simple subcycles."""

from __future__ import annotations

from typing import Optional, List
import numpy as np

from .stream_collection import StreamCollection
from .vapour_compression_cycle import VapourCompressionCycle


__all__ = ["ParallelVapourCompressionCycles"]


class ParallelVapourCompressionCycles:
    """Parallel set of vapour-compression heat pumps solved independently."""

    def __init__(self):
        """Initialise an unsolved multi-cycle heat pump model."""
        self._subcycles = []
        self._num_cycles = 1
        self._dtcont: float = 0.0
        self._dt_diff_max: float = 0.5  # Default value, used in piecewise approximation of non linear T-h profiles
        self._solved: bool = False

    @property
    def Q_evap(self) -> Optional[float]:
        """Total evaporator duty across all subcycles."""
        self._require_solution()
        return sum(cycle.Q_evap for cycle in self._subcycles)

    @property
    def Q_evap_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle evaporator duties."""
        self._require_solution()
        return np.array([cycle.Q_evap for cycle in self._subcycles])

    @property
    def Q_cas_cool(self) -> Optional[float]:
        """Total cooling handed off to cascade coupling, if used."""
        self._require_solution()
        return sum(cycle.Q_cas_cool for cycle in self._subcycles)

    @property
    def Q_cas_cool_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle cooling handed off to cascade coupling."""
        self._require_solution()
        return np.array([cycle.Q_cas_cool for cycle in self._subcycles])

    @property
    def Q_cool(self) -> Optional[float]:
        """Total cooling delivered to the process."""
        self._require_solution()
        return sum(cycle.Q_cool for cycle in self._subcycles)

    @property
    def Q_cool_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle cooling delivered to the process."""
        self._require_solution()
        return np.array([cycle.Q_cool for cycle in self._subcycles])

    @property
    def Q_cond(self) -> Optional[float]:
        """Total condenser duty across all subcycles."""
        self._require_solution()
        return sum(cycle.Q_cond for cycle in self._subcycles)

    @property
    def Q_cond_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle condenser duties."""
        self._require_solution()
        return np.array([cycle.Q_cond for cycle in self._subcycles])

    @property
    def Q_cas_heat(self) -> Optional[float]:
        """Total heat handed off to any downstream cascade usage."""
        self._require_solution()
        return sum(cycle.Q_cas_heat for cycle in self._subcycles)

    @property
    def Q_cas_heat_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle heat handed off to any downstream cascade usage."""
        self._require_solution()
        return np.array([cycle.Q_cas_heat for cycle in self._subcycles])

    @property
    def Q_heat(self) -> Optional[float]:
        """Total heat delivered to the process."""
        self._require_solution()
        return sum(cycle.Q_heat for cycle in self._subcycles)

    @property
    def Q_heat_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle heat delivered to the process."""
        self._require_solution()
        return np.array([cycle.Q_heat for cycle in self._subcycles])

    @property
    def work(self) -> Optional[float]:
        """Total compressor work across all subcycles."""
        self._require_solution()
        return sum(cycle.work for cycle in self._subcycles)

    @property
    def work_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle compressor work."""
        self._require_solution()
        return np.array([cycle.work for cycle in self._subcycles])

    @property
    def penalty(self) -> Optional[float]:
        """Total penalty for excessive subcooling."""
        if self.solved:
            return sum(
                cycle.penalty if cycle.solved else 0 for cycle in self._subcycles
            )
        else:
            return 0

    @property
    def dtcont(self) -> Optional[float]:
        """Minimum temperature approach propagated to derived stream profiles."""
        return self._dtcont

    @property
    def COP_h(self) -> Optional[float]:
        """Heating coefficient of performance for the full network."""
        self._require_solution()
        if abs(self.work) <= 1e-9:
            raise ZeroDivisionError("COP_h is undefined when net work is zero.")
        return self.Q_heat / self.work

    @property
    def COP_r(self) -> Optional[float]:
        """Cooling coefficient of performance for the full network."""
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
    def num_cycles(self) -> int:
        """Number of simple heat pump subcycles in the network."""
        return self._num_cycles

    @property
    def subcycles(self) -> List[VapourCompressionCycle]:
        """Solved simple heat pump subcycles that make up the network."""
        return self._subcycles

    @property
    def solved(self) -> bool:
        """Whether the parallel heat pumps have all been solved successfully."""
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
            raise ValueError(
                "Incompatible input to solving a multiple simple heat pump system."
            )
        if np.isnan(arr).all():
            arr = np.array([default], dtype=float)
        return arr

    def _normalize_temperature_arrays(
        self,
        T_evap,
        T_cond,
    ) -> tuple[np.ndarray, np.ndarray]:
        T_evap_arr = self._as_1d_numeric_array(T_evap, default=np.nan)
        T_cond_arr = self._as_1d_numeric_array(T_cond, default=np.nan)

        if np.isnan(T_evap_arr).any() or np.isnan(T_cond_arr).any():
            raise ValueError("Evaporator and condenser temperatures must be numeric.")

        if T_evap_arr.size == T_cond_arr.size:
            pass
        elif T_evap_arr.size == 1:
            T_evap_arr = np.full(T_cond_arr.size, T_evap_arr.item(), dtype=float)
        elif T_cond_arr.size == 1:
            T_cond_arr = np.full(T_evap_arr.size, T_cond_arr.item(), dtype=float)
        else:
            raise ValueError(
                "T_evap and T_cond must be scalar or have matching lengths."
            )

        if np.any(T_cond_arr <= T_evap_arr):
            raise ValueError("Invalid condenser and evaporator temperatures.")

        return T_evap_arr, T_cond_arr

    def _normalize_per_cycle_array(
        self,
        values,
        n_cycles: int,
        *,
        default: float = 0.0,
        name: str = "input",
    ) -> np.ndarray:
        arr = self._as_1d_numeric_array(values, default=default)
        if arr.size == n_cycles:
            return arr
        if arr.size == 1:
            return np.full(n_cycles, arr.item(), dtype=float)
        raise ValueError(f"{name} must be scalar or have one value per heat pump.")

    def _normalize_Q_heat(
        self,
        Q_heat,
        n_cycles: int,
    ) -> np.ndarray:
        if Q_heat is None:
            arr = np.array([1.0], dtype=float)
        else:
            arr = self._as_1d_numeric_array(Q_heat, default=np.nan)

        if arr.size == n_cycles:
            arr_out = arr
        elif arr.size == 1:
            arr_out = np.full(n_cycles, arr.item(), dtype=float)
        else:
            raise ValueError(
                "Incompatible Q_heat input to solving a multiple simple heat pump system."
            )

        nan_mask = np.isnan(arr_out)
        if np.any(nan_mask):
            arr_out = np.where(nan_mask, 1.0, arr_out)

        return arr_out

    def _normalize_Q_cool(
        self,
        Q_cool,
        n_cycles: int,
    ) -> np.ndarray:
        if Q_cool is None:
            return np.array([None] * n_cycles, dtype=object)

        arr = np.asarray(Q_cool, dtype=object)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(
                "Incompatible Q_cool input to solving a multiple simple heat pump system."
            )

        if arr.size == 1:
            v = arr[0]
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                arr = np.array([None] * n_cycles, dtype=object)
            else:
                arr = np.full(n_cycles, float(v), dtype=object)
        elif arr.size == n_cycles:
            arr = arr.copy()
        else:
            raise ValueError(
                "Incompatible Q_cool input to solving a multiple simple heat pump system."
            )

        for i in range(n_cycles):
            v = arr[i]
            if v is None:
                arr[i] = None
                continue
            try:
                v_float = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Q_cool values must be numeric, None, or np.nan."
                ) from e
            arr[i] = None if np.isnan(v_float) else v_float

        return arr

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
        Q_heat: np.ndarray | float | None = None,
        Q_cool: np.ndarray | float | None = None,
        is_heat_pump: bool = True,
    ) -> float:
        """
        Solve a set of parallel simple heat pump cycles.

        Parameters
        ----------
        T_evap : np.ndarray
            Liquid saturation temperatures in the evaporator [deg C].
        T_cond : np.ndarray
            Gas saturation temperatures in the condenser [deg C].
        dT_superheat : np.ndarray, optional
            Degree of superheating of the suction gas [K].
        dT_subcool : np.ndarray, optional
            Degree of subcooling after the condenser [K].
        eta_comp : float, optional
            Isentropic efficiency of the compressor [-].
        refrigerant : List[str] | str, optional
            Cycle refrigerants; one per heat pump or a scalar value.
        dt_ihx_gas_side : np.ndarray | float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : np.ndarray | float | None, optional
            Heat delivered to the process [W].
        Q_cool : np.ndarray | float | None, optional
            Cooling delivered to the process [W].
        is_heat_pump : bool, optional
            Flag to indicate if the cycle is in heat pump or refrigeration mode.

        Returns
        -------
        float
            Total compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self._subcycles = []

        T_evap_all, T_cond_all = self._normalize_temperature_arrays(T_evap, T_cond)
        self._num_cycles = T_evap_all.size

        dT_superheat_all = self._normalize_per_cycle_array(
            dT_superheat,
            self._num_cycles,
            default=0.0,
            name="dT_superheat",
        )
        dT_subcool_all = self._normalize_per_cycle_array(
            dT_subcool,
            self._num_cycles,
            default=0.0,
            name="dT_subcool",
        )
        Q_heat_all = self._normalize_Q_heat(Q_heat, self._num_cycles)
        Q_cool_all = self._normalize_Q_cool(Q_cool, self._num_cycles)

        if isinstance(refrigerant, list):
            if len(refrigerant) == self._num_cycles:
                refrigerant_all = refrigerant
            elif len(refrigerant) == 1:
                refrigerant_all = refrigerant * self._num_cycles
            else:
                raise ValueError(
                    f"Number of refrigerants must match the number of heat pumps, {self._num_cycles}."
                )
        else:
            refrigerant_all = [refrigerant] * self._num_cycles

        if np.isscalar(dt_ihx_gas_side):
            ihx_gas_dt_all = np.full(self._num_cycles, dt_ihx_gas_side, dtype=float)
        else:
            ihx_gas_dt_all = np.asarray(dt_ihx_gas_side, dtype=float)
            if ihx_gas_dt_all.size != self._num_cycles:
                raise ValueError("dt_ihx_gas_side must match the number of heat pumps.")

        for i in range(self._num_cycles):
            hp = VapourCompressionCycle()
            hp.solve(
                T_evap=T_evap_all[i],
                T_cond=T_cond_all[i],
                dT_superheat=dT_superheat_all[i],
                dT_subcool=dT_subcool_all[i],
                eta_comp=eta_comp,
                refrigerant=refrigerant_all[i],
                dt_ihx_gas_side=ihx_gas_dt_all[i],
                Q_heat=Q_heat_all[i],
                Q_cool=Q_cool_all[i],
                is_heat_pump=is_heat_pump,
            )
            self._subcycles.append(hp)

        Q_heat_arr = np.array([cycle.Q_heat for cycle in self._subcycles])
        if np.all(np.isclose(Q_heat_all, Q_heat_arr)):
            self._solved = True

        return sum(cycle.work for cycle in self._subcycles)

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
            raise RuntimeError("Solve the cycle before accessing results.")

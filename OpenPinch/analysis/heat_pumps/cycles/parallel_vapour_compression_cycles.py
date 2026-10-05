"""Parallel heat pump network assembled from independent subcycles."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..common.encoding import DutyAllocationRequest
from ._multi_cycle_base import _MultiVapourCompressionCycleBase
from .vapour_compression_cycle import VapourCompressionCycle

__all__ = ["ParallelVapourCompressionCycles"]


class ParallelVapourCompressionCycles(_MultiVapourCompressionCycleBase):
    """Parallel set of vapour-compression heat pumps solved independently."""

    _SHAPE_ERROR = "Incompatible input to solving a parallel heat pump system."

    def __init__(self):
        """Initialise an unsolved parallel heat pump model."""
        super().__init__()

    @property
    def work(self) -> Optional[float]:
        """Total compressor work across all subcycles."""
        self._require_solution()
        return sum(cycle.work for cycle in self._subcycles)

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
        """Return one process heat duty per cycle, ``None`` where unset.

        Each subcycle settles an unset duty: a heat pump solves for a unit
        duty and a refrigerator sends all its condenser heat to the process.
        """
        if Q_heat is None:
            return np.array([None] * n_cycles, dtype=object)

        arr = self._as_1d_numeric_array(Q_heat, default=np.nan)
        if arr.size == n_cycles:
            arr_out = arr
        elif arr.size == 1:
            arr_out = np.full(n_cycles, arr.item(), dtype=float)
        else:
            raise ValueError(
                "Incompatible Q_heat input for solving a parallel heat pump system."
            )

        return np.array(
            [None if np.isnan(value) else float(value) for value in arr_out],
            dtype=object,
        )

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
                "Incompatible Q_cool input for solving a parallel heat pump system."
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
                "Incompatible Q_cool input for solving a parallel heat pump system."
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

    def _allocate_process_duties(
        self,
        *,
        n_cycles: int,
        Q_heat,
        Q_cool,
        duty_allocation: DutyAllocationRequest,
        is_heat_pump: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if is_heat_pump and duty_allocation.heat.Q_base is not None:
            allocation = duty_allocation.heat.allocate("heat")
            return allocation.Q_model, self._normalize_Q_cool(Q_cool, n_cycles)

        if (not is_heat_pump) and duty_allocation.cool.Q_base is not None:
            allocation = duty_allocation.cool.allocate("cool")
            return self._normalize_Q_heat(Q_heat, n_cycles), allocation.Q_model

        return (
            self._normalize_Q_heat(Q_heat, n_cycles),
            self._normalize_Q_cool(Q_cool, n_cycles),
        )

    def _normalize_refrigerant(
        self,
        refrigerant: List[str] | str,
        n_cycles: int,
    ) -> List[str]:
        if isinstance(refrigerant, list):
            if len(refrigerant) == n_cycles:
                return refrigerant
            if len(refrigerant) == 1:
                return refrigerant * n_cycles
            raise ValueError(
                "Number of refrigerants must match the number of heat pumps, "
                f"{n_cycles}."
            )
        return [refrigerant] * n_cycles

    def _normalize_dT_ihx_gas_side(
        self,
        dT_ihx_gas_side,
        n_cycles: int,
    ) -> np.ndarray:
        if np.isscalar(dT_ihx_gas_side):
            return np.full(n_cycles, dT_ihx_gas_side, dtype=float)

        arr = np.asarray(dT_ihx_gas_side, dtype=float)
        if arr.size != n_cycles:
            raise ValueError("dT_ihx_gas_side must match the number of heat pumps.")
        return arr

    def solve(
        self,
        T_evap: np.ndarray,
        T_cond: np.ndarray,
        *,
        dtcont: float,
        dT_superheat: np.ndarray = 0.0,
        dT_subcool: np.ndarray = 0.0,
        eta_comp: float = 0.7,
        refrigerant: List[str] | str = "water",
        dT_ihx_gas_side: np.ndarray | float = 10.0,
        Q_heat: np.ndarray | float | None = None,
        Q_cool: np.ndarray | float | None = None,
        duty_allocation: DutyAllocationRequest | None = None,
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
        dtcont : float
            Minimum temperature approach used by HPR targeting [K].
        dT_superheat : np.ndarray, optional
            Degree of superheating of the suction gas [K].
        dT_subcool : np.ndarray, optional
            Degree of subcooling after the condenser [K].
        eta_comp : float, optional
            Isentropic efficiency of the compressor [-].
        refrigerant : List[str] | str, optional
            Cycle refrigerants; one per heat pump or a scalar value.
        dT_ihx_gas_side : np.ndarray | float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : np.ndarray | float | None, optional
            Heat delivered to the process [W].
        Q_cool : np.ndarray | float | None, optional
            Cooling delivered to the process [W].
        duty_allocation : DutyAllocationRequest, optional
            Base-duty/split/availability inputs; the primary side's allocation
            overrides ``Q_heat`` (heat pump) or ``Q_cool`` (refrigeration).
        is_heat_pump : bool, optional
            Flag to indicate if the cycle is in heat pump or refrigeration mode.

        Returns
        -------
        float
            Total compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self._subcycles = []
        self._dtcont = float(dtcont)

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
        Q_heat_all, Q_cool_all = self._allocate_process_duties(
            n_cycles=self._num_cycles,
            Q_heat=Q_heat,
            Q_cool=Q_cool,
            duty_allocation=duty_allocation or DutyAllocationRequest(),
            is_heat_pump=is_heat_pump,
        )
        refrigerant_all = self._normalize_refrigerant(refrigerant, self._num_cycles)
        ihx_gas_dt_all = self._normalize_dT_ihx_gas_side(
            dT_ihx_gas_side, self._num_cycles
        )

        for i in range(self._num_cycles):
            hp = VapourCompressionCycle()
            hp.solve(
                T_evap=T_evap_all[i],
                T_cond=T_cond_all[i],
                dtcont=self._dtcont,
                dT_superheat=dT_superheat_all[i],
                dT_subcool=dT_subcool_all[i],
                eta_comp=eta_comp,
                refrigerant=refrigerant_all[i],
                dT_ihx_gas_side=ihx_gas_dt_all[i],
                Q_heat=Q_heat_all[i],
                Q_cool=Q_cool_all[i],
                is_heat_pump=is_heat_pump,
            )
            self._subcycles.append(hp)

        self._solved = bool(np.all([cycle.solved for cycle in self._subcycles]))
        return sum(cycle.work for cycle in self._subcycles)

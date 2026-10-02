"""Cascade heat pump network assembled from staged subcycles."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..common.encoding import DutyAllocationRequest
from ._multi_cycle_base import _MultiVapourCompressionCycleBase
from .vapour_compression_cycle import VapourCompressionCycle

__all__ = ["CascadeVapourCompressionCycle"]

# TODO: Implement cascade for refrigerant mixtures, not just pure fluids.


class CascadeVapourCompressionCycle(_MultiVapourCompressionCycleBase):
    """Cascade of vapour-compression heat pumps coupled through cascade exchangers."""

    _SHAPE_ERROR = "Incompatible input to solving a cascade heat pump."

    def __init__(self):
        """Initialise an unsolved cascade with no configured subcycles."""
        super().__init__()
        self._max_work: float = 0.0

    @property
    def work(self) -> Optional[float]:
        """Total compressor work, or the infeasibility penalty while unsolved."""
        if self.solved:
            return sum(cycle.work for cycle in self._subcycles)
        else:
            return self._max_work

    @property
    def dt_cascade_hx(self) -> float:
        """Minimum approach temperature enforced between neighbouring stages."""
        self._require_solution()
        return self._dt_cascade_hx

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
        raise ValueError(
            "Incompatible dT_superheat input to solving a cascade heat pump."
        )

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
        raise ValueError(
            "Incompatible dT_subcool input to solving a cascade heat pump."
        )

    def _normalize_Q_heat(
        self,
        Q_heat: np.ndarray,
        n_heat: int,
        n_cool: int,
    ) -> np.ndarray:
        n_cycles = n_heat + n_cool - 1
        if Q_heat is None:
            return np.array(
                [0.0] * max(n_heat - 1, 0) + [None] + [0.0] * (n_cool - 1),
                dtype=object,
            )

        arr = np.asarray(Q_heat, dtype=object)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(
                "Incompatible Q_heat input to solving a cascade heat pump."
            )

        if arr.size == n_cycles:
            arr_out = arr.copy()
        elif arr.size == 1:
            v = arr[0]
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                arr_out = np.array(
                    [0.0] * max(n_heat - 1, 0) + [None] + [0.0] * (n_cool - 1),
                    dtype=object,
                )
            else:
                arr_out = np.full(n_cycles, float(v), dtype=object)
        elif arr.size == n_heat:
            arr_out = np.concatenate([arr, np.zeros(n_cool - 1, dtype=object)])
        else:
            raise ValueError(
                "Incompatible Q_heat input to solving a cascade heat pump."
            )

        heat_default_idx = max(n_heat - 1, 0)
        for i in range(n_cycles):
            v = arr_out[i]
            if v is None:
                if i == heat_default_idx:
                    arr_out[i] = None
                    continue
                raise ValueError("Only the last Q_heat value may be None or np.nan.")
            try:
                v_float = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Q_heat values must be numeric, None, or np.nan."
                ) from e
            if np.isnan(v_float):
                if i == heat_default_idx:
                    arr_out[i] = None
                    continue
                raise ValueError("Only the last Q_heat value may be None or np.nan.")
            arr_out[i] = v_float

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
            raise ValueError(
                "Incompatible Q_cool input to solving a cascade heat pump."
            )

        if arr.size == 1:
            v = arr[0]
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                arr = np.array([0.0] * (n_cycles - 1) + [None], dtype=object)
            else:
                arr = np.full(n_cycles, float(v), dtype=object)
        elif arr.size == n_cycles:
            arr = arr.copy()
        elif arr.size == n_cool:
            arr = np.concatenate([np.zeros(n_heat - 1, dtype=object), arr]).astype(
                object
            )
        else:
            raise ValueError(
                "Incompatible Q_cool input to solving a cascade heat pump."
            )

        for i in range(n_cycles - 1):
            v = arr[i]
            if v is None:
                raise ValueError("Only the last Q_cool value may be None or np.nan.")
            try:
                v_float = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Q_cool values must be numeric, None, or np.nan."
                ) from e
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
                raise ValueError(
                    "Q_cool values must be numeric, None, or np.nan."
                ) from e
            arr[-1] = None if np.isnan(last_float) else last_float

        return arr

    def _validate_T_cond_and_evap(
        self, T_cond: np.ndarray, T_evap: np.ndarray
    ) -> float:
        return (
            # Every condenser must sit dt_cascade_hx above every evaporator.
            np.min([T_cond.min() - T_evap.max() - self._dt_cascade_hx, 0.0])
            + np.min([(T_cond - np.roll(T_cond, 1))[:-1].sum(), 0.0])
            + np.min([(T_evap - np.roll(T_evap, 1))[:-1].sum(), 0.0])
        ) * -1

    def _normalize_secondary_process_duty(self, duty=None) -> np.ndarray | None:
        if duty is None:
            return None
        duty_arr = np.asarray(duty)
        if duty_arr.size == 1:
            return duty_arr
        if duty_arr[-1] is not None:
            duty_arr[-1] = np.nan
        return duty_arr

    def _prepare_process_duty_inputs(
        self,
        Q_heat: np.ndarray,
        Q_cool: np.ndarray,
        *,
        is_heat_pump: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if is_heat_pump:
            Q_heat_out = np.asarray(Q_heat if Q_heat is not None else 1.0, dtype=float)
            Q_cool_out = self._normalize_secondary_process_duty(Q_cool)
            return Q_heat_out, Q_cool_out

        Q_cool_out = np.asarray(Q_cool if Q_cool is not None else 1.0, dtype=float)
        Q_heat_out = Q_heat
        return Q_heat_out, Q_cool_out

    def _allocate_process_duties(
        self,
        *,
        Q_heat,
        Q_cool,
        duty_allocation: DutyAllocationRequest,
        is_heat_pump: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        heat, cool = duty_allocation.heat, duty_allocation.cool
        if is_heat_pump and heat.Q_base is not None:
            heat_allocation = heat.allocate("heat")
            Q_cool_out = self._normalize_secondary_process_duty(Q_cool)
            if cool.Q_base is not None:
                cool_allocation = cool.allocate("cool")
                Q_cool_out = self._normalize_secondary_process_duty(
                    np.concatenate([cool_allocation.Q_model, np.array([np.nan])])
                )
            return heat_allocation.Q_model, Q_cool_out

        if (not is_heat_pump) and cool.Q_base is not None:
            cool_allocation = cool.allocate("cool")
            Q_heat_out = Q_heat
            if heat.Q_base is not None:
                heat_allocation = heat.allocate("heat")
                Q_heat_out = self._normalize_secondary_process_duty(
                    np.concatenate([heat_allocation.Q_model, np.array([np.nan])])
                )
            return Q_heat_out, cool_allocation.Q_model

        return self._prepare_process_duty_inputs(
            Q_heat,
            Q_cool,
            is_heat_pump=is_heat_pump,
        )

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
        Q_heat: np.ndarray = None,
        Q_cool: np.ndarray = None,
        duty_allocation: DutyAllocationRequest | None = None,
        dt_cascade_hx: float = 1.0,
        is_heat_pump: bool = True,
    ) -> float:
        """
        Solve the heat pump cycle for the provided operating point.

        Parameters
        ----------
        T_evap : np.ndarray
            Liquid saturation temperature in the evaporator [deg C].
        T_cond : np.ndarray
            Gas saturation temperature in the condenser [deg C].
        dtcont : float
            Minimum temperature approach used by HPR targeting [K].
        dT_superheat : np.ndarray, optional
            Degree of superheating of the suction gas, supplied by the process [K].
        dT_subcool : np.ndarray, optional
            Degree of subcooling after the condenser, heat delivered to the process [K].
        eta_comp : float, optional
            Isentropic efficiency of the compressor [-].
        refrigerant : List[str], optional
            Cycle refrigerant; supports multi-component fluids.
        dT_ihx_gas_side : np.ndarray | float, optional
            Delta-T on the gas side of the internal heat exchanger [K].
        Q_heat : np.ndarray, optional
            Heat delivered to the process [W].
        Q_cool : np.ndarray, optional
            Cooling delivered to the process [W].
        duty_allocation : DutyAllocationRequest, optional
            Base-duty/split/availability inputs; a side with ``Q_base`` set
            overrides ``Q_heat`` or ``Q_cool``.
        dt_cascade_hx : float, optional
            Temperature difference between condensing and evaporating
            temperatures in the cascade heat exchanger.
        is_heat_pump : bool, optional
            Flag to indicate if the cycle is in heat pump or refrigeration mode.

        Returns
        -------
        float
            Compressor power requirement for the solved operating point [W].
        """
        self._solved = False
        self._subcycles = []
        self._dtcont = float(dtcont)
        Q_heat, Q_cool = self._allocate_process_duties(
            Q_heat=Q_heat,
            Q_cool=Q_cool,
            duty_allocation=duty_allocation or DutyAllocationRequest(),
            is_heat_pump=is_heat_pump,
        )
        self._dt_cascade_hx = dt_cascade_hx

        T_cond = np.asarray(T_cond, dtype=float)
        T_evap = np.asarray(T_evap, dtype=float)

        def _finite_positive_sum(values) -> float:
            try:
                arr = np.asarray(values, dtype=float).reshape(-1)
            except TypeError, ValueError:
                return 0.0
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 0.0
            return float(np.maximum(finite, 0.0).sum())

        self._max_work = max(
            _finite_positive_sum(Q_heat),
            _finite_positive_sum(Q_cool),
            1.0,
        )
        inf = self._validate_T_cond_and_evap(T_cond, T_evap)
        if inf > 0.0:
            self._max_work *= inf + 1
            return self._max_work

        T_cond_all = np.sort(
            np.concatenate([T_cond, T_evap[:-1] + self._dt_cascade_hx])
        )[::-1]
        T_evap_all = np.sort(
            np.concatenate([T_cond[1:] - self._dt_cascade_hx, T_evap])
        )[::-1]

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
                raise ValueError(
                    "Number of refrigerants must match the number of heat pumps, "
                    f"{self._num_cycles}."
                )
        else:
            refrigerant_all = [refrigerant] * self._num_cycles

        if np.isscalar(dT_ihx_gas_side):
            ihx_gas_dt_all = np.full(self._num_cycles, dT_ihx_gas_side, dtype=float)
        else:
            ihx_gas_dt_all = np.asarray(dT_ihx_gas_side, dtype=float)
            if ihx_gas_dt_all.size != self._num_cycles:
                raise ValueError("dT_ihx_gas_side must match the number of heat pumps.")

        Q_cas_heat = 0.0
        Q_cas_cool = 0.0
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
                Q_cas_heat=Q_cas_heat,
                Q_cool=Q_cool_all[i],
                Q_cas_cool=Q_cas_cool,
                is_heat_pump=is_heat_pump,
            )
            self._subcycles.append(hp)
            if not hp.solved:
                failed_work = abs(float(hp.work or 0.0))
                failed_work = failed_work if np.isfinite(failed_work) else 1.0
                self._max_work += max(failed_work, 1.0)
                return self._max_work
            Q_cas_heat = hp.Q_cas_cool if is_heat_pump else 0.0
            Q_cas_cool = 0.0 if is_heat_pump else hp.Q_cas_heat

        # Finish analysis
        work = sum(cycle.work for cycle in self._subcycles)
        if not np.isfinite(float(work)) or float(work) < 0.0:
            failed_work = abs(float(work))
            failed_work = failed_work if np.isfinite(failed_work) else 1.0
            self._max_work = max(failed_work, 1.0)
            return self._max_work
        self._solved = True
        return self.work

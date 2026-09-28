"""Shared aggregate behaviour for multi-subcycle vapour-compression models."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ....domain.stream_collection import StreamCollection
from .vapour_compression_cycle import VapourCompressionCycle

__all__: list[str] = []


class _MultiVapourCompressionCycleBase:
    """Aggregate results over a set of solved vapour-compression subcycles."""

    _SHAPE_ERROR = "Incompatible input to solving a multi-cycle heat pump."

    def __init__(self):
        """Initialise the shared unsolved state for a multi-cycle model."""
        self._subcycles = []
        self._num_cycles = 1
        self._dtcont: float = 0.0
        # Default value used in piecewise approximation of non-linear T-h profiles.
        self._dt_diff_max: float = 0.5
        self._solved: bool = False
        self._allocation_penalty = np.empty(0, dtype=float)

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
    def work_arr(self) -> Optional[np.ndarray]:
        """Per-subcycle compressor work."""
        self._require_solution()
        return np.array([cycle.work for cycle in self._subcycles])

    @property
    def penalty(self) -> Optional[float]:
        """Total penalty for excessive subcooling."""
        if self.solved:
            cycle_penalty = sum(
                float(np.asarray(cycle.penalty, dtype=float).sum())
                for cycle in self._subcycles
                if cycle.solved
            )
            return cycle_penalty + float(self._allocation_penalty.sum())
        else:
            return float(self._allocation_penalty.sum())

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
    def dT_ihx_gas_side(self) -> np.ndarray:
        """Internal heat exchanger gas-side delta-T for each subcycle."""
        self._require_solution()
        return np.array([cycle.dT_ihx_gas_side for cycle in self._subcycles])

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
        """Whether every subcycle in the model has been solved successfully."""
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
            raise ValueError(self._SHAPE_ERROR)
        if np.isnan(arr).all():
            arr = np.array([default], dtype=float)
        return arr

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

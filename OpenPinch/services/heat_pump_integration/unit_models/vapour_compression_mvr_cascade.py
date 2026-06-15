"""Vapour-compression plus serial MVR cascade model."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ....classes.stream import Stream
from ....classes.stream_collection import StreamCollection
from ..common.encoding import require_stage_duty_allocation
from .mechanical_vapour_recompression_cycle import MechanicalVapourRecompressionCycle
from .vapour_compression_cycle import VapourCompressionCycle

__all__ = ["VapourCompressionMvrCascade"]


class VapourCompressionMvrCascade:
    """Cascade top VC condenser heat into a serial MVR vapour train."""

    MAX_MVR_STAGE_LIFT = 20.0

    def __init__(self):
        """Initialise an unsolved VC+MVR cascade."""
        self._vc_cycles: list[VapourCompressionCycle] = []
        self._mvr_cycles: list[MechanicalVapourRecompressionCycle] = []
        self._source_split = 0.0
        self._process_split = np.empty(0, dtype=float)
        self._direct_vc_heat = np.empty(0, dtype=float)
        self._internal_heat = np.empty(0, dtype=float)
        self._mvr_stage_heat = np.empty(0, dtype=float)
        self._mvr_stage_desuperheat = np.empty(0, dtype=float)
        self._mvr_stage_latent_heat = np.empty(0, dtype=float)
        self._mvr_stage_subcool_heat = np.empty(0, dtype=float)
        self._mvr_stage_mass_in = np.empty(0, dtype=float)
        self._mvr_stage_mass_out = np.empty(0, dtype=float)
        self._T_evap_mvr = np.empty(0, dtype=float)
        self._T_cond_mvr = np.empty(0, dtype=float)
        self._dT_subcool_mvr = np.empty(0, dtype=float)
        self._solved = False
        self._dtcont = 0.0
        self._dt_diff_max = 0.5
        self._penalty = np.empty(0, dtype=float)
        self._max_work = 0.0

    @property
    def solved(self) -> bool:
        """Whether all stages solved successfully."""
        return self._solved

    @property
    def vc_cycles(self) -> List[VapourCompressionCycle]:
        """Solved low-stage vapour-compression cycles."""
        return self._vc_cycles

    @property
    def mvr_cycles(self) -> List[MechanicalVapourRecompressionCycle]:
        """Solved high-stage MVR cycles."""
        return self._mvr_cycles

    @property
    def subcycles(self) -> list:
        """All solved subcycles in low-stage then high-stage order."""
        return [*self._vc_cycles, *self._mvr_cycles]

    @property
    def source_split(self) -> float:
        """Split of the hottest VC condenser duty used to generate MVR vapour."""
        return self._source_split

    @property
    def process_split(self) -> np.ndarray:
        """MVR stage vapour fractions condensed for process heating."""
        return self._process_split

    @property
    def internal_heat(self) -> np.ndarray:
        """Top-stage VC heat transferred internally to the first MVR source."""
        return self._internal_heat

    @property
    def direct_vc_heat(self) -> np.ndarray:
        """VC condenser heat left as external process heat."""
        return self._direct_vc_heat

    @property
    def mvr_stage_heat(self) -> np.ndarray:
        """Useful MVR process heat from each serial stage."""
        self._require_solution()
        return self._mvr_stage_heat

    @property
    def mvr_stage_mass_in(self) -> np.ndarray:
        """Source vapour mass flow entering each MVR stage before injection."""
        self._require_solution()
        return self._mvr_stage_mass_in

    @property
    def mvr_stage_mass_out(self) -> np.ndarray:
        """Post-injection uncondensed vapour mass flow leaving each MVR stage."""
        self._require_solution()
        return self._mvr_stage_mass_out

    @property
    def T_evap_mvr(self) -> np.ndarray:
        """Derived MVR evaporating/saturation temperatures."""
        self._require_solution()
        return self._T_evap_mvr

    @property
    def T_cond_mvr(self) -> np.ndarray:
        """Derived MVR condensing/saturation temperatures."""
        self._require_solution()
        return self._T_cond_mvr

    @property
    def Q_evap(self) -> Optional[float]:
        """External evaporator/source duty across VC stages."""
        self._require_solution()
        return float(sum(cycle.Q_evap for cycle in self._vc_cycles))

    @property
    def Q_evap_arr(self) -> np.ndarray:
        """Per-stage external evaporator/source duties."""
        self._require_solution()
        return np.array(
            [cycle.Q_evap for cycle in self._vc_cycles]
            + [0.0 for _ in self._mvr_cycles],
            dtype=float,
        )

    @property
    def Q_cond(self) -> Optional[float]:
        """External condenser/sink duty across all stages."""
        self._require_solution()
        return float(np.asarray(self.Q_heat_arr, dtype=float).sum())

    @property
    def Q_cond_arr(self) -> np.ndarray:
        """Per-stage external condenser/sink duties."""
        return self.Q_heat_arr

    @property
    def Q_heat(self) -> Optional[float]:
        """Total external useful heating duty."""
        self._require_solution()
        return self.Q_cond

    @property
    def Q_heat_arr(self) -> np.ndarray:
        """Per-stage external useful heating duties."""
        self._require_solution()
        return np.array(
            self._direct_vc_heat.tolist() + self._mvr_stage_heat.tolist(),
            dtype=float,
        )

    @property
    def Q_cool(self) -> Optional[float]:
        """Total external cooling/source duty."""
        self._require_solution()
        return self.Q_evap

    @property
    def Q_cool_arr(self) -> np.ndarray:
        """Per-stage external cooling/source duties."""
        return self.Q_evap_arr

    @property
    def work(self) -> Optional[float]:
        """Total electric work, or finite infeasibility work if unsolved."""
        if self.solved:
            return float(sum(cycle.work for cycle in self.subcycles))
        return self._max_work

    @property
    def work_arr(self) -> np.ndarray:
        """Per-stage electric work."""
        self._require_solution()
        return np.array([cycle.work for cycle in self.subcycles], dtype=float)

    @property
    def COP_h(self) -> Optional[float]:
        """Heating COP for the full cascade."""
        self._require_solution()
        if abs(self.work) <= 1e-9:
            raise ZeroDivisionError("COP_h is undefined when electric work is zero.")
        return self.Q_heat / self.work

    @property
    def penalty(self) -> np.ndarray:
        """Finite infeasibility and soft-constraint penalties."""
        if self.solved:
            cycle_penalties = [
                np.asarray(cycle.penalty, dtype=float).reshape(-1)
                for cycle in self.subcycles
                if cycle.penalty is not None
            ]
            if cycle_penalties:
                return np.concatenate([self._penalty, *cycle_penalties])
        return self._penalty

    @property
    def T_evap(self) -> np.ndarray:
        """Evaporating temperatures for VC and MVR stages."""
        self._require_solution()
        return np.array([cycle.T_evap for cycle in self.subcycles], dtype=float)

    @property
    def T_cond(self) -> np.ndarray:
        """Condensing temperatures for VC and MVR stages."""
        self._require_solution()
        return np.array([cycle.T_cond for cycle in self.subcycles], dtype=float)

    def solve(
        self,
        *,
        T_evap_vc: np.ndarray,
        T_cond_vc: np.ndarray,
        dT_lift_mvr: np.ndarray,
        Q_heat_vc: np.ndarray | None = None,
        mvr_source_split: float = 0.0,
        mvr_process_split: np.ndarray | float | None = None,
        Q_heat_base: float | None = None,
        x_heat_split: np.ndarray | None = None,
        Q_heat_available: np.ndarray | None = None,
        dT_subcool_vc: np.ndarray | float = 0.0,
        dT_subcool_mvr: np.ndarray | float = 0.0,
        dT_ihx_gas_side_vc: np.ndarray | float = 0.0,
        eta_comp: float = 0.7,
        eta_mvr_comp: float = 0.7,
        eta_motor: float = 1.0,
        refrigerant: list[str] | str = "water",
        mvr_fluid: list[str] | str = "Water",
        dt_cascade_hx: float = 0.0,
        dtcont: float = 0.0,
    ) -> float:
        """Solve the VC+MVR cascade for serial MVR lift and split variables."""
        self._solved = False
        self._vc_cycles = []
        self._mvr_cycles = []
        self._penalty = np.empty(0, dtype=float)
        self._dtcont = float(dtcont)

        T_evap_vc = np.asarray(T_evap_vc, dtype=float).reshape(-1)
        T_cond_vc = np.asarray(T_cond_vc, dtype=float).reshape(-1)
        if Q_heat_base is not None:
            allocation = require_stage_duty_allocation(
                Q_base=Q_heat_base,
                x_split=x_heat_split,
                Q_available=Q_heat_available,
                duty_name="heat",
            )
            Q_heat_vc = allocation.Q_model
            self._penalty = allocation.Q_excess
        else:
            if Q_heat_vc is None:
                raise ValueError(
                    "Either Q_heat_vc or Q_heat_base/x_heat_split must be provided."
                )
            Q_heat_vc = np.asarray(Q_heat_vc, dtype=float).reshape(-1)
        dT_lift_mvr = np.asarray(dT_lift_mvr, dtype=float).reshape(-1)
        n_vc = Q_heat_vc.size
        n_mvr = dT_lift_mvr.size
        self._max_work = max(float(np.maximum(Q_heat_vc, 0.0).sum()), 1.0)

        if n_vc < 1 or n_mvr < 1:
            raise ValueError("VC+MVR cascade requires at least one VC and MVR stage.")
        if T_evap_vc.size != n_vc or T_cond_vc.size != n_vc:
            raise ValueError("VC+MVR cascade input shapes are inconsistent.")

        dT_subcool_vc = self._normalise_stage_array(dT_subcool_vc, n_vc)
        dT_subcool_mvr = self._normalise_stage_array(dT_subcool_mvr, n_mvr)
        dT_ihx_gas_side_vc = self._normalise_stage_array(
            dT_ihx_gas_side_vc,
            n_vc,
        )
        process_split = self._normalise_process_split(mvr_process_split, n_mvr)
        refrigerant_all = self._normalise_fluid_list(refrigerant, n_vc)
        mvr_fluid_all = self._normalise_fluid_list(mvr_fluid, n_mvr)

        penalties = []
        penalties.extend(
            np.maximum(
                T_evap_vc + dtcont - (T_cond_vc - dT_subcool_vc),
                0.0,
            )
            * self._max_work
        )
        penalties.extend(np.maximum(1e-6 - dT_lift_mvr, 0.0) * self._max_work)
        penalties.extend(
            np.maximum(
                dT_lift_mvr - self.MAX_MVR_STAGE_LIFT,
                0.0,
            )
            * self._max_work
        )
        penalties.extend(
            np.maximum(
                dT_subcool_mvr - dT_lift_mvr,
                0.0,
            )
            * self._max_work
        )
        penalties.append(max(-float(mvr_source_split), 0.0) * self._max_work)
        penalties.append(max(float(mvr_source_split) - 1.0, 0.0) * self._max_work)
        penalties.extend(np.maximum(-process_split, 0.0) * self._max_work)
        penalties.extend(np.maximum(process_split - 1.0, 0.0) * self._max_work)
        self._penalty = np.concatenate(
            [self._penalty, np.asarray(penalties, dtype=float).reshape(-1)]
        )
        if np.any(self._penalty > 0.0):
            self._max_work *= 1.0 + float(self._penalty.sum()) / self._max_work
            return self._max_work

        self._source_split = float(np.clip(mvr_source_split, 0.0, 1.0))
        self._process_split = np.clip(process_split, 0.0, 1.0)
        self._dT_subcool_mvr = dT_subcool_mvr
        self._internal_heat = np.zeros(n_vc, dtype=float)
        self._internal_heat[0] = self._source_split * Q_heat_vc[0]
        self._direct_vc_heat = Q_heat_vc.copy()
        self._direct_vc_heat[0] -= self._internal_heat[0]

        for i in range(n_vc):
            cycle = VapourCompressionCycle()
            cycle.solve(
                T_evap=T_evap_vc[i],
                T_cond=T_cond_vc[i],
                dT_subcool=dT_subcool_vc[i],
                eta_comp=eta_comp,
                refrigerant=refrigerant_all[i],
                dT_ihx_gas_side=dT_ihx_gas_side_vc[i],
                Q_heat=self._direct_vc_heat[i],
                Q_cas_heat=self._internal_heat[i],
                Q_cool=None,
                is_heat_pump=True,
            )
            self._vc_cycles.append(cycle)
            if not cycle.solved:
                failed_work = abs(float(cycle.work or 0.0))
                failed_work = failed_work if np.isfinite(failed_work) else 1.0
                self._penalty = np.concatenate(
                    [
                        self._penalty,
                        np.array([max(failed_work, 1.0)], dtype=float),
                    ]
                )
                self._max_work += max(failed_work, 1.0)
                return self._max_work

        T_evap_mvr, T_cond_mvr = self._derive_mvr_temperatures(
            vc_cycle=self._vc_cycles[0],
            dT_lift_mvr=dT_lift_mvr,
            dt_cascade_hx=float(dt_cascade_hx),
        )
        self._T_evap_mvr = T_evap_mvr
        self._T_cond_mvr = T_cond_mvr

        self._mvr_stage_heat = np.zeros(n_mvr, dtype=float)
        self._mvr_stage_desuperheat = np.zeros(n_mvr, dtype=float)
        self._mvr_stage_latent_heat = np.zeros(n_mvr, dtype=float)
        self._mvr_stage_subcool_heat = np.zeros(n_mvr, dtype=float)
        self._mvr_stage_mass_in = np.zeros(n_mvr, dtype=float)
        self._mvr_stage_mass_out = np.zeros(n_mvr, dtype=float)
        m_dot_in: float | None = None
        for j in range(n_mvr):
            cycle = MechanicalVapourRecompressionCycle()
            if j == 0:
                solve_mvr_stage = cycle.solve_from_source_heat
                solve_kwargs = {
                    "Q_source": self._internal_heat[0],
                    "source_heat_is_external": False,
                }
            else:
                solve_mvr_stage = cycle.solve_from_mass_flow
                solve_kwargs = {
                    "m_dot": float(m_dot_in or 0.0),
                    "dT_superheat": 0.0,
                }
            solve_mvr_stage(
                T_evap=T_evap_mvr[j],
                T_cond=T_cond_mvr[j],
                dT_subcool=dT_subcool_mvr[j],
                eta_mvr_comp=eta_mvr_comp,
                eta_motor=eta_motor,
                fluid=mvr_fluid_all[j],
                process_split=self._process_split[j],
                **solve_kwargs,
            )

            if not cycle.solved:
                failed_work = abs(float(cycle.work or 0.0))
                failed_work = failed_work if np.isfinite(failed_work) else 1.0
                self._penalty = np.concatenate(
                    [
                        self._penalty,
                        np.array([max(failed_work, 1.0)], dtype=float),
                    ]
                )
                self._max_work += max(failed_work, 1.0)
                return self._max_work

            self._mvr_cycles.append(cycle)
            self._mvr_stage_mass_in[j] = float(cycle.source_m_dot or 0.0)
            heat_components = cycle.process_heat_components()
            self._mvr_stage_desuperheat[j] = heat_components["desuperheat"]
            self._mvr_stage_latent_heat[j] = heat_components["latent"]
            self._mvr_stage_subcool_heat[j] = heat_components["subcool"]
            self._mvr_stage_heat[j] = heat_components["total"]
            m_dot_out = float(cycle.process_m_dot_out or 0.0)
            self._mvr_stage_mass_out[j] = m_dot_out
            m_dot_in = m_dot_out

        work = sum(cycle.work for cycle in self.subcycles)
        if not np.isfinite(float(work)) or float(work) < 0.0:
            failed_work = abs(float(work))
            failed_work = failed_work if np.isfinite(failed_work) else 1.0
            self._max_work = max(failed_work, 1.0)
            return self._max_work
        self._solved = True
        return self.work

    def build_stream_collection(
        self,
        *,
        include_cond: bool = True,
        include_evap: bool = True,
        is_process_stream: bool = False,
        dtcont: float = 0.0,
        dt_diff_max: float = 0.5,
        include_internal: bool = False,
    ) -> StreamCollection:
        """Build external HPR streams, excluding internal cascade heat by default."""
        self._require_solution()
        self._dtcont = dtcont
        self._dt_diff_max = dt_diff_max
        streams = StreamCollection()
        for cycle in self._vc_cycles:
            streams += cycle.build_stream_collection(
                include_cond=include_cond,
                include_evap=include_evap,
                is_process_stream=is_process_stream,
                dtcont=dtcont,
                dt_diff_max=dt_diff_max,
            )
        if include_cond:
            streams += self._build_mvr_process_streams(dtcont=dtcont)
        if include_internal and include_evap and self._internal_heat.size:
            heat_flow = self._internal_heat[0]
            if heat_flow > 0.0:
                streams.add(
                    Stream(
                        name="VC_to_MVR_source",
                        t_supply=self._T_evap_mvr[0],
                        t_target=self._T_evap_mvr[0] + 0.01,
                        heat_flow=heat_flow,
                        dt_cont=dtcont,
                    )
                )
        for stream in streams:
            stream.is_process_stream = is_process_stream
        return streams

    @classmethod
    def _derive_mvr_temperatures(
        cls,
        *,
        vc_cycle: VapourCompressionCycle,
        dT_lift_mvr: np.ndarray,
        dt_cascade_hx: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        T_evap_mvr = np.empty_like(dT_lift_mvr, dtype=float)
        T_cond_mvr = np.empty_like(dT_lift_mvr, dtype=float)
        T_evap_mvr[0] = vc_cycle.Ts[4] - 273.15 - dt_cascade_hx
        for j, lift in enumerate(dT_lift_mvr):
            T_cond_mvr[j] = T_evap_mvr[j] + lift
            if j + 1 < dT_lift_mvr.size:
                T_evap_mvr[j + 1] = T_cond_mvr[j]
        return T_evap_mvr, T_cond_mvr

    def _build_mvr_process_streams(self, *, dtcont: float) -> StreamCollection:
        streams = StreamCollection()
        for j, cycle in enumerate(self._mvr_cycles):
            stage_streams = cycle.build_stream_collection(
                include_cond=True,
                include_evap=False,
                dtcont=dtcont,
            )
            self._set_mvr_stream_stage_names(stage_streams, stage_index=j + 1)
            streams += stage_streams
        return streams

    @staticmethod
    def _set_mvr_stream_stage_names(
        streams: StreamCollection,
        *,
        stage_index: int,
    ) -> None:
        for stream in streams:
            stream.name = stream.name.rsplit("_H", 1)[0] + f"_H{stage_index}"

    @staticmethod
    def _normalise_stage_array(values, size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(size, arr.item(), dtype=float)
        if arr.size != size:
            raise ValueError(f"Expected {size} stage values, got {arr.size}.")
        return arr

    @staticmethod
    def _normalise_process_split(values, n_mvr: int) -> np.ndarray:
        if n_mvr == 1:
            return np.ones(1, dtype=float)
        if values is None:
            leading = np.zeros(n_mvr - 1, dtype=float)
        else:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == 1:
                leading = np.full(n_mvr - 1, arr.item(), dtype=float)
            elif arr.size == n_mvr - 1:
                leading = arr
            elif arr.size == n_mvr:
                leading = arr[:-1]
            else:
                raise ValueError(
                    f"Expected {n_mvr - 1} MVR process split values, got {arr.size}."
                )
        return np.concatenate([leading, np.ones(1, dtype=float)])

    @staticmethod
    def _normalise_fluid_list(values, size: int) -> list[str]:
        if isinstance(values, str):
            return [values] * size
        values = list(values)
        if not values:
            return ["Water"] * size
        if len(values) == 1:
            return values * size
        if len(values) < size:
            return values + [values[-1]] * (size - len(values))
        return values[:size]

    def _require_solution(self) -> None:
        if not self._solved:
            raise RuntimeError("Solve the cycle before accessing results.")

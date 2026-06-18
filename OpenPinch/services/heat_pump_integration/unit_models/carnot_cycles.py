"""Carnot-family HPR backend classes."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from ....classes.stream_collection import StreamCollection
from ....lib.config import tol
from ..common.encoding import require_stage_duty_allocation
from ..common.shared import (
    calc_carnot_heat_engine_eta,
    calc_carnot_heat_pump_cop,
    compute_entropic_mean_temperature,
    get_carnot_hpr_cycle_streams,
)

__all__ = ["CascadeCarnotCycle", "ParallelCarnotCycles"]


class ParallelCarnotCycles:
    """Parallel simple Carnot heat-pump, heat-engine, and recovery stages."""

    def __init__(self):
        self._solved = False
        self._T_cond = np.empty(0, dtype=float)
        self._T_evap = np.empty(0, dtype=float)
        self._Q_cond = np.empty(0, dtype=float)
        self._Q_evap = np.empty(0, dtype=float)
        self._Q_cond_he = np.empty(0, dtype=float)
        self._Q_evap_he = np.empty(0, dtype=float)
        self._w_hpr = np.empty(0, dtype=float)
        self._w_he = np.empty(0, dtype=float)
        self._heat_recovery = 0.0
        self._penalty = np.empty(0, dtype=float)
        self._args: Any = None

    @property
    def solved(self) -> bool:
        return self._solved

    @property
    def T_cond(self) -> np.ndarray:
        return self._T_cond

    @property
    def T_evap(self) -> np.ndarray:
        return self._T_evap

    @property
    def Q_heat(self) -> np.ndarray:
        return self._Q_cond

    @property
    def Q_cool(self) -> np.ndarray:
        return self._Q_evap

    @property
    def Q_cond(self) -> np.ndarray:
        return self._Q_cond

    @property
    def Q_evap(self) -> np.ndarray:
        return self._Q_evap

    @property
    def Q_cond_he(self) -> np.ndarray:
        return self._Q_cond_he

    @property
    def Q_evap_he(self) -> np.ndarray:
        return self._Q_evap_he

    @property
    def w_hpr(self) -> np.ndarray:
        return self._w_hpr

    @property
    def w_he(self) -> np.ndarray:
        return self._w_he

    @property
    def heat_recovery(self) -> float:
        return self._heat_recovery

    @property
    def work(self) -> float:
        return float(self._w_hpr.sum() - self._w_he.sum())

    @property
    def work_arr(self) -> np.ndarray:
        return self._w_hpr - self._w_he

    @property
    def penalty(self) -> np.ndarray:
        return self._penalty

    @property
    def COP_h(self) -> float:
        w_hpr = float(self._w_hpr.sum())
        return float(self._Q_cond.sum() / w_hpr) if w_hpr > tol else 0.0

    @property
    def eta_he(self) -> float:
        q_cond = float(self._Q_cond.sum())
        return float(self._w_he.sum() / (q_cond + 1e-6)) if q_cond != 0 else 0.0

    def solve(
        self,
        *,
        T_cond: np.ndarray,
        T_evap: np.ndarray,
        Q_heat_base: float,
        x_heat_split: np.ndarray,
        Q_heat_available: np.ndarray,
        Q_cool_available: np.ndarray,
        eta_ii_hpr_carnot: float,
        eta_ii_he_carnot: float,
        args: Any,
    ) -> float:
        self._solved = False
        self._args = args
        self._T_cond = T_cond
        self._T_evap = T_evap
        allocation = require_stage_duty_allocation(
            Q_base=Q_heat_base,
            x_split=x_heat_split,
            Q_available=Q_heat_available,
            duty_name="heat",
        )
        Q_cond = allocation.Q_model
        Q_cool_available = np.maximum(Q_cool_available, 0.0)
        if self._T_cond.size != self._T_evap.size or Q_cond.size != self._T_cond.size:
            raise ValueError("Parallel Carnot stage arrays must have matching sizes.")
        if Q_cool_available.size != self._T_evap.size:
            raise ValueError("Q_cool_available must match Carnot stage count.")

        self._penalty = allocation.Q_excess
        self._Q_cond_he = np.zeros_like(Q_cond)
        self._Q_evap_he = np.zeros_like(Q_cond)
        Qc_hx = np.zeros_like(Q_cond)
        Qe_hx = np.zeros_like(Q_cond)
        Qc_hpr = np.zeros_like(Q_cond)
        Qe_hpr = np.zeros_like(Q_cond)
        self._w_hpr = np.zeros_like(Q_cond)
        self._w_he = np.zeros_like(Q_cond)

        T_diff = self._T_cond - self._T_evap
        T_cond_abs = self._T_cond + 273.15
        T_evap_abs = self._T_evap + 273.15

        is_hp = T_diff >= tol
        if np.any(is_hp):
            cop_hp = calc_carnot_heat_pump_cop(
                T_cond_abs[is_hp],
                T_evap_abs[is_hp],
                eta_ii_hpr_carnot,
            )
            Qc_hpr[is_hp] = Q_cond[is_hp]
            self._w_hpr[is_hp] = Qc_hpr[is_hp] / cop_hp
            Qe_hpr[is_hp] = Qc_hpr[is_hp] - self._w_hpr[is_hp]

        is_he = (T_diff <= -tol) & (eta_ii_he_carnot >= tol)
        if np.any(is_he):
            eff_he = calc_carnot_heat_engine_eta(
                T_evap_abs[is_he],
                T_cond_abs[is_he],
                eta_ii_he_carnot,
            )
            self._Q_cond_he[is_he] = Q_cond[is_he]
            self._w_he[is_he] = self._Q_cond_he[is_he] * eff_he / (1 - eff_he)
            self._Q_evap_he[is_he] = self._Q_cond_he[is_he] + self._w_he[is_he]

        is_hx = ((T_diff > -tol) | (eta_ii_he_carnot < tol)) & (T_diff < tol)
        if np.any(is_hx):
            Qc_hx[is_hx] = Q_cond[is_hx]
            Qe_hx[is_hx] = Qc_hx[is_hx]

        Q_allocated = 0.0
        for i in np.argsort(-self._T_evap, kind="stable"):
            Q_stage = self._Q_evap_he[i] + Qe_hx[i] + Qe_hpr[i]
            if Q_stage < tol:
                self._Q_cond_he[i] = 0.0
                self._Q_evap_he[i] = 0.0
                Qc_hx[i] = 0.0
                Qe_hx[i] = 0.0
                Qc_hpr[i] = 0.0
                Qe_hpr[i] = 0.0
                self._w_he[i] = 0.0
                self._w_hpr[i] = 0.0
                continue

            Q_available = Q_cool_available[i] - Q_allocated
            scale = 0.0 if Q_available < tol else min(Q_available / Q_stage, 1.0)
            if scale < 1.0:
                self._penalty = np.concatenate(
                    [self._penalty, np.array([Q_stage - max(Q_available, 0.0)])]
                )
            self._Q_cond_he[i] *= scale
            self._Q_evap_he[i] *= scale
            Qc_hx[i] *= scale
            Qe_hx[i] *= scale
            Qc_hpr[i] *= scale
            Qe_hpr[i] *= scale
            self._w_he[i] *= scale
            self._w_hpr[i] *= scale
            Q_allocated += self._Q_evap_he[i] + Qe_hx[i] + Qe_hpr[i]

        self._Q_cond = self._Q_cond_he + Qc_hx + Qc_hpr
        self._Q_evap = self._Q_evap_he + Qe_hx + Qe_hpr
        self._heat_recovery = float(Qc_hx.sum())
        self._solved = bool(
            np.isclose(
                self._Q_cond.sum() + self._w_he.sum(),
                self._Q_evap.sum() + self._w_hpr.sum(),
                atol=tol,
            )
        )
        return self.work

    def build_stream_collection(self) -> StreamCollection:
        if self._args is None:
            return StreamCollection()
        return get_carnot_hpr_cycle_streams(
            self._T_cond,
            self._Q_cond,
            self._T_evap,
            self._Q_evap,
            self._args,
        )


class CascadeCarnotCycle:
    """Cascade Carnot backend with shared solve-state properties."""

    def __init__(self):
        self._solved = False
        self._T_cond = np.empty(0, dtype=float)
        self._T_evap = np.empty(0, dtype=float)
        self._Q_cond = np.empty(0, dtype=float)
        self._Q_evap = np.empty(0, dtype=float)
        self._Q_cond_he = np.empty(0, dtype=float)
        self._Q_evap_he = np.empty(0, dtype=float)
        self._w_hpr = 0.0
        self._w_he = 0.0
        self._cop = 1.0
        self._penalty = np.empty(0, dtype=float)
        self._args: Any = None

    @property
    def solved(self) -> bool:
        return self._solved

    @property
    def T_cond(self) -> np.ndarray:
        return self._T_cond

    @property
    def T_evap(self) -> np.ndarray:
        return self._T_evap

    @property
    def Q_heat(self) -> np.ndarray:
        return self._Q_cond

    @property
    def Q_cool(self) -> np.ndarray:
        return self._Q_evap

    @property
    def Q_cond(self) -> np.ndarray:
        return self._Q_cond

    @property
    def Q_evap(self) -> np.ndarray:
        return self._Q_evap

    @property
    def Q_cond_he(self) -> np.ndarray:
        return self._Q_cond_he

    @property
    def Q_evap_he(self) -> np.ndarray:
        return self._Q_evap_he

    @property
    def w_hpr(self) -> float:
        return self._w_hpr

    @property
    def w_he(self) -> float:
        return self._w_he

    @property
    def heat_recovery(self) -> np.ndarray:
        return self._Q_cond_he

    @property
    def work(self) -> float:
        return float(self._w_hpr - self._w_he)

    @property
    def work_arr(self) -> np.ndarray:
        return np.array([self.work], dtype=float)

    @property
    def penalty(self) -> np.ndarray:
        return self._penalty

    @property
    def COP_h(self) -> float:
        return self._cop

    def solve(
        self,
        *,
        T_cond: np.ndarray,
        T_evap: np.ndarray,
        Q_heat_base: float,
        x_heat_split: np.ndarray,
        Q_heat_available: np.ndarray,
        Q_cool_available: np.ndarray,
        eta_ii_hpr_carnot: float,
        eta_ii_he_carnot: float,
        args: Any,
    ) -> float:
        self._solved = False
        self._args = args
        self._T_cond = T_cond
        self._T_evap = T_evap
        heat_allocation = require_stage_duty_allocation(
            Q_base=Q_heat_base,
            x_split=x_heat_split,
            Q_available=Q_heat_available,
            duty_name="heat",
        )
        Q_cond = heat_allocation.Q_model
        Q_evap = np.maximum(Q_cool_available, 0.0)
        if Q_cond.size != self._T_cond.size or Q_evap.size != self._T_evap.size:
            raise ValueError("Cascade Carnot pool sizes are inconsistent.")
        self._penalty = heat_allocation.Q_excess

        T_diff = np.subtract.outer(self._T_cond, self._T_evap)
        is_hp = T_diff > tol
        self._Q_cond_he, self._Q_evap_he = self._get_heat_engine_and_recovery_duty(
            is_on=~is_hp,
            Qc_pool=Q_cond,
            Qe_pool=Q_evap,
            eta_ii_he_carnot=eta_ii_he_carnot,
        )

        if np.any(~is_hp):

            def fun(x: float) -> float:
                Qc_hpr, Qe_hpr = self._get_heat_pump_duty(
                    is_on=is_hp,
                    Qc_pool=Q_cond - self._Q_cond_he * x,
                    Qe_pool=Q_evap - self._Q_evap_he * x,
                    eta_ii_hpr_carnot=eta_ii_hpr_carnot,
                )
                w_he = (self._Q_evap_he.sum() - self._Q_cond_he.sum()) * x
                w_hpr = Qc_hpr.sum() - Qe_hpr.sum()
                return w_hpr - w_he

            res = minimize_scalar(fun=fun, bounds=(0, 1), method="bounded")
            self._Q_cond_he *= res.x
            self._Q_evap_he *= res.x

        Qc_hpr, Qe_hpr = self._get_heat_pump_duty(
            is_on=is_hp,
            Qc_pool=Q_cond - self._Q_cond_he,
            Qe_pool=Q_evap - self._Q_evap_he,
            eta_ii_hpr_carnot=eta_ii_hpr_carnot,
        )
        self._w_he = float(self._Q_evap_he.sum() - self._Q_cond_he.sum())
        self._w_hpr = float(Qc_hpr.sum() - Qe_hpr.sum())
        self._cop = float(Qc_hpr.sum() / self._w_hpr) if self._w_hpr > 0 else 1.0
        self._Q_cond = Qc_hpr + self._Q_cond_he
        self._Q_evap = Qe_hpr + self._Q_evap_he
        self._solved = bool(
            np.isclose(
                self._Q_cond_he.sum() + Qc_hpr.sum() + self._w_he,
                self._Q_evap_he.sum() + Qe_hpr.sum() + self._w_hpr,
                atol=tol,
            )
        )
        return self.work

    def _get_unique_idx(self, is_on: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        i_c, i_e = np.nonzero(is_on)
        return np.unique(i_c), np.unique(i_e)

    def _get_heat_engine_and_recovery_duty(
        self,
        *,
        is_on: np.ndarray,
        Qc_pool: np.ndarray,
        Qe_pool: np.ndarray,
        eta_ii_he_carnot: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Qc_he = np.zeros_like(Qc_pool)
        Qe_he = np.zeros_like(Qe_pool)
        if np.any(is_on):
            i_c, i_e = self._get_unique_idx(is_on)
            Qc_pool_sum, Qe_pool_sum = Qc_pool[i_c].sum(), Qe_pool[i_e].sum()
            if Qc_pool_sum * Qe_pool_sum > tol:
                if 1 > eta_ii_he_carnot > 0:
                    T_h = compute_entropic_mean_temperature(
                        self._T_evap[i_e],
                        Qe_pool[i_e],
                    )
                    T_l = compute_entropic_mean_temperature(
                        self._T_cond[i_c],
                        Qc_pool[i_c],
                    )
                    eta_he = calc_carnot_heat_engine_eta(T_h, T_l, eta_ii_he_carnot)
                    Qe_used = min(Qe_pool_sum, Qc_pool_sum / max(1.0 - eta_he, tol))
                    Qc_used = Qe_used * (1 - eta_he)
                else:
                    Qe_used = Qc_used = min(Qe_pool_sum, Qc_pool_sum)
                Qc_he[i_c] = Qc_pool[i_c] * (Qc_used / Qc_pool_sum)
                Qe_he[i_e] = Qe_pool[i_e] * (Qe_used / Qe_pool_sum)
        return Qc_he, Qe_he

    def _get_heat_pump_duty(
        self,
        *,
        is_on: np.ndarray,
        Qc_pool: np.ndarray,
        Qe_pool: np.ndarray,
        eta_ii_hpr_carnot: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Qc_hpr = np.zeros_like(Qc_pool)
        Qe_hpr = np.zeros_like(Qe_pool)
        if np.any(is_on):
            i_c, i_e = self._get_unique_idx(is_on)
            Qc_pool_sum, Qe_pool_sum = Qc_pool[i_c].sum(), Qe_pool[i_e].sum()
            if (Qc_pool_sum * Qe_pool_sum > tol) and (1 > eta_ii_hpr_carnot >= 0):
                T_h = compute_entropic_mean_temperature(
                    self._T_cond[i_c],
                    Qc_pool[i_c],
                )
                T_l = compute_entropic_mean_temperature(
                    self._T_evap[i_e],
                    Qe_pool[i_e],
                )
                if T_h > T_l and eta_ii_hpr_carnot > 0.0:
                    cop = calc_carnot_heat_pump_cop(T_h, T_l, eta_ii_hpr_carnot)
                    Qe_used = min(Qe_pool_sum, Qc_pool_sum * (cop - 1.0) / cop)
                    w_hpr = Qe_used / (cop - 1.0)
                    Qc_used = Qe_used + w_hpr
                    Qc_hpr[i_c] = Qc_pool[i_c] * (Qc_used / Qc_pool_sum)
                    Qe_hpr[i_e] = Qe_pool[i_e] * (Qe_used / Qe_pool_sum)
        return Qc_hpr, Qe_hpr

    def build_stream_collection(self) -> StreamCollection:
        if self._args is None:
            return StreamCollection()
        return get_carnot_hpr_cycle_streams(
            self._T_cond,
            self._Q_cond,
            self._T_evap,
            self._Q_evap,
            self._args,
        )

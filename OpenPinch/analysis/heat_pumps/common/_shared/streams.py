"""Private stream helpers for ``common.shared``."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .....contracts.hpr import HeatPumpTargetInputs
from .....domain.configuration import tol
from .....domain.stream import Stream
from .....domain.stream_collection import StreamCollection


def get_Q_vals_at_T_hpr_from_bckgrd_profile(
    T_hpr: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *,
    is_cond: bool = True,
) -> np.ndarray:
    """Read stage duties from a background profile at proposed HPR temperatures."""
    H_less_origin = np.interp(T_hpr, T_vals[::-1], H_vals[::-1])
    H = (
        np.concatenate((H_less_origin, np.array([0.0])))
        if is_cond
        else np.concatenate((np.array([0.0]), H_less_origin))
    )
    temp = np.roll(H, -1)
    temp[-1] = 0
    Q = H - temp
    Q_hx = Q[:-1]
    return np.where(Q_hx > 0.0, Q_hx, 0.0)


def get_carnot_hpr_cycle_streams(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    args: HeatPumpTargetInputs,
) -> StreamCollection:
    """Build one combined HPR utility stream collection for Carnot-cycle summaries."""
    dt_phase_change = float(getattr(args, "dt_phase_change", 1.0))
    return _build_latent_streams(
        T_cond, dt_phase_change, Q_cond, is_hot=True
    ) + _build_latent_streams(T_evap, dt_phase_change, Q_evap, is_hot=False)


def get_ambient_air_stream(
    Q_amb_hot: float = 0.0,
    Q_amb_cold: float = 0.0,
    args: HeatPumpTargetInputs = None,
) -> StreamCollection:
    """Build ambient-air exchange streams implied by the solved HPR result."""
    sc = StreamCollection()
    if Q_amb_hot > tol:
        sc += _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_hot]),
            dt_cont=args.dt_env_cont,
            is_hot=True,
            is_process_stream=True,
            prefix="AIR",
        )
    if Q_amb_cold > tol:
        sc += _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_cold]),
            dt_cont=args.dt_env_cont,
            is_hot=False,
            is_process_stream=True,
            prefix="AIR",
        )
    return sc


def _build_latent_streams(
    T_ls: np.ndarray,
    dT_phase_change: float,
    Q_ls: np.ndarray,
    *,
    dt_cont: float = 0.0,
    is_hot: bool = True,
    is_process_stream: bool = False,
    prefix: str = "HP",
) -> StreamCollection:
    if len(T_ls) > 1:
        T_ls, Q_ls = _get_carnot_hpr_cycle_cascade_profile(
            T_ls.tolist(), Q_ls.tolist(), dT_phase_change, is_hot
        )

    sc = StreamCollection()
    for i in range(len(Q_ls)):
        sc.add(
            Stream(
                name=f"{prefix}_H{i + 1}" if is_hot else f"{prefix}_C{i + 1}",
                t_supply=T_ls[i] if is_hot else T_ls[i] - dT_phase_change,
                t_target=T_ls[i] - dT_phase_change if is_hot else T_ls[i],
                heat_flow=Q_ls[i],
                dt_cont=dt_cont,
                is_process_stream=is_process_stream,
            )
        )
    return sc


def _get_carnot_hpr_cycle_cascade_profile(
    T_hpr: list,
    Q_hpr: list,
    dT_phase_change: float,
    is_hot: bool,
    i: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    inc = 1 if is_hot else -1
    if i is None:
        i = 0 if is_hot else len(T_hpr)

    i_range = range(i, len(T_hpr) - 1) if is_hot else reversed(range(1, i))
    for i in i_range:
        if abs(T_hpr[i] - T_hpr[i + inc]) < dT_phase_change:
            T_hpr.pop(i + inc)
            Q_hpr[i] += Q_hpr[i + inc]
            Q_hpr.pop(i + inc)
            T_hpr, Q_hpr = _get_carnot_hpr_cycle_cascade_profile(
                T_hpr,
                Q_hpr,
                dT_phase_change,
                is_hot,
                i,
            )
            break

    return T_hpr, Q_hpr

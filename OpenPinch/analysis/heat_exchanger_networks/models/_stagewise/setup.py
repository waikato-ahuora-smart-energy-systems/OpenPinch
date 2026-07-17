"""StageWise preprocessing owned by the StageWise model."""

from __future__ import annotations

import numpy as np


def set_preprocessing(owner) -> None:
    """Pre-process SynHEAT superstructure parameters for all states."""

    owner.S = owner.stages
    owner.K = owner.S + 1
    owner.I = owner.f_h_period.shape[1]
    owner.J = owner.f_c_period.shape[1]

    owner.Qtot_sh_period = np.array(
        [
            [
                owner._parent_profile_duty(
                    "hot",
                    n,
                    i,
                    owner.T_h_in_period[n][i],
                    owner.T_h_out_period[n][i],
                    owner.f_h_period[n][i],
                )
                for i in range(owner.I)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.Qtot_sc_period = np.array(
        [
            [
                owner._parent_profile_duty(
                    "cold",
                    n,
                    j,
                    owner.T_c_in_period[n][j],
                    owner.T_c_out_period[n][j],
                    owner.f_c_period[n][j],
                )
                for j in range(owner.J)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.Qtot_sh = np.max(owner.Qtot_sh_period, axis=0)
    owner.Qtot_sc = np.max(owner.Qtot_sc_period, axis=0)

    owner.U_r_period = np.array(
        [
            [
                [
                    1 / (1 / owner.htc_h_period[n][i] + 1 / owner.htc_c_period[n][j])
                    for j in range(owner.J)
                ]
                for i in range(owner.I)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.U_hu_period = np.array(
        [
            [
                1 / (1 / owner.htc_hu_period[n][0] + 1 / owner.htc_c_period[n][j])
                for j in range(owner.J)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.U_cu_period = np.array(
        [
            [
                1 / (1 / owner.htc_h_period[n][i] + 1 / owner.htc_cu_period[n][0])
                for i in range(owner.I)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.U_r = owner.U_r_period[0].copy()
    owner.U_hu = owner.U_hu_period[0].copy()
    owner.U_cu = owner.U_cu_period[0].copy()

    owner.Q_max_period = np.array(
        [
            [
                [
                    max(
                        owner._recovery_heat_upper_bound(
                            period_index=n,
                            hot_index=i,
                            cold_index=j,
                            hot_total_duty=owner.Qtot_sh_period[n][i],
                            cold_total_duty=owner.Qtot_sc_period[n][j],
                            hot_cp=owner.f_h_period[n][i],
                            cold_cp=owner.f_c_period[n][j],
                        ),
                        0.0,
                    )
                    for j in range(owner.J)
                ]
                for i in range(owner.I)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.Q_max = np.max(owner.Q_max_period, axis=0)
    owner.M_ij_period = np.array(
        [
            [
                [
                    max(
                        abs(owner.T_h_in_period[n][i] - owner.T_c_in_period[n][j]),
                        abs(owner.T_h_in_period[n][i] - owner.T_c_out_period[n][j]),
                        abs(owner.T_h_out_period[n][i] - owner.T_c_in_period[n][j]),
                        abs(owner.T_h_out_period[n][i] - owner.T_c_out_period[n][j]),
                    )
                    + owner._recovery_approach_temperature(i, j, n)
                    for j in range(owner.J)
                ]
                for i in range(owner.I)
            ]
            for n in range(owner.N_periods)
        ],
        dtype=float,
    )
    owner.M_ij = np.max(owner.M_ij_period, axis=0)

    owner.z_feasible = [
        [
            [
                (
                    1
                    if max(owner.Q_max_period[n][i][j] for n in range(owner.N_periods))
                    > owner.tol
                    else 0
                )
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
        for i in range(owner.I)
    ]
    owner.z_hu_feasible = [1 for _j in range(owner.J)]
    owner.z_cu_feasible = [1 for _i in range(owner.I)]

"""Superstructure builders shared by the PDM and StageWise models."""

from __future__ import annotations

from ...indexing import build_index_grid


def create_utility_duty_grids(owner) -> None:
    """Create the per-period cold- and hot-utility duty grids.

    Each (stream, utility) pair is its own heater or cooler:
    ``Q_c_by_period[period][hot][utility]`` and
    ``Q_h_by_period[period][cold][utility]``.
    """

    owner.Q_c_by_period = build_index_grid(
        lambda n, i, u: (
            owner.m.Var(
                value=0,
                ub=owner.Qtot_sh_period[n][i],
                lb=0.0,
                name=f"Q_H{i}_to_CU{u}_period{n}",
            )
            if owner.z_cu_allowed[i][u] > 0
            else owner.m.Param(value=0, name=f"Q_H{i}_to_CU{u}_period{n}")
        ),
        (owner.N_periods, owner.I, owner.NCU),
    )
    owner.Q_h_by_period = build_index_grid(
        lambda n, j, u: (
            owner.m.Var(
                value=0,
                ub=owner.Qtot_sc_period[n][j],
                lb=0.0,
                name=f"Q_HU{u}_to_C{j}_period{n}",
            )
            if owner.z_hu_allowed[j][u] > 0
            else owner.m.Param(value=0, name=f"Q_HU{u}_to_C{j}_period{n}")
        ),
        (owner.N_periods, owner.J, owner.NHU),
    )


def create_match_binaries(owner) -> None:
    """Create the recovery and utility match binaries (or relaxed params)."""

    if owner.integers:
        owner.z = [
            [
                [
                    (
                        owner.m.Var(
                            value=1,
                            ub=1,
                            lb=0,
                            integer=True,
                            name=f"z_H{i}_to_C{j}_at_S{k}",
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        owner.z_cu = build_index_grid(
            lambda i, u: (
                owner.m.Var(
                    value=1,
                    ub=1,
                    lb=0,
                    integer=True,
                    name=f"z_H{i}_to_CU{u}",
                )
                if owner.z_cu_allowed[i][u] > 0
                else owner.m.Param(value=0, name=f"z_H{i}_to_CU{u}")
            ),
            (owner.I, owner.NCU),
        )
        owner.z_hu = build_index_grid(
            lambda j, u: (
                owner.m.Var(
                    value=1,
                    ub=1,
                    lb=0,
                    integer=True,
                    name=f"z_HU{u}_to_C{j}",
                )
                if owner.z_hu_allowed[j][u] > 0
                else owner.m.Param(value=0, name=f"z_HU{u}_to_C{j}")
            ),
            (owner.J, owner.NHU),
        )
    else:
        owner.z = [
            [
                [
                    (
                        owner.m.Param(value=1, name=f"z_H{i}_to_C{j}_at_S{k}")
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        owner.z_hu = build_index_grid(
            lambda j, u: owner.m.Param(
                value=1 if owner.z_hu_allowed[j][u] > 0 else 0,
                name=f"z_HU{u}_to_C{j}",
            ),
            (owner.J, owner.NHU),
        )
        owner.z_cu = build_index_grid(
            lambda i, u: owner.m.Param(
                value=1 if owner.z_cu_allowed[i][u] > 0 else 0,
                name=f"z_H{i}_to_CU{u}",
            ),
            (owner.I, owner.NCU),
        )

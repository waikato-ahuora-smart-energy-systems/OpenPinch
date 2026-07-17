"""Alpha and dQ/dA equations for base HEN models."""

from __future__ import annotations

from typing import Any

from ...indexing import build_index_grid
from ...solver import backend


def get_alpha_values(model) -> list:
    """Calculate source alpha flow-on values in a post-optimisation solve."""

    if model.alpha != []:
        return model.alpha

    solver_model = backend.create_gekko_model(remote=False)
    solver_model.options.IMODE = 1
    solver_model.options.SOLVER = 1
    model.set_alpha_dqda_equations(m=solver_model, postoptimisation=True)
    try:
        with backend.suppress_gekko_numpy_array_copy_deprecation():
            solver_model.solve(disp=False)
    except Exception:
        pass
    return model.alpha


def set_alpha_dqda_equations(
    model,
    *,
    m: Any | None = None,
    postoptimisation: bool = False,
) -> None:
    """Move the source alpha and dQ/dA equations without changing formulas."""

    if postoptimisation:
        if m is None:
            raise ValueError("postoptimisation alpha equations require a model.")
    else:
        m = model.m
    recovery_grid_shape = (model.I, model.J, model.S)

    def postoptimisation_denominator(i: int, j: int, k: int) -> float:
        return model.T_h[i][k][0] - model.T_c[j][k + 1][0]

    def model_denominator(i: int, j: int, k: int) -> Any:
        return (model.T_h[i][k] - model.T_c[j][k + 1] - 1) * model.z[i][j][k] + 1

    if postoptimisation:
        if model.non_isothermal_model:
            model.P_h = build_index_grid(
                lambda i, j, k: (
                    (model.T_h[i][k][0] - model.T_h_out_x[i][j][k][0])
                    / postoptimisation_denominator(i, j, k)
                    if model.T_h[i][k][0] > model.T_c[j][k + 1][0]
                    else 0.0
                ),
                recovery_grid_shape,
            )
            model.P_c = build_index_grid(
                lambda i, j, k: (
                    (model.T_c_out_y[j][i][k][0] - model.T_c[j][k + 1][0])
                    / postoptimisation_denominator(i, j, k)
                    if model.T_h[i][k][0] > model.T_c[j][k + 1][0]
                    else 0.0
                ),
                recovery_grid_shape,
            )
        else:
            model.P_h = build_index_grid(
                lambda i, j, k: (
                    (model.T_h[i][k][0] - model.T_h[i][k + 1][0])
                    / postoptimisation_denominator(i, j, k)
                    if model.T_h[i][k][0] > model.T_c[j][k + 1][0]
                    else 0.0
                ),
                recovery_grid_shape,
            )
            model.P_c = build_index_grid(
                lambda i, j, k: (
                    (model.T_c[j][k][0] - model.T_c[j][k + 1][0])
                    / postoptimisation_denominator(i, j, k)
                    if model.T_h[i][k][0] > model.T_c[j][k + 1][0]
                    else 0.0
                ),
                recovery_grid_shape,
            )

        model.Sum_Qr_is = build_index_grid(
            lambda i, k: [sum(model.Q_r[i][j][k][0] for j in range(model.J))],
            (model.I, model.S),
        )
        model.Sum_Qr_js = build_index_grid(
            lambda j, k: [sum(model.Q_r[i][j][k][0] for i in range(model.I))],
            (model.J, model.S),
        )
        model.beta_h = build_index_grid(
            lambda i, j, k: (
                model.Q_r[i][j][k][0] / model.Sum_Qr_is[i][k][0]
                if model.Sum_Qr_is[i][k][0] > 0
                else 0.0
            ),
            recovery_grid_shape,
        )
        model.beta_c = build_index_grid(
            lambda i, j, k: (
                model.Q_r[i][j][k][0] / model.Sum_Qr_js[j][k][0]
                if model.Sum_Qr_js[j][k][0] > 0
                else 0.0
            ),
            recovery_grid_shape,
        )
        model.z_i = build_index_grid(
            lambda j, k: (
                sum(model.z[i][j][k][0] for i in range(model.I))
                / (sum(model.z[i][j][k][0] for i in range(model.I)) + 1e-9)
            ),
            (model.J, model.S),
        )
        model.z_j = build_index_grid(
            lambda i, k: (
                sum(model.z[i][j][k][0] for j in range(model.J))
                / (sum(model.z[i][j][k][0] for j in range(model.J)) + 1e-9)
            ),
            (model.I, model.S),
        )
    else:
        if model.non_isothermal_model:
            model.P_h = build_index_grid(
                lambda i, j, k: m.Intermediate(
                    (model.T_h[i][k] - model.T_h_out_x[i][j][k])
                    * model.z[i][j][k]
                    / model_denominator(i, j, k)
                ),
                recovery_grid_shape,
            )
            model.P_c = build_index_grid(
                lambda i, j, k: m.Intermediate(
                    (model.T_c_out_y[j][i][k] - model.T_c[j][k + 1])
                    * model.z[i][j][k]
                    / model_denominator(i, j, k)
                ),
                recovery_grid_shape,
            )
        else:
            model.P_h = build_index_grid(
                lambda i, j, k: m.Intermediate(
                    (model.T_h[i][k] - model.T_h[i][k + 1])
                    * model.z[i][j][k]
                    / model_denominator(i, j, k)
                ),
                recovery_grid_shape,
            )
            model.P_c = build_index_grid(
                lambda i, j, k: m.Intermediate(
                    (model.T_c[j][k] - model.T_c[j][k + 1])
                    * model.z[i][j][k]
                    / model_denominator(i, j, k)
                ),
                recovery_grid_shape,
            )

        model.Sum_Qr_j = build_index_grid(
            lambda i, k: m.Intermediate(
                sum(model.Q_r[i][j][k] for j in range(model.J))
            ),
            (model.I, model.S),
        )
        model.Sum_Qr_i = build_index_grid(
            lambda j, k: m.Intermediate(
                sum(model.Q_r[i][j][k] for i in range(model.I))
            ),
            (model.J, model.S),
        )
        model.beta_h = build_index_grid(
            lambda i, j, k: m.Intermediate(
                model.Q_r[i][j][k] / (model.Sum_Qr_j[i][k] + 1 - model.z[i][j][k])
            ),
            recovery_grid_shape,
        )
        model.beta_c = build_index_grid(
            lambda i, j, k: m.Intermediate(
                model.Q_r[i][j][k] / (model.Sum_Qr_i[j][k] + 1 - model.z[i][j][k])
            ),
            recovery_grid_shape,
        )
        model.z_i = build_index_grid(
            lambda j, k: m.Intermediate(
                sum(model.z[i][j][k] for i in range(model.I))
                / (sum(model.z[i][j][k] for i in range(model.I)) + 1e-9)
            ),
            (model.J, model.S),
        )
        model.z_j = build_index_grid(
            lambda i, k: m.Intermediate(
                sum(model.z[i][j][k] for j in range(model.J))
                / (sum(model.z[i][j][k] for j in range(model.J)) + 1e-9)
            ),
            (model.I, model.S),
        )

    model.alpha = build_index_grid(
        lambda i, j, k: m.Var(
            value=0.0,
            ub=1.0,
            lb=-1.0,
            name=f"alpha_H{i}_to_C{j}_at_S{k}",
        ),
        recovery_grid_shape,
    )
    model.gamma_h = build_index_grid(
        lambda i, j, k: m.Var(
            value=0.5,
            ub=1.0,
            lb=-1.0,
            name=f"gamma_h_H{i}_to_C{j}_at_S{k}",
        ),
        recovery_grid_shape,
    )
    model.gamma_c = build_index_grid(
        lambda i, j, k: m.Var(
            value=0.5,
            ub=1.0,
            lb=-1.0,
            name=f"gamma_c_H{i}_to_C{j}_at_S{k}",
        ),
        recovery_grid_shape,
    )

    model.gamma_h_eqn = []
    model.gamma_c_eqn = []
    for k in range(model.S):
        for j in range(model.J):
            for i in range(model.I):
                if k + 1 >= model.S:
                    model.gamma_h_eqn.append(
                        [m.Equation(model.gamma_h[i][j][k] == 0.0)]
                    )
                    model.gamma_c_eqn.append(
                        [
                            m.Equation(
                                model.gamma_c[i][j][k]
                                == sum(
                                    model.beta_c[i0][j][k - 1]
                                    * model.P_c[i0][j][k - 1]
                                    * model.alpha[i0][j][k - 1]
                                    for i0 in range(model.I)
                                )
                                + (1 - model.z_i[j][k - 1]) * model.gamma_c[i][j][k - 1]
                            )
                        ]
                    )
                elif k - 1 < 0:
                    model.gamma_h_eqn.append(
                        [
                            m.Equation(
                                model.gamma_h[i][j][k]
                                == sum(
                                    model.beta_h[i][j0][k + 1]
                                    * model.P_h[i][j0][k + 1]
                                    * model.alpha[i][j0][k + 1]
                                    for j0 in range(model.J)
                                )
                                + (1 - model.z_j[i][k + 1]) * model.gamma_h[i][j][k + 1]
                            )
                        ]
                    )
                    model.gamma_c_eqn.append(
                        [m.Equation(model.gamma_c[i][j][k] == 0.0)]
                    )
                else:
                    model.gamma_h_eqn.append(
                        [
                            m.Equation(
                                model.gamma_h[i][j][k]
                                == sum(
                                    model.beta_h[i][j0][k + 1]
                                    * model.P_h[i][j0][k + 1]
                                    * model.alpha[i][j0][k + 1]
                                    for j0 in range(model.J)
                                )
                                + (1 - model.z_j[i][k + 1]) * model.gamma_h[i][j][k + 1]
                            )
                        ]
                    )
                    model.gamma_c_eqn.append(
                        [
                            m.Equation(
                                model.gamma_c[i][j][k]
                                == sum(
                                    model.beta_c[i0][j][k - 1]
                                    * model.P_c[i0][j][k - 1]
                                    * model.alpha[i0][j][k - 1]
                                    for i0 in range(model.I)
                                )
                                + (1 - model.z_i[j][k - 1]) * model.gamma_c[i][j][k - 1]
                            )
                        ]
                    )

    model.alpha_eqn = [
        m.Equation(
            model.alpha[i][j][k]
            == (1 - 0.5 * (model.gamma_h[i][j][k] + model.gamma_c[i][j][k]))
        )
        for k in range(model.S)
        for j in range(model.J)
        for i in range(model.I)
        if postoptimisation or model.z_allowed[i][j][k] > 0
    ]
    if not postoptimisation:
        model.alpha_dQ_dA_eqn = [
            (
                m.Equation(
                    (
                        model.min_dqda * (model.T_h[i][k] - model.T_c[j][k + 1])
                        - model.alpha[i][j][k]
                        * model.theta_1[i][j][k]
                        * model.theta_2[i][j][k]
                        * model.U_r[i][j]
                    )
                    * model.z[i][j][k]
                    <= 0.0
                )
                if model.z_allowed[i][j][k] > 0
                else None
            )
            for k in range(model.S)
            for j in range(model.J)
            for i in range(model.I)
        ]

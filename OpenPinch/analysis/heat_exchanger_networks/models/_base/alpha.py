"""Alpha and dQ/dA equations for base HEN models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ...indexing import build_index_grid
from ...solver import backend

logger = logging.getLogger(__name__)


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
        # Callers index alpha[i][j][k][0] for benefit ranking, so keep the
        # unsolved initial values but make the failure visible.
        logger.warning(
            "Post-optimisation alpha solve failed for %s; alpha values are "
            "unsolved initial guesses and benefit ranking may be unreliable.",
            getattr(model, "name", "model"),
            exc_info=True,
        )
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
        _set_postoptimisation_flow_fractions(model)
    else:
        m = model.m
        _set_model_flow_fractions(model, m)

    _set_alpha_gamma_variables(model, m)
    _set_gamma_equations(model, m)
    model.alpha_eqn = [
        m.Equation(
            model.alpha[i][j][k]
            == (1 - 0.5 * (model.gamma_h[i][j][k] + model.gamma_c[i][j][k]))
        )
        for k, j, i in _recovery_indices(model)
        if postoptimisation or model.z_allowed[i][j][k] > 0
    ]
    if not postoptimisation:
        _set_dqda_equations(model, m)


def _recovery_grid_shape(model) -> tuple[int, int, int]:
    return (model.I, model.J, model.S)


def _recovery_indices(model):
    """Yield ``(k, j, i)`` in the stage-major order used for equation lists."""

    for k in range(model.S):
        for j in range(model.J):
            for i in range(model.I):
                yield k, j, i


def _solved_value(value: Any) -> Any:
    return value[0]


def _model_value(value: Any) -> Any:
    return value


def _unchanged(value: Any) -> Any:
    return value


def _temperature_drop_functions(
    model, value: Callable[[Any], Any]
) -> tuple[Callable[[int, int, int], Any], Callable[[int, int, int], Any]]:
    """Return hot and cold match temperature-change numerators."""

    if model.non_isothermal_model:

        def hot_drop(i: int, j: int, k: int) -> Any:
            return value(model.T_h[i][k]) - value(model.T_h_out_x[i][j][k])

        def cold_rise(i: int, j: int, k: int) -> Any:
            return value(model.T_c_out_y[j][i][k]) - value(model.T_c[j][k + 1])

    else:

        def hot_drop(i: int, j: int, k: int) -> Any:
            return value(model.T_h[i][k]) - value(model.T_h[i][k + 1])

        def cold_rise(i: int, j: int, k: int) -> Any:
            return value(model.T_c[j][k]) - value(model.T_c[j][k + 1])

    return hot_drop, cold_rise


def _set_flow_fraction_grids(
    model,
    value: Callable[[Any], Any],
    flow_fraction: Callable[[Any, int, int, int], Any],
) -> None:
    """Set ``P_h`` and ``P_c`` from the hot drop and cold rise numerators."""

    hot_drop, cold_rise = _temperature_drop_functions(model, value)
    shape = _recovery_grid_shape(model)
    model.P_h = build_index_grid(
        lambda i, j, k: flow_fraction(lambda: hot_drop(i, j, k), i, j, k), shape
    )
    model.P_c = build_index_grid(
        lambda i, j, k: flow_fraction(lambda: cold_rise(i, j, k), i, j, k), shape
    )


def _heat_load_total_grids(
    model, value: Callable[[Any], Any], wrap: Callable[[Any], Any]
) -> tuple[list, list]:
    """Return recovered heat totals per hot stream and per cold stream."""

    per_hot_stream = build_index_grid(
        lambda i, k: wrap(sum(value(model.Q_r[i][j][k]) for j in range(model.J))),
        (model.I, model.S),
    )
    per_cold_stream = build_index_grid(
        lambda j, k: wrap(sum(value(model.Q_r[i][j][k]) for i in range(model.I))),
        (model.J, model.S),
    )
    return per_hot_stream, per_cold_stream


def _set_match_indicator_grids(
    model, value: Callable[[Any], Any], wrap: Callable[[Any], Any]
) -> None:
    """Set ``z_i`` and ``z_j``: smoothed indicators that a stage has a match."""

    def cold_match_count(j: int, k: int) -> Any:
        return sum(value(model.z[i][j][k]) for i in range(model.I))

    def hot_match_count(i: int, k: int) -> Any:
        return sum(value(model.z[i][j][k]) for j in range(model.J))

    model.z_i = build_index_grid(
        lambda j, k: wrap(cold_match_count(j, k) / (cold_match_count(j, k) + 1e-9)),
        (model.J, model.S),
    )
    model.z_j = build_index_grid(
        lambda i, k: wrap(hot_match_count(i, k) / (hot_match_count(i, k) + 1e-9)),
        (model.I, model.S),
    )


def _set_postoptimisation_flow_fractions(model) -> None:
    """Set numeric flow fractions from a solved model's ``[value]`` arrays."""

    def flow_fraction(numerator: Callable[[], float], i: int, j: int, k: int):
        if model.T_h[i][k][0] > model.T_c[j][k + 1][0]:
            return numerator() / (model.T_h[i][k][0] - model.T_c[j][k + 1][0])
        return 0.0

    _set_flow_fraction_grids(model, _solved_value, flow_fraction)

    model.Sum_Qr_is, model.Sum_Qr_js = _heat_load_total_grids(
        model, _solved_value, lambda total: [total]
    )
    model.beta_h = build_index_grid(
        lambda i, j, k: (
            model.Q_r[i][j][k][0] / model.Sum_Qr_is[i][k][0]
            if model.Sum_Qr_is[i][k][0] > 0
            else 0.0
        ),
        _recovery_grid_shape(model),
    )
    model.beta_c = build_index_grid(
        lambda i, j, k: (
            model.Q_r[i][j][k][0] / model.Sum_Qr_js[j][k][0]
            if model.Sum_Qr_js[j][k][0] > 0
            else 0.0
        ),
        _recovery_grid_shape(model),
    )
    _set_match_indicator_grids(model, _solved_value, _unchanged)


def _set_model_flow_fractions(model, m: Any) -> None:
    """Set flow fractions as GEKKO intermediates of the optimisation model."""

    def flow_fraction(numerator: Callable[[], Any], i: int, j: int, k: int):
        return m.Intermediate(
            numerator()
            * model.z[i][j][k]
            / ((model.T_h[i][k] - model.T_c[j][k + 1] - 1) * model.z[i][j][k] + 1)
        )

    _set_flow_fraction_grids(model, _model_value, flow_fraction)

    model.Sum_Qr_j, model.Sum_Qr_i = _heat_load_total_grids(
        model, _model_value, m.Intermediate
    )
    model.beta_h = build_index_grid(
        lambda i, j, k: m.Intermediate(
            model.Q_r[i][j][k] / (model.Sum_Qr_j[i][k] + 1 - model.z[i][j][k])
        ),
        _recovery_grid_shape(model),
    )
    model.beta_c = build_index_grid(
        lambda i, j, k: m.Intermediate(
            model.Q_r[i][j][k] / (model.Sum_Qr_i[j][k] + 1 - model.z[i][j][k])
        ),
        _recovery_grid_shape(model),
    )
    _set_match_indicator_grids(model, _model_value, m.Intermediate)


def _set_alpha_gamma_variables(model, m: Any) -> None:
    """Create the alpha and hot/cold gamma flow-on variables."""

    def variable_grid(prefix: str, initial_value: float) -> list:
        return build_index_grid(
            lambda i, j, k: m.Var(
                value=initial_value,
                ub=1.0,
                lb=-1.0,
                name=f"{prefix}_H{i}_to_C{j}_at_S{k}",
            ),
            _recovery_grid_shape(model),
        )

    model.alpha = variable_grid("alpha", 0.0)
    model.gamma_h = variable_grid("gamma_h", 0.5)
    model.gamma_c = variable_grid("gamma_c", 0.5)


def _hot_gamma_flow_on(model, i: int, j: int, k: int) -> Any:
    """Flow-on from the hot stream's matches in the next stage."""

    return (
        sum(
            model.beta_h[i][j0][k + 1]
            * model.P_h[i][j0][k + 1]
            * model.alpha[i][j0][k + 1]
            for j0 in range(model.J)
        )
        + (1 - model.z_j[i][k + 1]) * model.gamma_h[i][j][k + 1]
    )


def _cold_gamma_flow_on(model, i: int, j: int, k: int) -> Any:
    """Flow-on from the cold stream's matches in the previous stage."""

    return (
        sum(
            model.beta_c[i0][j][k - 1]
            * model.P_c[i0][j][k - 1]
            * model.alpha[i0][j][k - 1]
            for i0 in range(model.I)
        )
        + (1 - model.z_i[j][k - 1]) * model.gamma_c[i][j][k - 1]
    )


def _set_gamma_equations(model, m: Any) -> None:
    """Set gamma equations; the last stage has no hot flow-on, the first no cold."""

    model.gamma_h_eqn = []
    model.gamma_c_eqn = []
    for k, j, i in _recovery_indices(model):
        last_stage = k + 1 >= model.S
        first_stage = not last_stage and k - 1 < 0
        hot_flow_on = 0.0 if last_stage else _hot_gamma_flow_on(model, i, j, k)
        model.gamma_h_eqn.append([m.Equation(model.gamma_h[i][j][k] == hot_flow_on)])
        cold_flow_on = 0.0 if first_stage else _cold_gamma_flow_on(model, i, j, k)
        model.gamma_c_eqn.append([m.Equation(model.gamma_c[i][j][k] == cold_flow_on)])


def _set_dqda_equations(model, m: Any) -> None:
    """Set the minimum dQ/dA constraint for each allowed match."""

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
        for k, j, i in _recovery_indices(model)
    ]

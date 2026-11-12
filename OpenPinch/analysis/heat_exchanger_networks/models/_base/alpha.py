"""Post-optimisation alpha flow-on values for base HEN models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ...indexing import build_index_grid
from . import execution as _execution

logger = logging.getLogger(__name__)

# Alpha and the gammas are fractions of a match's duty, so they lie in [-1, 1].
_ALPHA_BOUND = 1.0
_SOLVE_TOLERANCE = 1e-8


def get_alpha_values(model) -> list | None:
    """Calculate source alpha flow-on values from a solved model.

    The alpha and gamma equations are linear once the solved temperatures
    and duties are fixed, so they are solved as one sparse system. Returns
    ``None`` when the system is singular or its solution leaves [-1, 1];
    nothing is cached, callers treat alpha as unavailable, and a later call
    retries.
    """

    if model.alpha != []:
        return model.alpha

    _set_postoptimisation_flow_fractions(model)
    solution = _solve_flow_on_system(model)
    if solution is None:
        logger.warning(
            "Post-optimisation alpha solve for %s has no unique solution in "
            "[-1, 1]; alpha values are unavailable and benefit ranking falls "
            "back to neutral alpha.",
            getattr(model, "name", "model"),
        )
        model.alpha = []
        return None
    model.alpha = build_index_grid(
        lambda i, j, k: [float(solution[i, j, k])], _recovery_grid_shape(model)
    )
    return model.alpha


def _solve_flow_on_system(model) -> np.ndarray | None:
    """Solve for alpha, ``gamma_h`` and ``gamma_c`` at every recovery index.

    ``alpha = 1 - (gamma_h + gamma_c) / 2``; ``gamma_h`` takes flow-on from the
    hot stream's matches in the next stage (none in the last), ``gamma_c``
    from the cold stream's matches in the previous stage (none in the first).
    Returns the alpha grid, or ``None`` if the solution is not usable.
    """

    count = model.I * model.J * model.S

    def alpha(i: int, j: int, k: int) -> int:
        return (i * model.J + j) * model.S + k

    def gamma_h(i: int, j: int, k: int) -> int:
        return count + alpha(i, j, k)

    def gamma_c(i: int, j: int, k: int) -> int:
        return 2 * count + alpha(i, j, k)

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    right_hand_side = np.zeros(3 * count)

    def add(row: int, column: int, coefficient: float) -> None:
        # Inactive matches give zero flow-on coefficients. Storing them makes
        # the LU factorisation fill in: on a 64-stream network most of the
        # matrix would be stored zeros.
        if coefficient == 0.0:
            return
        rows.append(row)
        columns.append(column)
        coefficients.append(coefficient)

    for k, j, i in _recovery_indices(model):
        row = alpha(i, j, k)
        add(row, row, 1.0)
        add(row, gamma_h(i, j, k), 0.5)
        add(row, gamma_c(i, j, k), 0.5)
        right_hand_side[row] = 1.0

        row = gamma_h(i, j, k)
        add(row, row, 1.0)
        if k + 1 < model.S:
            for j0 in range(model.J):
                add(
                    row,
                    alpha(i, j0, k + 1),
                    -model.beta_h[i][j0][k + 1] * model.P_h[i][j0][k + 1],
                )
            add(row, gamma_h(i, j, k + 1), -(1 - model.z_j[i][k + 1]))

        row = gamma_c(i, j, k)
        add(row, row, 1.0)
        if k > 0:
            for i0 in range(model.I):
                add(
                    row,
                    alpha(i0, j, k - 1),
                    -model.beta_c[i0][j][k - 1] * model.P_c[i0][j][k - 1],
                )
            add(row, gamma_c(i, j, k - 1), -(1 - model.z_i[j][k - 1]))

    matrix = sp.csc_matrix(
        (coefficients, (rows, columns)), shape=(3 * count, 3 * count)
    )
    with warnings.catch_warnings():
        # A singular matrix is reported by the checks below.
        warnings.simplefilter("ignore", spla.MatrixRankWarning)
        solution = spla.spsolve(matrix, right_hand_side)
    if not np.all(np.isfinite(solution)):
        return None
    if np.max(np.abs(matrix @ solution - right_hand_side)) > _SOLVE_TOLERANCE:
        return None
    if np.max(np.abs(solution)) > _ALPHA_BOUND + _SOLVE_TOLERANCE:
        return None
    return solution[:count].reshape(model.I, model.J, model.S)


def alpha_value(model, i: int, j: int, k: int) -> float:
    """Return one solved alpha, or a neutral 1.0 when alpha is unavailable.

    Alpha scales a match's recovered heat by its flow-on effect, so 1.0
    ranks matches on their own duty and area alone.
    """

    if len(model.alpha) == 0:
        return 1.0
    return float(_execution._value(model.alpha[i][j][k]))


def _recovery_grid_shape(model) -> tuple[int, int, int]:
    return (model.I, model.J, model.S)


def _recovery_indices(model):
    """Yield ``(k, j, i)`` in the stage-major order used for equation lists."""

    for k in range(model.S):
        for j in range(model.J):
            for i in range(model.I):
                yield k, j, i


def _solved_value(value: Any) -> Any:
    return _execution._value(value)


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
        if _execution._value(model.T_h[i][k]) > _execution._value(model.T_c[j][k + 1]):
            return numerator() / (
                _execution._value(model.T_h[i][k])
                - _execution._value(model.T_c[j][k + 1])
            )
        return 0.0

    _set_flow_fraction_grids(model, _solved_value, flow_fraction)

    model.Sum_Qr_is, model.Sum_Qr_js = _heat_load_total_grids(
        model, _solved_value, lambda total: [total]
    )
    model.beta_h = build_index_grid(
        lambda i, j, k: (
            _execution._value(model.Q_r[i][j][k])
            / _execution._value(model.Sum_Qr_is[i][k])
            if _execution._value(model.Sum_Qr_is[i][k]) > 0
            else 0.0
        ),
        _recovery_grid_shape(model),
    )
    model.beta_c = build_index_grid(
        lambda i, j, k: (
            _execution._value(model.Q_r[i][j][k])
            / _execution._value(model.Sum_Qr_js[j][k])
            if _execution._value(model.Sum_Qr_js[j][k]) > 0
            else 0.0
        ),
        _recovery_grid_shape(model),
    )
    _set_match_indicator_grids(model, _solved_value, _unchanged)

"""Post-optimisation alpha flow-on values for base HEN models."""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

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

    Unknowns are numbered alpha, then ``gamma_h``, then ``gamma_c``, each in
    (hot, cold, stage) order. The coefficients are built as whole-grid array
    operations: assembling them one at a time took seconds per pulp mill
    post-process.
    """

    hot, cold, stages = model.I, model.J, model.S
    count = hot * cold * stages
    shape = (hot, cold, stages)
    alpha = np.arange(count).reshape(shape)
    gamma_h = count + alpha
    gamma_c = 2 * count + alpha
    beta_h = np.asarray(model.beta_h, dtype=float).reshape(shape)
    p_h = np.asarray(model.P_h, dtype=float).reshape(shape)
    beta_c = np.asarray(model.beta_c, dtype=float).reshape(shape)
    p_c = np.asarray(model.P_c, dtype=float).reshape(shape)
    z_j = np.asarray(model.z_j, dtype=float).reshape(hot, stages)
    z_i = np.asarray(model.z_i, dtype=float).reshape(cold, stages)

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []

    def add(row, column, coefficient) -> None:
        # Inactive matches give zero flow-on coefficients. Storing them makes
        # the LU factorisation fill in: on a 64-stream network most of the
        # matrix would be stored zeros.
        row, column, coefficient = np.broadcast_arrays(row, column, coefficient)
        kept = coefficient != 0.0
        rows.append(row[kept])
        columns.append(column[kept])
        coefficients.append(coefficient[kept])

    add(alpha, alpha, 1.0)
    add(alpha, gamma_h, 0.5)
    add(alpha, gamma_c, 0.5)
    add(gamma_h, gamma_h, 1.0)
    add(gamma_c, gamma_c, 1.0)

    # gamma_h at (i, j, k) takes every hot match (i, j0, k + 1) of the next
    # stage, whatever its partner j.
    hot_flow_on = -beta_h * p_h
    i, j0, k = np.nonzero(hot_flow_on[:, :, 1:])
    k = k + 1
    add(
        gamma_h[i, :, k - 1],
        alpha[i, j0, k][:, None],
        hot_flow_on[i, j0, k][:, None],
    )
    add(gamma_h[:, :, :-1], gamma_h[:, :, 1:], -(1 - z_j[:, None, 1:]))

    # gamma_c at (i, j, k) takes every cold match (i0, j, k - 1) of the
    # previous stage, whatever its partner i.
    cold_flow_on = -beta_c * p_c
    i0, j, k = np.nonzero(cold_flow_on[:, :, :-1])
    add(
        gamma_c[:, j, k + 1].T,
        alpha[i0, j, k][:, None],
        cold_flow_on[i0, j, k][:, None],
    )
    add(gamma_c[:, :, 1:], gamma_c[:, :, :-1], -(1 - z_i[None, :, :-1]))

    right_hand_side = np.zeros(3 * count)
    right_hand_side[:count] = 1.0
    matrix = sp.csc_matrix(
        (
            np.concatenate(coefficients),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(3 * count, 3 * count),
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
    return solution[:count].reshape(shape)


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


def _solved_arrays(model) -> SimpleNamespace:
    """Read each solved value the flow fractions use once, into arrays."""

    def solved(grid, shape):
        values = np.fromiter(
            (_execution._value(item) for item in _flattened(grid, len(shape))),
            dtype=float,
            count=int(np.prod(shape)),
        )
        return values.reshape(shape)

    boundaries = model.S + 1
    recovery = _recovery_grid_shape(model)
    arrays = SimpleNamespace(
        T_h=solved(model.T_h, (model.I, boundaries)),
        T_c=solved(model.T_c, (model.J, boundaries)),
        Q_r=solved(model.Q_r, recovery),
        z=solved(model.z, recovery),
    )
    if model.non_isothermal_model:
        arrays.T_h_out_x = solved(model.T_h_out_x, recovery)
        arrays.T_c_out_y = solved(model.T_c_out_y, (model.J, model.I, model.S))
    return arrays


def _flattened(grid, depth: int):
    if depth == 0:
        yield grid
        return
    for item in grid:
        yield from _flattened(item, depth - 1)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator`` where the denominator is positive, else 0."""

    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape)),
        where=denominator > 0,
    )


def _sum_over(values: np.ndarray, axis: int) -> np.ndarray:
    """Sum along ``axis`` with Python's ``sum``.

    Python's float ``sum`` is compensated, so a NumPy sum can differ in the
    last place; the totals set the duty fractions that rank matches.
    """

    lines = np.moveaxis(values, axis, -1)
    totals = [sum(line) for line in lines.reshape(-1, lines.shape[-1]).tolist()]
    return np.array(totals, dtype=float).reshape(lines.shape[:-1])


def _set_postoptimisation_flow_fractions(model) -> None:
    """Set the flow fractions, as (hot, cold, stage) arrays, from a solved model."""

    solved = _solved_arrays(model)
    t_h, t_c = solved.T_h, solved.T_c
    # Temperature span from the hot inlet to the cold inlet of each match.
    span = t_h[:, None, :-1] - t_c[None, :, 1:]
    if model.non_isothermal_model:
        hot_drop = t_h[:, None, :-1] - solved.T_h_out_x
        cold_rise = solved.T_c_out_y.transpose(1, 0, 2) - t_c[None, :, 1:]
    else:
        hot_drop = np.broadcast_to((t_h[:, :-1] - t_h[:, 1:])[:, None, :], span.shape)
        cold_rise = np.broadcast_to((t_c[:, :-1] - t_c[:, 1:])[None, :, :], span.shape)
    model.P_h = _ratio(hot_drop, span)
    model.P_c = _ratio(cold_rise, span)

    model.Sum_Qr_is = _sum_over(solved.Q_r, axis=1)
    model.Sum_Qr_js = _sum_over(solved.Q_r, axis=0)
    model.beta_h = _ratio(solved.Q_r, model.Sum_Qr_is[:, None, :])
    model.beta_c = _ratio(solved.Q_r, model.Sum_Qr_js[None, :, :])

    # Smoothed indicators that a stream has a match in a stage.
    cold_matches = _sum_over(solved.z, axis=0)
    hot_matches = _sum_over(solved.z, axis=1)
    model.z_i = cold_matches / (cold_matches + 1e-9)
    model.z_j = hot_matches / (hot_matches + 1e-9)

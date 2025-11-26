"""Shared numerical helpers."""

from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing, minimize

from ..lib import *


def key_name(zone_name: str, target_type: str = TargetType.DI.value):
    """Compose the canonical dictionary key for storing zone targets."""
    return f"{zone_name}/{target_type}"


def get_value(val: Union[float, dict, ValueWithUnit]) -> float:
    """Extract a numeric value from raw floats, dict payloads, or :class:`ValueWithUnit`."""
    if isinstance(val, float):
        return val
    elif isinstance(val, dict):
        return val["value"]
    elif isinstance(val, ValueWithUnit):
        return val.value
    else:
        raise TypeError(
            f"Unsupported type: {type(val)}. Expected float, dict, or ValueWithUnit."
        )

def linear_interpolation(xi: float, x1: float, x2: float, y1: float, y2: float) -> float:
    """Performs linear interpolation to estimate y at a given x, using two known points (x1, y1) and (x2, y2)."""
    if x1 == x2:
        raise ValueError(
            "Cannot perform interpolation when x1 == x2 (undefined slope)."
        )
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    yi = m * xi + c
    return yi


def delta_with_zero_at_start(x: np.ndarray) -> np.ndarray:
    """Compute difference between successive entries in a column and include a zero in the first entry."""  
    return np.insert(
        delta_vals(x),
        0,
        0.0
    ) 


def delta_vals(x: np.ndarray, descending_vals: bool = True) -> np.ndarray:
    """Compute difference between successive entries in a column."""        
    return (
        x[:-1] - x[1:] 
        if descending_vals else
        x[1:] - x[:-1]
    )


def clean_composite_curve_ends(
    y_vals: np.ndarray | list, x_vals: np.ndarray | list
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove redundant points in composite curves."""
    y_vals = np.array(y_vals)
    x_vals = np.array(x_vals)
    
    if np.abs(x_vals.sum()) < tol or np.abs(x_vals.var()) < tol:
        return np.array([]), np.array([])
    
    mask_0 = ~np.isclose(x_vals, x_vals[0] * np.ones(len(x_vals)), atol=tol)
    start = np.flatnonzero(mask_0)[0] - 1
    mask_1 = ~np.isclose(x_vals, x_vals[-1] * np.ones(len(x_vals)), atol=tol)
    end = np.flatnonzero(mask_1)[-1] + 1

    x_clean = x_vals[start:end+1]
    y_clean = y_vals[start:end+1]      
    return y_clean, x_clean


def clean_composite_curve(
    y_array: np.ndarray | list, x_array: np.ndarray | list
) -> Tuple[np.ndarray | list]:
    """Remove redundant points in composite curves."""

    # Round to avoid tiny numerical errors
    y_vals, x_vals = clean_composite_curve_ends(y_array, x_array)

    if len(x_vals) <= 2:
        return y_vals, x_vals

    x_clean, y_clean = [x_vals[0]], [y_vals[0]]

    for i in range(1, len(x_vals) - 1):
        x1, x2, x3 = x_vals[i - 1], x_vals[i], x_vals[i + 1]
        y1, y2, y3 = y_vals[i - 1], y_vals[i], y_vals[i + 1]

        if x1 == x3:
            # All three x are the same; keep x2 only if y2 is different
            if x1 != x2:
                x_clean.append(x2)
                y_clean.append(y2)
        else:
            # Linear interpolation check
            y_interp = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
            if abs(y2 - y_interp) > tol:
                x_clean.append(x2)
                y_clean.append(y2)

    x_clean.append(x_vals[-1])
    y_clean.append(y_vals[-1])

    if abs(x_clean[0] - x_clean[1]) < tol:
        x_clean.pop(0)
        y_clean.pop(0)

    i = len(x_clean) - 1
    if abs(x_clean[i] - x_clean[i - 1]) < tol:
        x_clean.pop(i)
        y_clean.pop(i)

    return y_clean, x_clean


def graph_simple_cc_plot(Tc, Hc, Th, Hh):
    fig, ax = plt.subplots()
    ax.plot(Hc, Tc, label="Cold composite")
    ax.plot(Hh, Th, label="Hot composite")
    ax.set_ylabel("Temperature")
    ax.set_xlabel("Enthalpy")
    ax.set_title("Balanced Composite Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def interp_with_plateaus(
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    targets: np.ndarray,
    side: str,
    tol: float = 1e-6,
) -> np.ndarray:
    """Interpolate temperatures while respecting vertical segments in the composite curves."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    h_vals = np.asarray(h_vals, dtype=float)
    t_vals = np.asarray(t_vals, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if h_vals.size == 1:
        return np.full_like(targets, t_vals[0], dtype=float)

    h_monotonic = make_monotonic(h_vals, side, tol)
    return np.interp(targets, h_monotonic, t_vals)


# def _make_monotonic(
#     h_vals: np.ndarray, 
#     side: str, 
#     tol: float = 1e-6
# ) -> np.ndarray:
#     """Adjust an array so repeated values become strictly increasing for interpolation."""
#     adjusted = np.array(h_vals, dtype=float, copy=True)
#     if adjusted.size <= 1:
#         return adjusted

#     eps = tol * 0.5

#     idx = 0
#     n = adjusted.size
#     while idx < n - 1:
#         j = idx + 1
#         while j < n and abs(adjusted[j] - adjusted[idx]) <= tol:
#             j += 1
#         if j - idx > 1:
#             length = j - idx
#             if side == "right":
#                 offsets = np.arange(length - 1, -1, -1, dtype=float) * eps
#                 adjusted[idx:j] = adjusted[idx] - offsets
#             else:  # side == "left"
#                 offsets = np.arange(length, dtype=float) * eps
#                 adjusted[idx:j] = adjusted[idx] + offsets
#         idx = j

#     return adjusted


def make_monotonic(
    h_vals: np.ndarray, 
    side: str, 
    tol: float = 1e-6
) -> np.ndarray:
    """Adjust an array so repeated values become strictly increasing for interpolation."""
    adjusted = np.asarray(h_vals, dtype=float).copy()
    if adjusted.size <= 1:
        return adjusted

    eps = tol * 0.5
    # Identify the start of each strictly increasing block
    diff = np.abs(np.diff(adjusted)) > tol
    starts = np.flatnonzero(np.concatenate(([True], diff)))
    n = adjusted.size
    lengths = np.diff(np.append(starts, n))

    if np.all(lengths == 1):
        return adjusted

    # Compute position within each block using vectorised repetition
    within_block = np.arange(n) - np.repeat(starts, lengths)
    block_lengths = np.repeat(lengths, lengths)
    mask = block_lengths > 1

    offsets = np.zeros_like(adjusted)
    if side == "right":
        offsets[mask] = (block_lengths[mask] - 1 - within_block[mask]) * eps
        adjusted[mask] -= offsets[mask]
    else:  # side == "left"
        offsets[mask] = within_block[mask] * eps
        adjusted[mask] += offsets[mask]

    return adjusted


def dual_annealing_multiminima(
    func,
    bounds,
    args=(),
    constraints=(),
    n_runs=6,
    maxiter=300,
    seed=150,
    no_local_search=False,
    initial_temp=5230.0,
    restart_temp_ratio=2e-5,
    visit=2.62,
    accept=-5.0,
    maxfun=1_000_000,
    # clustering + verification parameters
    cluster_tol=0.05,      # normalized distance for *first* clustering
    x_tol_norm=0.01,       # normalized distance to consider two minima identical
    f_tol_rel=0.01,        # relative f difference tolerance
    local_method="SLSQP",  # default local method
):
    """
    Multi-start dual_annealing that returns *verified* local optima by:
      - collecting candidate minima via callback across multiple runs,
      - clustering them in normalized decision space,
      - polishing each cluster representative via constrained local optimization,
      - deduplicating polished minima in (x, f) space.

    Parameters
    ----------
    func : callable
        Objective f(x, *args) -> scalar, with x a 1D array-like.
    bounds : sequence of (lb, ub)
        Bounds [(lb1, ub1), ..., (lbn, ubn)].
    args : tuple
        Extra arguments passed to func as func(x, *args).
    constraints : dict or sequence of dict, optional
        SciPy-style constraints for `minimize` (e.g. for SLSQP or trust-constr).
        They are applied only in the polishing step, not in dual_annealing itself.
    n_runs : int
        Number of independent dual_annealing runs with different seeds.
    maxiter : int
        dual_annealing global search iterations per run.
    seed : int
        Base seed; run r uses seed + r.
    no_local_search : bool
        Passed through to dual_annealing.
    initial_temp, restart_temp_ratio, visit, accept, maxfun :
        Standard dual_annealing parameters.
    cluster_tol : float
        Distance tolerance in *normalized* decision space (0-1 per dim)
        for first-stage basin clustering.
    x_tol_norm : float
        Normalized distance threshold to treat two polished minima as identical.
    f_tol_rel : float
        Relative objective difference threshold to treat minima as identical.
    local_method : str
        Local method for polishing (e.g. 'SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr').

    Returns
    -------
    global_best : dict
        {"x": ndarray, "fun": float}
    local_minima : list of dict
        [{"x": ndarray, "fun": float}, ...] verified and deduplicated minima.
    all_points : ndarray
        All candidate minima *before* polishing, shape (N, dim).
    all_values : ndarray
        Corresponding objective values, shape (N,).
    """
    bounds = np.asarray(bounds, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    span = ub - lb

    all_x = []
    all_f = []

    # --------- 1. Collect candidate minima across multiple DA runs ----------
    for run in range(n_runs):
        run_minima_x = []
        run_minima_f = []

        def callback(x, f, context):
            # context: 0 = annealing, 1 = end of local search, 2 = end of run
            run_minima_x.append(np.array(x, dtype=float))
            run_minima_f.append(float(f))
            return False

        res = dual_annealing(
            func,
            bounds=bounds,
            args=args,
            maxiter=maxiter,
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            visit=visit,
            accept=accept,
            maxfun=maxfun,
            seed=seed + run,
            no_local_search=no_local_search,
            callback=callback,
        )

        # Ensure final best of this run is included
        run_minima_x.append(np.array(res.x, dtype=float))
        run_minima_f.append(float(res.fun))

        all_x.extend(run_minima_x)
        all_f.extend(run_minima_f)

    all_x = np.asarray(all_x)
    all_f = np.asarray(all_f)

    # --------- Global best over all runs (raw) ----------
    g_idx = np.argmin(all_f)
    global_best_raw = {"x": all_x[g_idx], "fun": float(all_f[g_idx])}

    # --------- 2. First-stage clustering (basin representatives) ----------
    def _cluster_candidates(xs, fs, lb, ub, tol_norm):
        """
        Greedy clustering in normalized decision space.
        Accepts the best point first, then any point farther than tol_norm
        from all existing centers (in normalized coordinates).
        """
        idx_sorted = np.argsort(fs)
        xs_norm = (xs - lb) / (ub - lb)

        centers_norm = []
        centers_idx = []

        for idx in idx_sorted:
            x_norm = xs_norm[idx]
            if not centers_norm:
                centers_norm.append(x_norm.copy())
                centers_idx.append(idx)
                continue

            dists = np.linalg.norm(np.asarray(centers_norm) - x_norm, axis=1)
            if np.all(dists > tol_norm):
                centers_norm.append(x_norm.copy())
                centers_idx.append(idx)

        return centers_idx

    basin_reps_idx = _cluster_candidates(all_x, all_f, lb, ub, cluster_tol)

    # --------- 3. Polishing: constrained local optimization from each rep ----------
    polished_x = []
    polished_f = []
    constraints = () if constraints is None else constraints

    for idx in basin_reps_idx:
        x0 = all_x[idx]

        res_loc = minimize(
            lambda x: func(x, *args),
            x0,
            method=local_method,
            bounds=bounds if local_method.upper() in ("SLSQP", "L-BFGS-B", "TNC") else None,
            constraints=constraints,
        )
        polished_x.append(np.array(res_loc.x, dtype=float))
        polished_f.append(float(res_loc.fun))

    polished_x = np.asarray(polished_x)
    polished_f = np.asarray(polished_f)

    # --------- 4. Deduplicate polished minima ----------
    def _dedup_minima(xs, fs, lb, ub, x_tol_norm, f_tol_rel):
        """
        Treat minima as identical if:
          - normalized distance <= x_tol_norm, AND
          - |f1 - f2| <= f_tol_rel * max(1, |f1|, |f2|)
        We keep the best (lowest f) representative.
        """
        idx_sorted = np.argsort(fs)
        xs_norm = (xs - lb) / (ub - lb)

        reps = []
        reps_vals = []

        for idx in idx_sorted:
            x_norm = xs_norm[idx]
            f = fs[idx]

            if not reps:
                reps.append(xs[idx].copy())
                reps_vals.append(f)
                continue

            is_new = True
            for r, fr in zip(reps, reps_vals):
                r_norm = (r - lb) / (ub - lb)
                dx = np.linalg.norm(x_norm - r_norm)
                df = abs(f - fr)
                scale = max(1.0, abs(f), abs(fr))
                if dx <= x_tol_norm and df <= f_tol_rel * scale:
                    is_new = False
                    break

            if is_new:
                reps.append(xs[idx].copy())
                reps_vals.append(f)

        return [float(fr) for fr in reps_vals], [r.tolist() for r in reps]

    local_minima_fun, local_minima_x = _dedup_minima(
        polished_x,
        polished_f,
        lb,
        ub,
        x_tol_norm=x_tol_norm,
        f_tol_rel=f_tol_rel,
    )

    return local_minima_fun, local_minima_x

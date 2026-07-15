"""Generate piecewise-linear approximations for non-linear thermodynamic streams."""

from typing import List

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from ..lib.schemas.io import NonLinearStream

__all__ = [
    "align_temperature_heat_profiles",
    "build_segmented_stream_from_profile",
    "get_piecewise_data_points",
    "get_piecewise_linearisation_for_streams",
    "normalise_temperature_heat_profile",
]


################################################################################
# Public API
################################################################################


def get_piecewise_linearisation_for_streams(
    streams: List[NonLinearStream],
    t_h_data: list,
    dt_diff_max: float = 0.1,
) -> dict[str, list[list[list[float]]]]:
    """Generate piecewise-linear T-H profiles for non-linear streams."""
    if len(streams) != len(t_h_data):
        raise ValueError(
            "Piecewise linearisation failed due to a different number of "
            "streams and temperature-enthalpy datasets."
        )

    return_data: dict[str, list[list[list[float]]]] = {"t_h_points": []}

    # Create and Linearize stream
    for index, s in enumerate(streams):
        is_hot_stream = s.t_supply > s.t_target
        curve_points = t_h_data[index]
        mask_points = get_piecewise_data_points(
            curve=curve_points, dt_diff_max=dt_diff_max, is_hot_stream=is_hot_stream
        )
        return_data["t_h_points"].append(mask_points.tolist())

    return return_data


def normalise_temperature_heat_profile(
    profile,
    *,
    is_hot_stream: bool,
    minimum_temperature_span: float = 0.01,
) -> np.ndarray:
    """Preserve profile order while enforcing the sensible-stream span convention."""
    points = np.asarray(profile, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("A temperature-heat profile requires at least two points.")
    if not np.isfinite(points).all():
        raise ValueError("Temperature-heat profile points must be finite.")
    if minimum_temperature_span <= 0.0:
        raise ValueError("minimum_temperature_span must be positive.")
    heat_steps = np.diff(points[:, 0])
    if np.any(heat_steps == 0.0) or not np.all(
        np.sign(heat_steps) == np.sign(heat_steps[0])
    ):
        raise ValueError("Profile heat coordinates must be strictly monotonic.")

    direction = -1.0 if is_hot_stream else 1.0
    directed = direction * points[:, 1]
    total_span = directed[-1] - directed[0]
    if total_span <= 0.0:
        directed[-1] = directed[0] + minimum_temperature_span * (len(points) - 1)
        total_span = directed[-1] - directed[0]
    step = min(minimum_temperature_span, total_span / (len(points) - 1))
    previous = directed[0]
    for index in range(1, len(points) - 1):
        lower = previous + step
        upper = directed[-1] - step * (len(points) - 1 - index)
        directed[index] = np.clip(directed[index], lower, upper)
        previous = directed[index]
    points[:, 1] = direction * directed
    return points


def align_temperature_heat_profiles(profiles) -> tuple[np.ndarray, ...]:
    """Interpolate period profiles onto their union cumulative-duty-fraction grid."""
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    breakpoints: list[np.ndarray] = []
    for profile in profiles:
        points = np.asarray(profile, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError(
                "Every period profile requires at least two [heat, temperature] points."
            )
        if not np.isfinite(points).all():
            raise ValueError("Temperature-heat profile points must be finite.")
        heat_steps = np.diff(points[:, 0])
        if np.any(heat_steps == 0.0) or not np.all(
            np.sign(heat_steps) == np.sign(heat_steps[0])
        ):
            raise ValueError("Profile heat coordinates must be strictly monotonic.")
        increments = np.abs(heat_steps)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        fractions = cumulative / cumulative[-1]
        prepared.append((points, fractions))
        breakpoints.append(fractions)

    if not prepared:
        raise ValueError("At least one period profile is required.")
    union = np.unique(np.concatenate(breakpoints))
    aligned = []
    for points, fractions in prepared:
        heat = np.interp(union, fractions, points[:, 0])
        temperature = np.interp(union, fractions, points[:, 1])
        aligned.append(np.column_stack((heat, temperature)))
    return tuple(aligned)


def build_segmented_stream_from_profile(
    *,
    name: str,
    profile,
    heat_scale: float = 1.0,
    heat_unit: str = "kW",
    is_hot_stream: bool,
    minimum_temperature_span: float = 0.01,
    **stream_kwargs,
):
    """Build one parent :class:`Stream` from an ordered linearised profile."""
    from ..classes.stream import Stream

    raw_points = np.asarray(profile, dtype=float)
    if raw_points.ndim != 2 or raw_points.shape[1] != 2 or len(raw_points) < 2:
        raise ValueError("A temperature-heat profile requires at least two points.")
    keep = np.concatenate(
        ([True], np.abs(np.diff(raw_points[:, 0])) > np.finfo(float).eps)
    )
    points = normalise_temperature_heat_profile(
        raw_points[keep],
        is_hot_stream=is_hot_stream,
        minimum_temperature_span=minimum_temperature_span,
    )
    return Stream.from_temperature_heat_profile(
        name=name,
        points=points,
        heat_scale=heat_scale,
        heat_unit=heat_unit,
        **stream_kwargs,
    )


def get_piecewise_data_points(
    curve: list,
    is_hot_stream: bool,
    dt_diff_max: float = 0.1,
) -> np.array:
    """
    Perform piecewise linearisation with the Ramer-Douglas-Peucker algorithm.

    :param curve: Numpy array of plot points for th curve
    :param dt_diff_max: Maximum allowed temperature differential tolerance
    :returns: Numpy array of new curve points
    """
    curve = np.array(curve)
    try:
        return _get_piecewise_breakpoints(
            curve=curve, epsilon=dt_diff_max, is_hot_stream=is_hot_stream
        )
    except FloatingPointError, RuntimeError, ValueError:
        try:
            return _rdp(
                curve=curve,
                epsilon=dt_diff_max,
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            raise ValueError("Piecewise linearisation failed.") from exc


################################################################################
# Helper functions
################################################################################


def _rdp(curve: np.array, epsilon: float) -> np.array:
    """
    Linearize and simplify a curve using the Ramer-Douglas-Peucker (_rdp) algorithm.

    :param curve: Array of points (N, 2).
    :param epsilon: Maximum allowed perpendicular distance for simplification
    :returns: Simplified array of points.
    """
    n = len(curve)
    indices = np.ones(n, dtype=bool)
    stack = [[0, n - 1]]

    while stack:
        start, end = stack.pop()
        dmax = 0.0
        index = start
        line_vector = curve[end] - curve[start]
        line_length = np.linalg.norm(line_vector)

        if line_length == 0:
            continue

        # Get point with max distance from line
        for i in range(start + 1, end):
            point_vector = curve[i] - curve[start]
            # Use a determinant form to avoid NumPy 2D cross deprecation.
            cross_magnitude = (
                line_vector[0] * point_vector[1] - line_vector[1] * point_vector[0]
            )
            distance = abs(cross_magnitude) / line_length
            if distance > dmax:
                dmax = distance
                index = i

        # Split if distance > epsilon (dt_diff_max)
        if dmax > epsilon:
            stack.append([start, index])
            stack.append([index, end])
        else:
            indices[start + 1 : end] = False

    return curve[indices]


def _refine_pw_points_for_heating_or_cooling(
    curve: np.array, pw_points: np.array, eps_lb: float = 0.0, hot_stream: bool = True
) -> np.array:
    """
    Refine a piecewise T-h approximation to preserve hot/cold stream integrity.

    :param curve: Array of points (N, 2).
    :param eps_lb: Maximum allowed hot or cold stream violation.
    :param hot_stream: True if the stream is hot, False if the stream is cold.
    :returns: Simplified array of points, maximum error.
    """
    # Ensure the data is a numpy array
    curve = np.flipud(curve)
    pw_points = np.flipud(pw_points)

    # Remove the first and last points because they are fixed
    # Convert 2d array to 1d array
    x0 = pw_points[1:-1].flatten()

    # Get arguments for the optimisation
    args = {
        "first_point": pw_points[0],
        "last_point": pw_points[-1],
    }

    def delta_pw_and_data(x, args):
        """Return the difference between the data and the piecewise points."""
        # Reshape so the first half are x values and the second half are y values.
        new_pw_points = np.vstack(
            (args["first_point"], x.reshape(-1, 2), args["last_point"])
        )
        # Interpolate the piecewise points to the original data points
        int_points = np.interp(curve[:, 0], new_pw_points[:, 0], new_pw_points[:, 1])
        # Find the difference between the data and the piecewise points
        return int_points - curve[:, 1]

    # Define the constraints for the optimisation
    if hot_stream:

        def con(x):
            return np.max(delta_pw_and_data(x, args))

        nlc = NonlinearConstraint(con, -np.inf, eps_lb)
    else:

        def con(x):
            return np.min(delta_pw_and_data(x, args))

        nlc = NonlinearConstraint(con, -eps_lb, np.inf)

    def obj(x, args):
        """Return the L2 norm between the data and the piecewise points."""
        return np.sum(np.square(delta_pw_and_data(x, args)))

    # Perform the optimisation
    res = minimize(fun=obj, x0=x0, constraints=nlc, args=args, method="SLSQP", tol=1e-6)
    refined_pw_points = np.vstack(
        (args["first_point"], res.x.reshape(-1, 2), args["last_point"])
    )
    return np.flipud(refined_pw_points), np.max(np.abs(delta_pw_and_data(res.x, args)))


def _get_piecewise_breakpoints(
    curve: np.array, epsilon: float, is_hot_stream: bool = True
) -> np.array:
    """
    Get piecewise breakpoints using RDP plus an integrity refinement step.

    :param curve: Array of points (N, 2).
    :param epsilon: Maximum allowed perpendicular distance for simplification.
    :param hot_stream: True if the stream is hot, False if the stream is cold.
    :returns: Simplified array of breakpoints that define the piecewise linearisation.
    """
    for _ in range(10):
        pw_points = _rdp(curve, epsilon=epsilon)

        if len(pw_points) > 10:
            pw_points, max_err = _refine_pw_points_for_heating_or_cooling(
                curve, pw_points, epsilon / 10, is_hot_stream
            )
        else:
            break

        if max_err > epsilon:
            epsilon = epsilon * 0.9
        else:
            break

    return pw_points

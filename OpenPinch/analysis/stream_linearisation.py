import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from ..lib import *

__all__ = ["get_piecewise_linearisation_for_streams"]


#######################################################################################################
# Public API
#######################################################################################################

def get_piecewise_linearisation_for_streams(streams: List[NonLinearStream], t_h_data: list, dt_diff_max: float) -> np.array:
    """Generate piecewise-linear T-H profiles for each non-linear stream using a tolerance cap."""
    if len(streams) != len(t_h_data):
        raise ValueError(f"Piecewise linearisation failed due to a different number of streams and temperature-enthalpy datasets.")

    return_data = {"t_h_points": []}

    # Create and Linearize stream
    for index, s in enumerate(streams):
        is_hot_stream = s.t_supply > s.t_target
        curve_points = t_h_data[index]
        mask_points = get_piecewise_data_points(
            curve=curve_points, 
            dt_diff_max=dt_diff_max, 
            is_hot_stream=is_hot_stream
        )
        return_data["t_h_points"] = mask_points.tolist()

    return return_data

def get_piecewise_data_points(curve: list, dt_diff_max: float, is_hot_stream:bool) -> np.array:
    """
    Performs piecewise linearisation on a curve using the Ramer-Douglas-Peucker (_rdp) algorithm.

    :param curve: Numpy array of plot points for th curve
    :param dt_diff_max: Maximum allowed temperature differential (dt_diff_max, tolerance etc)
    :returns: Numpy array of new curve points
    """
    curve = np.array(curve)
    try:
        return _get_piecewise_breakpoints(
            curve=curve, 
            epsilon=dt_diff_max, 
            is_hot_stream=is_hot_stream
        )
    except:
        try:
            return _rdp(
                curve=curve, 
                epsilon=dt_diff_max,
            )
        except:
            raise ValueError(f"Piecewise linearisation failed.")

#######################################################################################################
# Helper functions
#######################################################################################################

def _rdp(curve: np.array, epsilon: float) -> np.array:
    """
    Linearize and simplify a curve using the Ramer-Douglas-Peucker (_rdp) algorithm.

    :param curve: Array of points (N, 2).
    :param epsilon: Maximum allowed perpendicular distance for simplification. Can be considered the deviation tolerance or (dt_diff_max)
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
            distance = abs(np.cross(line_vector, point_vector)) / line_length
            if distance > dmax:
                dmax = distance
                index = i

        # Split if distance > epsilon (dt_diff_max) 
        if dmax > epsilon:
            stack.append([start, index])
            stack.append([index, end])
        else:
            indices[start + 1:end] = False

    return curve[indices]

def _refine_pw_points_for_heating_or_cooling(curve: np.array, pw_points: np.array, eps_lb: float=0.0, hot_stream: bool=True) -> np.array:
    """
    Refines a piecewise linear approximation of a T-h profile to ensure feasibility as a hot or cold stream within eps.
    The refinement is done by optimising the piecewise points to minimise the L2 norm of the difference between the
    data and the piecewise points.

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
        'first_point' : pw_points[0],
        'last_point' : pw_points[-1],
    }

    def delta_pw_and_data(x, args):
        '''Returns the difference between the data and the piecewise points'''
        # Reshape the piecewise points such that the first half are the x values and the second half are the y values
        new_pw_points = np.vstack((args['first_point'], x.reshape(-1, 2), args['last_point']))
        # Interpolate the piecewise points to the original data points
        int_points = np.interp(curve[:, 0], new_pw_points[:, 0], new_pw_points[:, 1])
        # Find the difference between the data and the piecewise points
        return int_points - curve[:, 1]
    
    # Define the constraints for the optimisation
    if hot_stream:
        con = lambda x: np.max(delta_pw_and_data(x, args))
        nlc = NonlinearConstraint(con, -np.inf, eps_lb)
    else:
        con = lambda x: np.min(delta_pw_and_data(x, args))
        nlc = NonlinearConstraint(con, -eps_lb, np.inf)

    def obj(x, args):
        '''Returns the L2 norm of the difference between the data and the piecewise points'''
        return  np.sum(
                    np.square(
                            delta_pw_and_data(x, args)
                        )
                    )
    
    # Perform the optimisation
    res = minimize(fun=obj, x0=x0, constraints=nlc, args=args, method='SLSQP', tol=1e-6) 
    refined_pw_points = np.vstack((args['first_point'], res.x.reshape(-1, 2), args['last_point']))
    return np.flipud(refined_pw_points), np.max(np.abs(delta_pw_and_data(res.x, args)))

def _get_piecewise_breakpoints(curve: np.array, epsilon: float, is_hot_stream: bool=True) -> np.array:
    """
    Get the piecewise breakpoints for a curve using the Ramer-Douglas-Peucker (_rdp) algorithm followed 
    by an optimisation-based refinement to ensure hot and cold stream integrity for Pinch Analysis.

    :param curve: Array of points (N, 2).
    :param epsilon: Maximum allowed perpendicular distance for simplification; the deviation tolerance or delta_t_dev.
    :param hot_stream: True if the stream is hot, False if the stream is cold.
    :returns: Simplified array of breakpoints that define the piecewise linearisation.
    """
    for _ in range(10):
        pw_points = _rdp(curve, epsilon=epsilon)

        if len(pw_points) > 10:
            pw_points, max_err = _refine_pw_points_for_heating_or_cooling(curve, pw_points, epsilon / 10, is_hot_stream)
        else:
            break

        if max_err > epsilon:
            epsilon = epsilon * 0.9
        else:
            break

    return pw_points


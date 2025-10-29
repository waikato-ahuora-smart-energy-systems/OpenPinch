"""Target heat pump integration for given heating or cooler profiles."""

from scipy.optimize import minimize

from ..classes import *
from ..lib import *
from ..utils import *
from .problem_table_analysis import get_utility_heat_cascade, create_problem_table_with_t_int
from .gcc_manipulation import *
from .temperature_driving_force import get_temperature_driving_forces


__all__ = ["get_optimal_heat_pump_placement"]

#######################################################################################################
# Public API
#######################################################################################################


def get_optimal_heat_pump_placement(
    T_vals: np.ndarray,
    T_hot_net: np.ndarray,
    T_cold_net: np.ndarray,
    n_cond: int,
    n_evap: int,
    hot_pinch: float,
    cold_pinch: float,
    eff_isen: float = 0.7,
    dtmin_hp: float = 5.0,
):


    return None


#######################################################################################################
# Helper functions
#######################################################################################################


def _initialise_heat_pump_temperatures(
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    n_cond: int,
    n_evap: int,
    eff_isen: float = 0.7,
    dtmin_hp: float = 5.0,
    is_T_vals_shifted: bool = True,
):
    idx_dict = _get_extreme_temperatures_idx(H_hot, H_cold)
    T_bnds = _convert_idx_to_temperatures(idx_dict, T_vals)
    T_cond_init, T_evap_init = _set_initial_values_for_condenser_and_evaporator(n_cond, n_evap, T_bnds)
    x0, args, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)

    if x0.shape == (0,):
        pass # nothing to optimise
    else:
        res = minimize(
            fun=None,
            x0=x0,
            args=(args),
            method="COBYQA",
            bounds=bnds,
            # constraints=None,
        )

def _prepare_data_for_minimizer(
    T_cond_init: np.ndarray, 
    T_evap_init: np.ndarray,
    T_bnds: dict,
) -> Tuple[list, dict, Tuple]:
    
    args = {
        "T_cond_hi": T_cond_init[0], 
        "T_evap_lo": T_evap_init[0],
    }
    x0 = np.concatenate(
        (T_cond_init[1:], T_evap_init[1:])
    )
    bnds = []
    for _ in range(len(T_cond_init[1:])):
        bnds.append(T_bnds["HU"])
    for _ in range(len(T_evap_init[1:])):
        bnds.append(T_bnds["CU"])
    return x0, args, bnds


def _set_initial_values_for_condenser_and_evaporator(
    n_cond: int,
    n_evap: int,
    T_bnds: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cond = max(1, int(n_cond))
    n_evap = max(1, int(n_evap))

    hot_lo, hot_hi = T_bnds["HU"]
    cold_lo, cold_hi = T_bnds["CU"]

    # Distribute condenser nodes from hottest to coldest limits.
    T_cond_init = np.linspace(hot_hi, hot_lo, n_cond + 1)[:-1]
    # Distribute evaporator nodes from coldest to hottest limits.
    T_evap_init = np.linspace(cold_lo, cold_hi, n_evap + 1)[:-1]

    return T_cond_init, T_evap_init


def _convert_idx_to_temperatures(idx_dict: dict, T_vals: np.ndarray) -> dict:
    return {
        "HU": (T_vals[idx_dict["HU"][0]], T_vals[idx_dict["HU"][1]]),
        "CU": (T_vals[idx_dict["CU"][0]], T_vals[idx_dict["CU"][1]]),
    }


def _get_extreme_temperatures_idx(
    H_hot: np.ndarray,
    H_cold: np.ndarray,        
) -> dict:
    i0 = _get_first_or_last_zero_value_idx(H_cold, True)
    i1 = _get_first_or_last_zero_value_idx(H_cold - H_cold[0], False)
    j0 = _get_first_or_last_zero_value_idx(H_hot - H_hot[-1], True)
    j1 = _get_first_or_last_zero_value_idx(H_hot, False)
    return {
        "HU": (i0, i1),
        "CU": (j0, j1),
    }


def _get_first_or_last_zero_value_idx(x: np.ndarray, find_first_zero: bool = True) -> int:
    mask = np.abs(x) < tol
    zero_hits = np.flatnonzero(mask)
    i = 0 if find_first_zero else -1
    return int(zero_hits[i])


def _optimize():
    minimize(
        fun=None,
        x0=None,
        
    )


def _get_heat_pump_heating_and_cooling_streams() -> StreamCollection:

    hp_hot_streams, hp_cold_streams = StreamCollection(), StreamCollection()

    return hp_hot_streams, hp_cold_streams


def _get_heat_pump_cascade(
    hp_hot_streams: StreamCollection, 
    hp_cold_streams: StreamCollection,
):
    pt: ProblemTable
    pt = create_problem_table_with_t_int(
        hp_hot_streams + hp_cold_streams,
        True,
    )
    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value],
            hp_hot_streams,
            hp_cold_streams,
            is_shifted=True,
        )
    )

    return {
        PT.T.value: pt.col[PT.T.value],
        PT.H_HOT_UT.value: pt.col[PT.H_HOT_UT.value],
        PT.H_COLD_UT.value: pt.col[PT.H_COLD_UT.value],
    }


def _get_min_temperature_approach(
    hp_profile: dict, 
    snk_profile: dict, 
    src_profile: dict
) -> Tuple[float, float]:
    
    hot_side_tdf = get_temperature_driving_forces(
        T_hot=hp_profile[PT.T.value],
        H_hot=hp_profile[PT.H_HOT_UT.value],
        T_cold=snk_profile[PT.T.value],
        H_cold=snk_profile[PT.H_HOT_UT.value],
    )
    min_hot_side_tdf = min(
        hot_side_tdf["delta_T1"].min(), 
        hot_side_tdf["delta_T2"].min(),
    )

    cold_side_tdf = get_temperature_driving_forces(
        T_hot=src_profile[PT.T.value],
        H_hot=src_profile[PT.H_COLD_UT.value],
        T_cold=hp_profile[PT.T.value],
        H_cold=hp_profile[PT.H_COLD_UT.value],
    )
    min_cold_side_tdf = min(
        cold_side_tdf["delta_T1"].min(), 
        cold_side_tdf["delta_T2"].min(),
    )
    return min_hot_side_tdf, min_cold_side_tdf

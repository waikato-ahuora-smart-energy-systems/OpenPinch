"""Target heat pump integration for given heating or cooler profiles."""

from scipy.optimize import minimize, brentq

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
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    n_cond: int,
    n_evap: int,
    eff_isen: float = 0.7,
    dtmin_hp: float = 5.0,
    is_T_vals_shifted: bool = True,
):
    init_res = _initialise_heat_pump_temperatures(
        T_vals,
        H_hot,
        H_cold,
        n_cond,
        n_evap,
        eff_isen,
        dtmin_hp,
        is_T_vals_shifted,        
    )

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
    x0, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)
    args = HeatPumpPlacementArgs(
        T_cond_hi=T_cond_init[0],
        T_evap_lo=T_evap_init[0],
        T_vals=T_vals,
        H_hot=H_hot,
        H_cold=H_cold,
        n_cond=int(n_cond),
        n_evap=int(n_evap),
        bnds_cond=T_bnds["HU"],
        bnds_evap=T_bnds["CU"],
        eff_isen=float(eff_isen),
        dtmin_hp=float(dtmin_hp),
        is_T_vals_shifted=bool(is_T_vals_shifted),
    )

    if x0.shape == (0,):
        res = _get_heat_pump_placement_performance(None, args)
    else:
        res = minimize(
            fun=lambda x: _get_heat_pump_placement_performance(x, args),
            x0=x0,
            method="COBYQA",
            bounds=bnds,
            # constraints=None,
        )
    total_work = res.x
    T_cond, T_evap = _get_all_cond_and_evap_temperature_levels(res.x, args)
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_vals, args.H_cold)
    H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_vals, args.H_hot)

    return {
        "T_cond": T_cond, 
        "H_cond": H_cond, 
        "T_evap": T_evap, 
        "H_evap": H_evap, 
        "total_work": total_work,
    }


def _get_entropic_average_temperature(
    T: np.ndarray,
    H: np.ndarray,        
):
    if T.var() < tol:
        return T[0]
    
    temp = np.roll(H, -1)
    temp[-1] = 0
    Q = H - temp
    S = Q / (T + 273.15)

    if Q.sum() > tol:
        return Q.sum() / S.sum()
    else: 
        return T.mean()
    

def _get_carnot_COP(
    T_cond: np.ndarray,
    H_cond: np.ndarray, 
    T_evap: np.ndarray,
    H_evap: np.ndarray,  
    eff: float = 0.6,           
):  
    T_hi = _get_entropic_average_temperature(T_cond, H_cond)
    T_lo = _get_entropic_average_temperature(T_evap, H_evap * -1)
    if T_hi > T_lo:
        return T_hi / (T_hi - T_lo) * eff
    else:
        return 1000


def _get_all_cond_and_evap_temperature_levels(x, args: HeatPumpPlacementArgs) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n_cond_vars = max(int(args.n_cond) - 1, 0)
    n_evap_vars = max(int(args.n_evap) - 1, 0)
    T_cond = np.concatenate((np.array([args.T_cond_hi]), x[:n_cond_vars]))
    T_evap = np.concatenate((np.array([args.T_evap_lo]), x[n_cond_vars:n_cond_vars + n_evap_vars]))
    return T_cond, T_evap


def _get_H_vals_from_T_hp_vals(
    T_hp: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> np.ndarray:
    return np.interp(T_hp, T_vals[::-1], H_vals[::-1])


def _get_heat_pump_placement_performance(x, args: HeatPumpPlacementArgs) -> float:
    T_cond, T_evap = _get_all_cond_and_evap_temperature_levels(x, args)
    T_evap0 = T_evap.copy()
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_vals, args.H_cold)
    
    def _get_optimal_min_evap_T(T_lo):
        for i in range(len(T_evap)):
            if (T_lo > T_evap[i]) or (abs(T_lo - T_evap[i]) > tol and i == 0):
                T_evap[i] = T_lo
            else:
                T_evap[i] = T_evap0[i]  
        H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_vals, args.H_hot)
        cop = _get_carnot_COP(T_cond, H_cond, T_evap, H_evap)
        Q_evap_tot = H_cond[0] * (1 - 1 / cop) if cop > 0 else 0
        work = H_cond[0] - Q_evap_tot
        T_lo_calc = interp_with_plateaus(args.H_hot[::-1], args.T_vals[::-1], -Q_evap_tot, "right")
        return {
            "work": work, 
            "T_evap": T_evap, 
            "H_evap": H_evap, 
            "cop": cop, 
            "Q_evap_tot": Q_evap_tot, 
            "T_lo_calc": T_lo_calc,
            "error": T_lo - T_lo_calc,
        }     

    res = brentq(
        f=lambda T: _get_optimal_min_evap_T(T)["error"],
        a=args.bnds_evap[0],
        b=args.bnds_evap[1],
        # full_output=True,
    )
    total_work = _get_optimal_min_evap_T(res)["work"]
    return total_work


def _prepare_data_for_minimizer(
    T_cond_init: np.ndarray, 
    T_evap_init: np.ndarray,
    T_bnds: dict,
) -> Tuple[list, dict, Tuple]:
    
    x0 = np.concatenate(
        (T_cond_init[1:], T_evap_init[1:])
    )
    bnds = []
    for _ in range(len(T_cond_init[1:])):
        bnds.append(T_bnds["HU"])
    for _ in range(len(T_evap_init[1:])):
        bnds.append(T_bnds["CU"])
    return x0, bnds


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

def cut_profile(src_profile, H_cut, side="right", profile_type="source"):
    """
    Cut a source or sink profile dict at a given enthalpy H_cut, using plateau-aware interpolation.

    side : {'left', 'right'}
        'right' means after the cut (colder outlet for sources, hotter outlet for sinks)
        'left'  means before the cut (hotter inlet for sources, colder inlet for sinks)
    """
    H_key = [k for k in src_profile.keys() if k != 'T'][0]
    H = np.array(src_profile[H_key], dtype=float)
    T = np.array(src_profile['T'], dtype=float)

    flipped = False
    if H[0] > H[-1]:
        H = H[::-1]
        T = T[::-1]
        flipped = True

    T_cut = interp_with_plateaus(H, T, np.array([H_cut]), side=side)[0]
    idx = np.searchsorted(H, H_cut)

    # Adjust logic based on type
    if profile_type == "source":
        keep_after = side == "right"
    else:  # sink
        keep_after = side == "right"  # 'right' still means after, but physically it's the opposite end

    if keep_after:
        new_H = np.concatenate([[H_cut], H[idx:]])
        new_T = np.concatenate([[T_cut], T[idx:]])
    else:
        new_H = np.concatenate([H[:idx], [H_cut]])
        new_T = np.concatenate([T[:idx], [T_cut]])

    if flipped:
        new_H = new_H[::-1]
        new_T = new_T[::-1]

    return {H_key: new_H, 'T': new_T}

def add_ambient_source(
    src_profile: dict,
    heat_flow: float,
    t_ambient_supply: float = None,
    t_ambient_target: float = None,
) -> dict:
    """
    Add an ambient heat source contribution to an existing source (cold utility) profile.

    This function no longer modifies any StreamCollection. It simply extends the
    given source profile by including an ambient stream defined by the specified
    heat flow and ambient temperatures.

    Parameters
    ----------
    src_profile : dict
        Existing source (cold utility) profile, containing at least:
        - PT.T.value
        - PT.H_COLD_UT.value
    heat_flow : float
        Heat flow of the ambient stream [same units as src_profile].
    t_ambient_supply : float, optional
        Supply temperature of the ambient stream (default: Configuration.T_ENV).
    t_ambient_target : float, optional
        Target temperature of the ambient stream (default: Configuration.T_ENV - 1°C).

    Returns
    -------
    dict
        Updated cold utility profile (same structure as input) that includes
        the ambient source contribution.
    """
    # Default temperatures from configuration
    if t_ambient_supply is None:
        t_ambient_supply = Configuration.T_ENV
    if t_ambient_target is None:
        t_ambient_target = Configuration.T_ENV - 1.0  # small delta

    # Extract current arrays
    T_vals = np.array(src_profile[PT.T.value])
    H_vals = np.array(src_profile[PT.H_COLD_UT.value])

    # Append ambient source at the lower end of temperature and enthalpy
    # Assume the ambient source adds heat_flow starting from t_ambient_target
    # up to t_ambient_supply
    new_T_vals = np.append(T_vals, [t_ambient_target, t_ambient_supply])
    new_H_vals = np.append(H_vals, [H_vals.min()- heat_flow, H_vals.min() ])

    # Sort by temperature to maintain consistent profile order
    sort_idx = np.argsort(new_T_vals)
    new_T_vals = new_T_vals[sort_idx]
    new_H_vals = new_H_vals[sort_idx]

    # Return updated profile
    return {
        PT.T.value: new_T_vals,
        PT.H_COLD_UT.value: new_H_vals,
    }



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



"""Target heat pump integration for given heating or cooler profiles."""

from scipy.optimize import minimize, brentq

from ..lib import *
from ..utils import *
from .problem_table_analysis import get_utility_heat_cascade, create_problem_table_with_t_int
from .gcc_manipulation import *
from .temperature_driving_force import get_temperature_driving_forces
from ..classes.stream import Stream
from ..classes.stream_collection import StreamCollection
from ..classes.problem_table import ProblemTable
from ..classes.simple_heat_pump import HeatPumpCycle

__all__ = ["get_optimal_heat_pump_placement"]


#######################################################################################################
# Public API
#######################################################################################################


def get_optimal_heat_pump_placement(
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    n_cond: int = 1,
    n_evap: int = 1,
    eta_comp: float = 0.7,
    dtmin_hp: float = 5.0,
    is_T_vals_shifted: bool = True,
    zone_config: Configuration = None,
    is_cycle_simulation: bool = False,
    is_heat_pump: bool = True,
):
    """
    Determine an optimal multi-stage heat-pump placement for the supplied heat
    cascade envelopes.

    Args:
        T_hot: Shifted hot composite-curve temperatures (descending °C).
        H_hot: Matching hot composite enthalpy values.
        T_cold: Shifted cold composite-curve temperatures (descending °C).
        H_cold: Matching cold composite enthalpy values.
        n_cond: Number of condenser segments to consider.
        n_evap: Number of evaporator segments to consider.
        eta_comp: Isentropic efficiency estimate for the optimiser.
        dtmin_hp: Minimum allowable approach temperature for the HP.
        is_T_vals_shifted: True when composite curves already include ΔTmin/2
            shifting; False for real temperatures.
        zone_config: Optional zone configuration providing ambient parameters.

    Returns:
        None: The routine is currently a placeholder that normalises inputs and
        delegates to the basic placement helper.
    """
    zone_config.HP_HEATING_FRACTION = 0.6
    ####
    Q_hp_target = zone_config.HP_HEATING_FRACTION * np.abs(H_cold).max()
    if zone_config.HP_HEATING_FRACTION < tol:
        return {}
    T_hot, T_cold, dT_shift = _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(T_vals, dtmin_hp, is_T_vals_shifted)   
    if zone_config.HP_HEATING_FRACTION < 1.0:
        T_cold, H_cold = _get_H_cold_within_target_Q_hp(Q_hp_target, T_cold, H_cold)
    H_hot, H_cold, Q_amb_max = _balance_hot_and_cold_heat_loads_with_ambient_air(T_hot, H_hot, T_cold, H_cold, dT_shift, zone_config)
    idx_dict = _get_extreme_temperatures_idx(H_hot, H_cold)
    T_bnds = _convert_idx_to_temperatures(idx_dict, T_hot, T_cold)
    T_cond_init, T_evap_init = _set_initial_values_for_condenser_and_evaporator(n_cond, n_evap, T_bnds)

    args = HeatPumpPlacementArgs(
        Q_hp_target=Q_hp_target,
        Q_amb_max=Q_amb_max,
        T_cond_hi=T_cond_init[0],
        T_evap_lo=T_evap_init[-1],
        T_cond=T_cond_init,
        T_evap=T_evap_init,
        T_hot=T_hot,
        H_hot=H_hot,
        T_cold=T_cold,
        H_cold=H_cold,
        n_cond=int(n_cond),
        n_evap=int(n_evap),
        T_bnds_cond=T_bnds["HU"],
        T_bnds_evap=T_bnds["CU"],
        eta_comp=float(eta_comp),
        dtmin_hp=float(dtmin_hp),
        is_T_vals_shifted=bool(is_T_vals_shifted),
        is_heat_pump=bool(is_heat_pump),
        refrigerant=zone_config.REFRIGERANTS[0]
    )
        
    _optimise_multi_temperature_carnot_heat_pump_placement(args)
    if 0:
        _plot_multi_hp_profiles(args, False)
    if is_cycle_simulation and 0:
        _optimise_multi_simulated_heat_pump_placement(args)
        if 1:
            _plot_multi_hp_profiles(args, True)        
    else:
        hp_streams = _build_latent_streams(args.T_cond, args.Q_cond, args.dtmin_hp, is_hot=True) + _build_latent_streams(args.T_evap, args.Q_evap, args.dtmin_hp, is_hot=False)


    return {
        "hp_streams": hp_streams,
        "T_cond": args.T_cond,
        "Q_cond": args.Q_cond,
        "T_evap": args.T_evap,
        "Q_evap": args.Q_evap,
        "total_work": args.total_work,
        "COP_ave": args.Q_hp_target / args.total_work,
    }


#######################################################################################################
# Helper functions - primary
#######################################################################################################


def _optimise_multi_temperature_carnot_heat_pump_placement(args: HeatPumpPlacementArgs) -> Tuple[dict, HeatPumpPlacementArgs]:
    """
    Compute baseline condenser/evaporator temperature levels and duties for a
    single heat-pump cascade layout.

    Args:
        T_hot: Shifted hot composite-curve temperatures (descending °C).
        H_hot: Corresponding hot composite enthalpy values.
        T_cold: Shifted cold composite-curve temperatures (descending °C).
        H_cold: Corresponding cold composite enthalpy values.
        n_cond: Number of condenser control points.
        n_evap: Number of evaporator control points.
        eta_comp: Isentropic efficiency used when estimating COP.
        dtmin_hp: Minimum approach temperature for HP integration.
        is_T_vals_shifted: Indicates whether composite curves are shifted.

    Returns:
        dict: Dictionary containing condenser/evaporator temperatures,
        enthalpy samples, and total compressor work.
    """
    x0, bnds = _prepare_data_for_minimizer(args.T_cond, args.T_evap, args.T_bnds_cond, args.T_bnds_evap)

    if x0.shape == (0,):
        x = None
    else:
        res = minimize(
            fun=lambda x: _compute_carnot_heat_pump_system_performance(x, args)["work"],
            x0=x0,
            method="COBYQA",
            bounds=bnds,
        )
        x = res.x

    opt = _compute_carnot_heat_pump_system_performance(x, args)
    args.T_cond, args.T_evap = opt["T_cond"], opt["T_evap"]
    args.H_cond, args.H_evap = opt["H_cond"], opt["H_evap"]
    args.total_work = opt["work"]
    args.Q_cond = _get_Q_from_H(args.H_cond)
    args.Q_evap = _get_Q_from_H(args.H_evap)

    return args


def _optimise_multi_simulated_heat_pump_placement(args: HeatPumpPlacementArgs):
    """
    Optimise a multi-unit heat-pump cascade against composite curve constraints.

    Args:
        T_vals: Temperature grid shared by hot/cold composite curves.
        H_hot: Hot composite cumulative enthalpy profile.
        H_cold: Cold composite cumulative enthalpy profile.
        n_cond: Number of condenser units.
        n_evap: Number of evaporator units.
        dtmin_hp: Minimum temperature approach constraint.
        refrigerant: Working refrigerant for the HP cycles.
        unit_system: Display unit system to pass to HeatPumpCycle.

    Returns:
        scipy.optimize.OptimizeResult: res from the COBYQA optimisation.
    """
    # --- Prepare data
    args.T_bnds_cond, args.T_bnds_evap = _correct_T_bnds_for_refrigerant(args.T_bnds_cond, args.T_bnds_evap, args.refrigerant, args.unit_system)

    # --- Initial variable values and bounds
    x0, bnds = _prepare_data_for_minimizer(args.T_cond, args.T_evap, args.T_bnds_cond, args.T_bnds_evap, args.Q_cond, args.Q_evap, args.Q_hp_target, args.is_multi_temperature_hp)

    # --- Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: _mta_constraint_multi(x, args)},
        {'type': 'ineq', 'fun': lambda x: _min_Q_cond_constraint(x, args)},
        {'type': 'ineq', 'fun': lambda x: _condenser_vs_evaporator_constraint(x, args)},
    ]

    # --- Optimization
    res = minimize(
        fun=lambda x: _compute_simulated_heat_pump_system_performance(x, args),
        x0=x0,
        constraints=constraints,
        method="COBYQA",
        bounds=bnds,
        options={'disp': False, 'maxiter': 1000}
    )

    if res.success:
        args.T_evap, args.dT_sh, args.T_cond, args.dT_sc, args.Q_cond = _parse_simulated_hp_state_temperatures(res.x, args)
    else:
        print("Optimization failed:", res.message)

    return args


#######################################################################################################
# Helper functions - _optimise_multi_temperature_carnot_heat_pump_placement
#######################################################################################################


def _get_Q_from_H(
    H: np.ndarray,
):
    """Convert enthalpy cascade into interval duties.
    """
    temp = np.roll(H, -1)
    temp[-1] = 0
    Q = H - temp
    Q_hx = Q[:-1]
    return Q_hx


def _compute_entropic_average_temperature_in_K(
    T: np.ndarray,
    Q: np.ndarray,        
):
    """Compute the entropic average temperature.
    """
    if T.var() < tol:
        return T[0] + 273.15
    S = Q / (T + 273.15)
    if Q.sum() > tol:
        return Q.sum() / S.sum()
    else: 
        return T.mean() + 273.15
    

def _compute_COP_estimate_from_carnot_limit(
    T_cond: np.ndarray,
    H_cond: np.ndarray, 
    T_evap: np.ndarray,
    H_evap: np.ndarray,  
    eff: float = 0.6,           
):  
    """Estimate COP by scaling the Carnot limit using entropic mean temperatures.
    """
    T_hi = _compute_entropic_average_temperature_in_K(T_cond, _get_Q_from_H(H_cond))
    T_lo = _compute_entropic_average_temperature_in_K(T_evap, _get_Q_from_H(H_evap))
    if T_hi > T_lo:
        return T_hi / (T_hi - T_lo) * eff
    else:
        return 1000


def _parse_carnot_hp_state_temperatures(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n_cond_vars = max(int(args.n_cond) - 1, 0)
    n_evap_vars = max(int(args.n_evap) - 1, 0)
    T_cond = np.concatenate((np.array([args.T_cond_hi]), x[:n_cond_vars]))
    T_evap = np.concatenate((x[n_cond_vars:n_cond_vars + n_evap_vars], np.array([args.T_evap_lo])))
    return T_cond, T_evap


def _get_H_vals_from_T_hp_vals(
    T_hp: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    is_cond: bool = True,
) -> np.ndarray:
    """Interpolate the cascade at a specified temperature to find the corresponding heat flow.
    """
    H_less_origin = np.interp(T_hp, T_vals[::-1], H_vals[::-1])
    H = np.concatenate((H_less_origin, np.array([0.0]))) if is_cond else np.concatenate((np.array([0.0]), H_less_origin))
    return H


def _compute_carnot_heat_pump_system_performance(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs
) -> float:
    """Evaluate compressor work for a candidate HP placement defined by vector `x`.
    """
    T_cond, T_evap = _parse_carnot_hp_state_temperatures(x, args)
    T_evap0 = T_evap.copy()
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold, True)
    H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot, False)

    def _get_optimal_min_evap_T(T_lo):
        """Adjust evaporator ladder to honour minimum temperature limits."""
        for i in range(len(T_evap)):
            if (T_lo > T_evap[i]) or (abs(T_lo - T_evap[i]) > tol and i == len(T_evap) - 1):
                T_evap[i] = T_lo
            else:
                T_evap[i] = T_evap0[i]  
        H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot, False)
        cop = _compute_COP_estimate_from_carnot_limit(T_cond, H_cond, T_evap, H_evap)
        work = H_cond[0] / cop if cop > 0 else 0
        Q_evap_tot = H_cond[0] - work
        q_error = Q_evap_tot - (-H_evap[-1])
        return {
            "work": work, 
            "T_evap": T_evap, 
            "H_evap": H_evap, 
            "cop": cop, 
            "Q_evap_tot": Q_evap_tot, 
            "x": x,
            "error": q_error,
        }
    
    try:
        res = brentq(
            f=lambda T: _get_optimal_min_evap_T(T)["error"],
            a=args.T_bnds_evap[0],
            b=args.T_bnds_evap[1],
        )
    except: 
        raise ValueError("Failed to determine a feasible Q of the evaporator.")
    
    res = _get_optimal_min_evap_T(res)
    res.update(
        {
            "T_cond": T_cond,
            "H_cond": H_cond,
        }
    )
    return res


def _prepare_data_for_minimizer(
    T_cond: np.ndarray = None, 
    T_evap: np.ndarray = None,
    T_bnds_cond: Tuple = None,
    T_bnds_evap: Tuple = None,
    Q_cond: np.ndarray = None,
    Q_evap: np.ndarray = None,
    Q_max: float = None,
    is_first_level_included: bool = False,
    is_multi_temperature_hp: bool = True,
    is_carnot_cycle: bool = True,
) -> Tuple[list, dict, Tuple]:
    """Build initial guesses and bounds for the condenser/evaporator temperature.
    """
    x_ls = []
    bnds = []
    j0 = None if is_first_level_included else 1
    j1 = None if is_first_level_included else -1

    dT_sc = T_cond - T_bnds_cond[0] if not is_carnot_cycle else None
    dT_sh = T_bnds_evap[1] - T_evap if not is_carnot_cycle else None
    Qe = Q_evap if is_multi_temperature_hp else None

    def _build_lists(candidate, limits, k0, k1):
        if candidate is not None:
            x_ls.append(candidate[k0:k1])
            for _ in range(len(candidate[k0:k1])):
                bnds.append(limits)

    pairs = [
        (T_cond, T_bnds_cond,  j0,    None),
        (dT_sc,  (0.0, None),  None,  None),
        (T_evap, T_bnds_evap,  None,  j1),
        (dT_sh,  (0.0, None),  None,  None),
        (Q_cond, (0.0, Q_max), 1,     None),
        (Qe,     (0.0, Q_max), 1,     None),
    ]

    for candidate, limits, k0, k1 in pairs:
        _build_lists(candidate, limits, k0, k1)

    x0 = np.concatenate(x_ls) 
    return x0, bnds


def _set_initial_values_for_condenser_and_evaporator(
    n_cond: int,
    n_evap: int,
    T_bnds: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate evenly spaced starting points for condenser and evaporator levels.
    """
    n_cond = max(1, int(n_cond))
    n_evap = max(1, int(n_evap))

    hot_lo, hot_hi = T_bnds["HU"]
    cold_lo, cold_hi = T_bnds["CU"]

    # Distribute condenser nodes from hottest to coldest limits.
    T_cond_init = np.linspace(hot_hi, hot_lo, n_cond + 1)[:-1]
    # Distribute evaporator nodes from coldest to hottest limits.
    T_evap_init = np.linspace(cold_hi, cold_lo, n_evap + 1)[1:]

    return T_cond_init, T_evap_init


def _convert_idx_to_temperatures(
    idx_dict: dict, 
    T_hot: np.ndarray, 
    T_cold: np.ndarray,
) -> dict:
    """Translate extreme enthalpy indices into temperature values.
    """
    return {
        "HU": (T_cold[idx_dict["HU"][0]], T_cold[idx_dict["HU"][1]]),
        "CU": (T_hot[idx_dict["CU"][0]], T_hot[idx_dict["CU"][1]]),
    }


def _get_extreme_temperatures_idx(
    H_hot: np.ndarray,
    H_cold: np.ndarray,        
) -> dict:
    """Locate index pairs describing the active composite curve start and end points.
    """
    i0 = _get_first_or_last_zero_value_idx(H_cold, True)
    i1 = _get_first_or_last_zero_value_idx(H_cold - H_cold[0], False)
    j0 = _get_first_or_last_zero_value_idx(H_hot - H_hot[-1], True)
    j1 = _get_first_or_last_zero_value_idx(H_hot, False)
    return {
        "HU": (i0, i1),
        "CU": (j0, j1),
    }


def _get_first_or_last_zero_value_idx(
    H: np.ndarray, 
    find_first_zero: bool = True
) -> int:
    """Return the cascade index of the first or last element within tolerance of zero.
    """
    mask = np.abs(H) < tol
    zero_hits = np.flatnonzero(mask)
    i = 0 if find_first_zero else -1
    return int(zero_hits[i])


def _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
    T_vals: np.ndarray, 
    dtmin_hp: float,
    is_T_vals_shifted: bool,
):
    """Apply ΔTmin adjustments to cascade temperatures for heat pump calculations.
    """
    dT_shift = dtmin_hp / 2 if is_T_vals_shifted else dtmin_hp
    T_hot  = T_vals - dT_shift
    T_cold = T_vals + dT_shift
    return T_hot, T_cold, dT_shift


def _get_H_cold_within_target_Q_hp(
    Q_hp_target: float,        
    T_cold: np.ndarray,    
    H_cold: np.ndarray,        
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim the cold composite to the target heat duty, interpolating the boundary point."""
    
    if np.abs(H_cold).max() == Q_hp_target:
        return T_cold, H_cold
    mask = np.where(
        H_cold >= Q_hp_target,
        1.0, 0.0
    )
    i = np.flatnonzero(mask)[-1]
    T_cold[i] = linear_interpolation(
        Q_hp_target, H_cold[i], H_cold[i + 1], T_cold[i], T_cold[i + 1]
    )
    H_cold[i] = Q_hp_target
    return T_cold[i:], H_cold[i:]


def _balance_hot_and_cold_heat_loads_with_ambient_air(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray,
    dT_shift: float,
    zone_config: Configuration,   
    is_heat_pump: bool = True,    
):
    """Balance net hot and cold duties using an ambient sink/source if required.
    """
    delta_H = np.abs(H_cold).max() - np.abs(H_hot).max()
    if zone_config == None or np.abs(delta_H) < tol:
        return H_hot, H_cold
    
    if H_hot.max() > tol:
        H_hot *= -1

    if H_cold.min() < -tol:
        H_cold *= -1
    
    if delta_H > tol and is_heat_pump:
        # Heat pump with potentially limited sources
        mask = np.where(
            T_hot <= (zone_config.T_ENV - zone_config.DT_ENV_CONT - zone_config.DT_PHASE_CHANGE - dT_shift),
            delta_H, 0.0
        )
        H_hot -= mask

    elif delta_H < -tol and not is_heat_pump:
        # Refrigeration with potentially limited sinks
        mask = np.where(
            T_cold >= (zone_config.T_ENV + zone_config.DT_ENV_CONT + zone_config.DT_PHASE_CHANGE + dT_shift),
            delta_H, 0.0
        )
        H_cold += mask    

    return H_hot, H_cold, delta_H


#######################################################################################################
# Helper functions for _optimise_multi_simulated_heat_pump_placement
#######################################################################################################


def _cut_cascade_profile(
    src_profile: np.ndarray, 
    H_cut: float,
    is_source_profile: bool = True,
):
    """Cut a source or sink profile dict at a given enthalpy H_cut, using plateau-aware interpolation.
    """
    H_key = [k for k in src_profile.keys() if k != 'T'][0]
    H = np.array(src_profile[H_key], dtype=float)
    T = np.array(src_profile['T'], dtype=float)

    flipped = False
    if H[0] > H[-1]:
        H = H[::-1]
        T = T[::-1]
        flipped = True

    if is_source_profile:
        side="right"
        profile_type="source"
    else:
        side="left"
        profile_type="sink"              
    
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


def _add_ambient_source(
    src_profile: dict,
    heat_flow: float,
    t_ambient_supply: float = None,
    t_ambient_target: float = None,
) -> dict:
    """Add an ambient heat source contribution to an existing source (cold utility) profile.
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


def _get_heat_pump_cascade(
    hp_hot_streams: StreamCollection, 
    hp_cold_streams: StreamCollection,
):
    """Construct a problem table-based cascade from HP condenser/evaporator streams.
    """
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


def _compute_min_temperature_approach(
    hp_profile: dict, 
    snk_profile: dict, 
    src_profile: dict
) -> Tuple[float, float]:
    """Compute the minimum temperature approach on both condenser and evaporator sides.
    """
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


# ============================================================
# ─── SUPPORT FUNCTIONS ──────────────────────────────────────
# ============================================================

def _correct_T_bnds_for_refrigerant(
    T_bnds_cond: tuple,
    T_bnds_evap: tuple,
    refrigerant: str, 
    unit_system: str ='EUR'
):
    """Return min and critical temperatures of a refrigerant.
    """
    T_min, T_crit = PropsSI('Tmin', refrigerant), PropsSI('Tcrit', refrigerant)
    if unit_system == 'EUR':  # Convert to °C if needed
        T_min -= 273.15
        T_crit -= 273.15
    T_bnds_cond = (
        max(T_bnds_cond[0], T_min + 1),
        min(T_bnds_cond[1], T_crit - 1),
    )
    T_bnds_evap = (
        max(T_bnds_evap[0], T_min + 1),
        min(T_bnds_evap[1], T_crit - 1),
    )    
    return T_bnds_cond, T_bnds_evap


def _parse_simulated_hp_state_temperatures(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract HP variables from optimization vector x.
    """
    nc = args.n_cond
    ne = args.n_evap
    n_hp = max(nc, ne)
    
    T_cond = x[:nc]
    i = nc
    dT_sc = x[i:i+nc]
    i += nc
    T_evap = x[i:i+ne][::-1]
    i += ne
    dT_sh = x[i:i+ne]
    i += ne
    Q_vars = x[i:]  # length = n_hp-1
    if len(Q_vars) != n_hp - 1:
        raise ValueError(f"Q_vars length {len(Q_vars)} does not match the number of condnesers and evaporators, {nc}")
    
    Q_cond = np.append(args.Q_hp_target - np.sum(Q_vars), Q_vars)  # highest temperature HP fills remaining heat duty
    return T_evap, dT_sh, T_cond, dT_sc, Q_cond


def _create_heat_pump_list(args: HeatPumpPlacementArgs):
    """Instantiate HeatPumpCycle objects that span the condenser/evaporator ladders.
    """
    hp_list = []
    n_hp = max(args.n_cond, args.n_evap)
    for i in range(n_hp):
        evap_idx = int(i * args.n_evap / n_hp)
        cond_idx = int(i * args.n_cond / n_hp)
        hp = HeatPumpCycle(
            refrigerant=args.refrigerant, 
            unit_system=args.unit_system, 
            Q_total=args.Q_cond[i],
        )
        hp.solve_t_dt(
            T_evap=args.T_evap[evap_idx],
            T_cond=args.T_cond[cond_idx],
            dT_sh=args.dT_sh[evap_idx] if isinstance(args.dT_sh, np.ndarray) else 0.01,
            dT_sc=args.dT_sc[cond_idx] if isinstance(args.dT_sh, np.ndarray) else 0.01,
            eta_comp=args.eta_comp,
            SI=False
        )
        hp_list.append(hp)
    return hp_list


# ============================================================
# ─── OBJECTIVE AND CONSTRAINTS ───────────────────────────────


def _compute_simulated_heat_pump_system_performance(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs
) -> float:
    """Objective: minimize total compressor work for multi-HP configuration."""
    T_evap, dT_sh, T_cond, dT_sc, Q_cond = _parse_simulated_hp_state_temperatures(x, args.n_cond, args.n_evap, args.Q_hp_target)

    # Create HP objects with correct temperature mapping
    hp_list = _create_heat_pump_list(T_evap, dT_sh, T_cond, dT_sc, Q_cond, args.n_cond, args.n_evap, args.eta_comp, args.refrigerant, args.unit_system)

    total_work = 0.0
    for i, hp in enumerate(hp_list):
        try:
            total_work += Q_cond[i] / hp.COP_heating()  # use Q_cond directly
        except ValueError:
            return 1e6  # Penalize infeasible points

    print(Q_cond, T_evap, T_cond, total_work)
    return total_work


def _mta_constraint_multi(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs,
):
    """Ensure minimum temperature approach ≥ dtmin_hp. Inputted composite curves have been shifted to the refrigerant temperature scale.
    """
    T_evap, dT_sh, T_cond, dT_sc, Q_cond = _parse_simulated_hp_state_temperatures(x, args.n_cond, args.n_evap, args.Q_hp_target)
    hp_list = _create_heat_pump_list(T_evap, dT_sh, T_cond, dT_sc, Q_cond, args.n_cond, args.n_evap, args.eta_comp, args.refrigerant, args.unit_system)

    try:
        min_mta = _profiles_crossing_check_multi(args.T_hot, args.H_hot, args.T_cold, args.H_cold, hp_list)
    except ValueError:
        min_mta = -1e6  # Infeasible solution

    return min_mta


def _condenser_vs_evaporator_constraint(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs,
):
    """Ensure T_cond[i] > T_evap[i % n_evap] + margin."""
    T_evap, _, T_cond, dT_sc, _ = _parse_simulated_hp_state_temperatures(x, args)
    return T_cond - dT_sc - np.array([T_evap[i % args.n_evap] for i in range(args.n_cond)]) - args.dtmin_hp


def _min_Q_cond_constraint(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs,
):
    """Ensure individual condenser duties remain above a minimum fraction of total.
    """
    _, _, _, _, Q_cond_hi = _parse_simulated_hp_state_temperatures(x, args)
    return Q_cond_hi - 0.05 * args.Q_hp_target  # elementwise, must be >= 0


def _profiles_crossing_check_multi(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,
    H_cold: np.ndarray,
    hp_list: list,
):
    """Check minimum temperature approach (MTA) between source/sink profiles and a multi-heat-pump cascade.
    """
    # --- Define source and sink profiles with correct PT keys
    snk_profile = {PT.T.value: T_hot, PT.H_HOT_UT.value: H_hot}
    src_profile = {PT.T.value: T_cold, PT.H_COLD_UT.value: H_cold}

    # --- Build condenser and evaporator profiles for all HPs
    cond_streams_all, evap_streams_all = StreamCollection(), StreamCollection()
    hp: HeatPumpCycle
    for hp in hp_list:     
        cond_streams_all.add_many(
            hp.build_stream_collection(include_cond=True)
        )
        evap_streams_all.add_many(
            hp.build_stream_collection(include_evap=True)
        )

    # --- Merge streams into single HP cascade
    cascade = _get_heat_pump_cascade(
        hp_hot_streams=cond_streams_all,
        hp_cold_streams=evap_streams_all,
    )

    # --- Align source profile to cascade
    try:
        if min(src_profile[PT.H_COLD_UT.value]) < min(cascade[PT.H_COLD_UT.value]):
            src_profile_cut = _cut_cascade_profile(
                src_profile, min(cascade[PT.H_COLD_UT.value])
            )
        else:
            ambient_heat_flow = (
                min(src_profile[PT.H_COLD_UT.value]) - min(cascade[PT.H_COLD_UT.value])
            )
            src_profile_cut = _add_ambient_source(src_profile, ambient_heat_flow)
    except Exception:
        return -1e6  # Unbalanced composite curves or data mismatch

    # --- Compute minimum temperature approach
    try:
        deltaTs = _compute_min_temperature_approach(cascade, snk_profile, src_profile_cut)
        return min(deltaTs)
    except ValueError:
        return -1e6  # Infeasible temperature overlap


def _build_latent_streams(
    T_hp: np.ndarray, 
    Q_hp: np.ndarray,
    dt_min: float,
    is_hot: bool
) -> StreamCollection: 
    sc = StreamCollection()
    for i in range(len(Q_hp)):
        sc.add(
            Stream(
                name=f"HP_H{i + 1}" if is_hot else f"HP_C{i + 1}",
                t_supply=T_hp[i],
                t_target=T_hp[i] - 0.01 if is_hot else T_hp[i] + 0.01,
                heat_flow=Q_hp[i],
                dt_cont=dt_min * 1.5, # Shift to intermediate process temperature scale
                is_process_stream=False,
            )            
        )
    return sc


# ============================================================
# ─── VISUALIZATION ──────────────────────────────────────────
# ============================================================

def _plot_multi_hp_profiles(args: HeatPumpPlacementArgs, is_simulated_hp: bool = True):
    """Plot HP cascade and source/sink profiles.
    """
    T_hot, H_hot = clean_composite_curve_ends(args.T_hot, args.H_hot)
    T_cold, H_cold = clean_composite_curve_ends(args.T_cold, args.H_cold)
    if is_simulated_hp:
        hp_list = _create_heat_pump_list(args.T_evap, args.dT_sh, args.T_cond, args.dT_sc, args.Q_cond, args.n_cond, args.n_evap, args.refrigerant, args.unit_system)

        condenser_streams, evaporator_streams = [], []
        hp: HeatPumpCycle
        for hp in hp_list:
            condenser_streams += hp.build_stream_collection(include_cond=True)
            evaporator_streams += hp.build_stream_collection(include_evap=True)

        cascade = _get_heat_pump_cascade(hp_hot_streams=condenser_streams, hp_cold_streams=evaporator_streams)
        T_hp_hot = cascade[PT.T.value]
        T_hp_cold = cascade[PT.T.value]
        H_hp_hot = cascade[PT.H_HOT_UT.value]
        H_hp_cold = cascade[PT.H_COLD_UT.value]
    else:
        T_hp_hot =[]
        H_hp_hot = []        
        H = args.Q_hp_target
        for i in range(len(args.Q_cond)):
            T_hp_hot.append(args.T_cond[i])
            H_hp_hot.append(H)
            H -= args.Q_cond[i]
            T_hp_hot.append(args.T_cond[i])
            H_hp_hot.append(H)

        T_hp_cold =[]
        H_hp_cold = []
        H = 0
        for i in range(len(args.Q_evap)):
            T_hp_cold.append(args.T_evap[i])
            H_hp_cold.append(H)
            H -= args.Q_evap[i]
            T_hp_cold.append(args.T_evap[i])
            H_hp_cold.append(H)

    plt.figure(figsize=(7, 5))
    plt.plot(H_hp_hot, T_hp_hot, "--", color="darkred", linewidth=1.8, label="Condenser")
    plt.plot(H_hp_cold, T_hp_cold, "--", color="darkblue", linewidth=1.8, label="Evaporator")
    plt.plot(H_hot, T_hot, label="Sink", linewidth=2, color="red")
    plt.plot(H_cold, T_cold, label="Source", linewidth=2, color="blue")
    plt.xlabel("Heat Flow / kW")
    plt.ylabel("Temperature / °C")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axvline(0.0, color='black', linewidth=2)
    plt.tight_layout()
    plt.show()

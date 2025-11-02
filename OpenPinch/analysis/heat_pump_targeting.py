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
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,
    H_cold: np.ndarray,
    n_cond: int = 1,
    n_evap: int = 1,
    eff_isen: float = 0.7,
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
        eff_isen: Isentropic efficiency estimate for the optimiser.
        dtmin_hp: Minimum allowable approach temperature for the HP.
        is_T_vals_shifted: True when composite curves already include ΔTmin/2
            shifting; False for real temperatures.
        zone_config: Optional zone configuration providing ambient parameters.

    Returns:
        None: The routine is currently a placeholder that normalises inputs and
        delegates to the basic placement helper.
    """
    res, T_bnds = _optimise_multi_carnot_heat_pump_placement(
        T_hot,
        H_hot,
        T_cold,
        H_cold,
        n_cond,
        n_evap,
        eff_isen,
        dtmin_hp,
        is_T_vals_shifted,
        zone_config,
        is_heat_pump,
    )
    if is_cycle_simulation:
        res = _optimise_multi_simulated_heat_pump_placement(
            T_hot,
            H_hot,
            T_cold,    
            H_cold,
            n_cond,
            res["T_cond"],
            res["Q_cond"],
            n_evap,
            res["T_evap"],
            res["Q_evap"],
            T_bnds,
            eff_isen,
            dtmin_hp,       
            'Ammonia', 
            "EUR",
            is_T_vals_shifted,
            zone_config,
            is_heat_pump,   
        )
    return res


#######################################################################################################
# Helper functions - primary
#######################################################################################################


def _optimise_multi_carnot_heat_pump_placement(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray,
    n_cond: int,
    n_evap: int,
    eff_isen: float = 0.7,
    dtmin_hp: float = 5.0,
    is_T_vals_shifted: bool = True,
    zone_config: Configuration = None,
    is_heat_pump: bool = True,
):
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
        eff_isen: Isentropic efficiency used when estimating COP.
        dtmin_hp: Minimum approach temperature for HP integration.
        is_T_vals_shifted: Indicates whether composite curves are shifted.

    Returns:
        dict: Dictionary containing condenser/evaporator temperatures,
        enthalpy samples, and total compressor work.
    """
    H_hot, H_cold = _balance_hot_and_cold_heat_loads_with_ambient_air(T_hot, H_hot, T_cold, H_cold, zone_config, is_heat_pump)
    T_hot, T_cold = _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(T_hot, T_cold, dtmin_hp, is_T_vals_shifted)   
    idx_dict = _get_extreme_temperatures_idx(H_hot, H_cold)
    T_bnds = _convert_idx_to_temperatures(idx_dict, T_hot, T_cold)
    T_cond_init, T_evap_init = _set_initial_values_for_condenser_and_evaporator(n_cond, n_evap, T_bnds)
    x0, bnds = _prepare_data_for_minimizer(T_cond_init, T_evap_init, T_bnds)
    args = HeatPumpPlacementArgs(
        T_cond_hi=T_cond_init[0],
        T_evap_lo=T_evap_init[0],
        T_hot=T_hot,
        H_hot=H_hot,
        T_cold=T_cold,
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
        var_x = None
    else:
        res = minimize(
            fun=lambda x: _compute_carnot_heat_pump_system_performance(x, args),
            x0=x0,
            method="COBYQA",
            bounds=bnds,
            # constraints=None,
        )
        var_x = res.x
        
    total_work = _compute_carnot_heat_pump_system_performance(var_x, args)
    T_cond, T_evap = _compile_all_cond_and_evap_temperature_levels(var_x, args)
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold)
    H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot)

    return {
        "T_cond": T_cond,
        "Q_cond": _get_Q_from_H(H_cond),
        "T_evap": T_evap,  
        "Q_evap": _get_Q_from_H(H_evap),
        "total_hp_work": total_work,
    }, T_bnds


def _optimise_multi_simulated_heat_pump_placement(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray,
    n_cond: int,
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    n_evap: int,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    T_bnds: dict,
    eff_isen: float = 0.7,
    dtmin_hp: float = 5.0,       
    refrigerant: str ='Ammonia', 
    unit_system: str ='EUR',
    is_T_vals_shifted: bool = True,
    zone_config: Configuration = None,    
    is_heat_pump: bool = True,
):
    """
    Optimise a multi-unit heat-pump cascade against composite curve constraints.

    Args:
        T_vals: Temperature grid shared by hot/cold composite curves.
        H_hot: Hot composite cumulative enthalpy profile.
        H_cold: Cold composite cumulative enthalpy profile.
        n_cond: Number of condenser units.
        n_evap: Number of evaporator units.
        dtmin_hp: Minimum temperature approach constraint.
        fluid: Working fluid for the HP cycles.
        unit_system: Display unit system to pass to HeatPumpCycle.

    Returns:
        scipy.optimize.OptimizeResult: res from the COBYQA optimisation.
    """
    # --- Prepare data
    H_hot, H_cold = _balance_hot_and_cold_heat_loads_with_ambient_air(T_hot, H_hot, T_cold, H_cold, zone_config)
    T_hot, T_cold = _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(T_hot, T_cold, dtmin_hp, is_T_vals_shifted)     

    Q_total = H_hot.max()
    T_bnds = _correct_T_bnds_for_refrigerant(refrigerant, unit_system)

    # --- Initial variable values and bounds
    x0, bnds = _prepare_data_for_minimizer(
        T_cond, 
        T_evap,
        T_bnds,
        Q_cond,
        Q_evap,
        Q_total,
        True,
    )

    # --- Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: _mta_constraint_multi(x, n_cond, n_evap, T_hot, H_hot, T_cold, H_cold, dtmin_hp, refrigerant, unit_system)},
        {'type': 'ineq', 'fun': lambda x: _min_Q_cond_constraint(x, n_cond, n_evap, Q_total)},
        {'type': 'ineq', 'fun': lambda x: _condenser_vs_evaporator_constraint(x, n_cond, n_evap)},
    ]

    # --- Build arguments to pass into minimizer
    args = HeatPumpPlacementArgs(
        T_hot=T_hot,
        H_hot=H_hot,
        T_cold=T_cold,
        H_cold=H_cold,
        n_cond=int(n_cond),
        n_evap=int(n_evap),
        eff_isen=float(eff_isen),
        dtmin_hp=float(dtmin_hp),
        refrigerant=refrigerant,
        unit_system=unit_system,
    )    

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
        Te, dT_sh, Tc, dT_sc, Q_cond = _parse_hp_variables(res.x, n_cond, n_evap, Q_total)
        print(f"\n--- Optimization Results ---")
        print(f"Te       = {Te}")
        print(f"dT_sh    = {dT_sh}")
        print(f"Tc       = {Tc}")
        print(f"dT_sc    = {dT_sc}")
        print(f"Q_cond   = {Q_cond}")
        print(f"Total W  = {res.fun:.3f}")
    else:
        print("Optimization failed:", res.message)

    return res


#######################################################################################################
# Helper functions - _optimise_multi_carnot_heat_pump_placement
#######################################################################################################


def _get_Q_from_H(H: np.ndarray):
    """Convert enthalpy cascade into interval duties.
    """
    H0 = H * -1 if H.min() < -tol else H
    temp = np.roll(H0, -1)
    temp[-1] = 0
    Q = H0 - temp
    return Q[:-1]


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


def _compile_all_cond_and_evap_temperature_levels(var_T, args: HeatPumpPlacementArgs) -> Tuple[np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels.
    """
    var_T = np.asarray(var_T, dtype=float).reshape(-1)
    n_cond_vars = max(int(args.n_cond) - 1, 0)
    n_evap_vars = max(int(args.n_evap) - 1, 0)
    T_cond = np.concatenate((np.array([args.T_cond_hi]), var_T[:n_cond_vars]))
    T_evap = np.concatenate((np.array([args.T_evap_lo]), var_T[n_cond_vars:n_cond_vars + n_evap_vars]))
    return T_cond, T_evap


def _get_H_vals_from_T_hp_vals(
    T_hp: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> np.ndarray:
    """Interpolate the cascade at a specified temperature to find the corresponding heat flow.
    """
    H_less_origin = np.interp(T_hp, T_vals[::-1], H_vals[::-1])
    H = np.concatenate((H_less_origin, np.array([0.0])))
    return H


def _compute_carnot_heat_pump_system_performance(var_T, args: HeatPumpPlacementArgs) -> float:
    """Evaluate compressor work for a candidate HP placement defined by `var_T`.
    """
    T_cond, T_evap = _compile_all_cond_and_evap_temperature_levels(var_T, args)
    T_evap0 = T_evap.copy()
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold)
    H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot)

    def _get_optimal_min_evap_T(T_lo):
        """Adjust evaporator ladder to honour minimum temperature limits."""
        for i in range(len(T_evap)):
            if (T_lo > T_evap[i]) or (abs(T_lo - T_evap[i]) > tol and i == 0):
                T_evap[i] = T_lo
            else:
                T_evap[i] = T_evap0[i]  
        H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot)
        cop = _compute_COP_estimate_from_carnot_limit(T_cond, H_cond, T_evap, H_evap)
        Q_evap_tot = H_cond[0] * (1 - 1 / cop) if cop > 0 else 0
        work = H_cond[0] - Q_evap_tot
        T_lo_calc = interp_with_plateaus(args.H_hot[::-1], args.T_hot[::-1], -Q_evap_tot, "right")
        return {
            "work": work, 
            "T_evap": T_evap, 
            "H_evap": H_evap, 
            "cop": cop, 
            "Q_evap_tot": Q_evap_tot, 
            "T_lo_calc": T_lo_calc,
            "error": T_lo - T_lo_calc,
        }
    
    try:
        res = brentq(
            f=lambda T: _get_optimal_min_evap_T(T)["error"],
            a=args.bnds_evap[0],
            b=args.bnds_evap[1],
        )
    except: 
        if abs(args.bnds_evap[0] - args.bnds_evap[1]) < 0.1:
            res = args.bnds_evap[0]
        else:
            return 0.0
    
    total_work = _get_optimal_min_evap_T(res)["work"]
    return total_work


def _prepare_data_for_minimizer(
    T_cond: np.ndarray, 
    T_evap: np.ndarray,
    T_bnds: dict,
    Q_cond: np.ndarray = None,
    Q_evap: np.ndarray = None,
    Q_max: float = None,
    is_first_level_included: bool = False,
) -> Tuple[list, dict, Tuple]:
    """Build initial guesses and bounds for the condenser/evaporator temperature.
    """
    x_ls = []
    bnds = []
    j = 0 if is_first_level_included else 1

    def _build_lists(candidate, limits):
        if candidate is not None:
            x_ls.append(candidate[j:])
            for _ in range(len(candidate[j:])):
                bnds.append(limits)

    pairs = [
        (T_cond, T_bnds["HU"]),
        (T_evap, T_bnds["CU"]),
        (Q_cond, (0, Q_max)),
        (Q_evap, (0, Q_max)),
    ]
    for candidate, limits in pairs:
        _build_lists(candidate, limits)

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
    T_evap_init = np.linspace(cold_lo, cold_hi, n_evap + 1)[:-1]

    return T_cond_init, T_evap_init


def _convert_idx_to_temperatures(idx_dict: dict, T_hot: np.ndarray, T_cold: np.ndarray) -> dict:
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


def _get_first_or_last_zero_value_idx(H: np.ndarray, find_first_zero: bool = True) -> int:
    """Return the cascade index of the first or last element within tolerance of zero.
    """
    mask = np.abs(H) < tol
    zero_hits = np.flatnonzero(mask)
    i = 0 if find_first_zero else -1
    return int(zero_hits[i])


def _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
    T_hot: np.ndarray,    
    T_cold: np.ndarray, 
    dtmin_hp: float,
    is_T_vals_shifted: bool,
):
    """Apply ΔTmin adjustments to cascade temperatures for heat pump calculations.
    """
    T_hot  -= dtmin_hp / 2 if is_T_vals_shifted else dtmin_hp
    T_cold += dtmin_hp / 2 if is_T_vals_shifted else dtmin_hp

    return T_hot, T_cold


def _balance_hot_and_cold_heat_loads_with_ambient_air(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray, 
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
            T_hot <= (zone_config.T_ENV - zone_config.DT_ENV_CONT - zone_config.DT_PHASE_CHANGE),
            delta_H, 0.0
        )
        H_hot -= mask

    elif delta_H < -tol and not is_heat_pump:
        # Refrigeration with potentially limited sinks
        mask = np.where(
            T_cold >= (zone_config.T_ENV + zone_config.DT_ENV_CONT + zone_config.DT_PHASE_CHANGE),
            delta_H, 0.0
        )
        H_cold += mask    

    return H_hot, H_cold   


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
    T_bnds: dict, 
    fluid: str, 
    unit_system: str ='EUR'
):
    """Return min and critical temperatures of a fluid.
    """
    T_min, T_crit = PropsSI('Tmin', fluid), PropsSI('Tcrit', fluid)
    if unit_system == 'EUR':  # Convert to °C if needed
        T_min -= 273.15
        T_crit -= 273.15

    for key in T_bnds:
        T_bnds[key] = (
            np.max((T_bnds[key][0], T_min + 1)),
            np.min((T_bnds[key][1], T_crit - 1))
        )

    return T_bnds


def _parse_hp_variables(x, n_cond, n_evap, Q_total):
    """Extract HP variables from optimization vector x.
    """
    N_hp = max(n_cond, n_evap)
    
    Te = x[:n_evap]
    dT_sh = x[n_evap:2*n_evap]
    Tc = x[2*n_evap:2*n_evap+n_cond]
    dT_sc = x[2*n_evap+n_cond:2*n_evap+2*n_cond]
    Q_vars = x[2*n_evap+2*n_cond:]  # length = N_hp-1
    if len(Q_vars) != N_hp - 1:
        raise ValueError(f"Q_vars length {len(Q_vars)} does not match N_hp-1 = {N_hp-1}")
    
    Q_cond = np.append(Q_vars, Q_total - np.sum(Q_vars))  # last HP fills remaining heat duty
    return Te, dT_sh, Tc, dT_sc, Q_cond


def _create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system):
    """Instantiate HeatPumpCycle objects that span the condenser/evaporator ladders.
    """
    hp_list = []
    N_hp = max(n_cond, n_evap)
    for i in range(N_hp):
        evap_idx = int(i * n_evap / N_hp)
        cond_idx = int(i * n_cond / N_hp)
        hp = HeatPumpCycle(fluid_ref=fluid, unit_system=unit_system, Q_total=Q_cond[i])
        hp.solve_t_dt(
            Te=Te[evap_idx],
            Tc=Tc[cond_idx],
            dT_sh=dT_sh[evap_idx],
            dT_sc=dT_sc[cond_idx],
            eta_com=0.7,
            SI=False
        )
        hp_list.append(hp)
    return hp_list


# ============================================================
# ─── OBJECTIVE AND CONSTRAINTS ───────────────────────────────


def _compute_simulated_heat_pump_system_performance(x, n_cond, n_evap, T_vals, H_hot, H_cold, dtmin_hp,
                            fluid='Ammonia', unit_system='EUR'):
    """Objective: minimize total compressor work for multi-HP configuration."""
    Q_total = max(H_hot)
    Te, dT_sh, Tc, dT_sc, Q_cond = _parse_hp_variables(x, n_cond, n_evap, Q_total)

    # Create HP objects with correct temperature mapping
    hp_list = _create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system)

    total_work = 0.0
    for i, hp in enumerate(hp_list):
        try:
            total_work += Q_cond[i] / hp.COP_heating()  # use Q_cond directly
        except ValueError:
            return 1e6  # Penalize infeasible points

    print(Q_cond, Te, Tc, total_work)
    return total_work


def _mta_constraint_multi(x, n_cond, n_evap, T_hot, H_hot, T_cold, H_cold, dtmin_hp,
                         fluid='Ammonia', unit_system='EUR'):
    """Ensure minimum temperature approach ≥ dtmin_hp."""
    Q_total = max(H_hot)
    Te, dT_sh, Tc, dT_sc, Q_cond = _parse_hp_variables(x, n_cond, n_evap, Q_total)
    hp_list = _create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system)

    try:
        min_mta = _profiles_crossing_check_multi(T_hot, H_hot, T_cold, H_cold, hp_list)
    except ValueError:
        min_mta = -1e6  # Infeasible solution

    return min_mta - dtmin_hp


def _condenser_vs_evaporator_constraint(x, n_cond, n_evap):
    """Ensure Tc[i] > Te[i % n_evap] + margin."""
    Te = x[:n_evap]
    Tc = x[2 * n_evap:2 * n_evap + n_cond]
    margin = 3.0
    return Tc - np.array([Te[i % n_evap] for i in range(n_cond)]) - margin


def _min_Q_cond_constraint(x, n_cond, n_evap, Q_total):
    """Ensure individual condenser duties remain above a minimum fraction of total.
    """
    _, _, _, _, Q_cond = _parse_hp_variables(x, n_cond, n_evap, Q_total)
    return Q_cond - 0.05 * Q_total  # elementwise, must be >= 0


def _profiles_crossing_check_multi(T_hot, H_hot, T_cold, H_cold, hp_list):
    """Check minimum temperature approach (MTA) between source/sink profiles and a multi-heat-pump cascade.
    """
    # --- Define source and sink profiles with correct PT keys
    snk_profile = {PT.T.value: T_hot, PT.H_HOT_UT.value: H_hot}
    src_profile = {PT.T.value: T_cold, PT.H_COLD_UT.value: H_cold}

    # --- Build condenser and evaporator profiles for all HPs
    condenser_streams_all, evaporator_streams_all = [], []

    for hp in hp_list:
        condenser_streams_all.extend(
            hp._build_stream_collection(hp._build_condenser_profile(), is_hot=True)
        )
        evaporator_streams_all.extend(
            hp._build_stream_collection(hp._build_evaporator_profile(), is_hot=False)
        )

    # --- Merge streams into single HP cascade
    cascade = _get_heat_pump_cascade(
        hp_hot_streams=condenser_streams_all,
        hp_cold_streams=evaporator_streams_all,
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


# ============================================================
# ─── VISUALIZATION ──────────────────────────────────────────
# ============================================================

def plot_multi_hp_profiles(T_vals, H_hot, H_cold, x, n_cond, n_evap, 
                           fluid='Ammonia', unit_system='EUR'):
    """Plot HP cascade and source/sink profiles.
    """
    Te, dT_sh, Tc, dT_sc, Q_cond = _parse_hp_variables(x, n_cond, n_evap, Q_total=max(H_hot))
    hp_list = _create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system)

    condenser_streams, evaporator_streams = [], []
    for hp in hp_list:
        condenser_streams += hp._build_stream_collection(hp._build_condenser_profile(), is_hot=True)
        evaporator_streams += hp._build_stream_collection(hp._build_evaporator_profile(), is_hot=False)

    cascade = _get_heat_pump_cascade(hp_hot_streams=condenser_streams, hp_cold_streams=evaporator_streams)

    plt.figure(figsize=(7, 5))
    plt.plot(cascade["H_hot_utility"], cascade["T"], "--", color="darkred", linewidth=1.8, label="Condenser")
    plt.plot(cascade["H_cold_utility"], cascade["T"], "--", color="darkblue", linewidth=1.8, label="Evaporator")
    plt.plot(H_hot, T_vals, label="Sink", linewidth=2, color="red")
    plt.plot(H_cold, T_vals, label="Source", linewidth=2, color="blue")
    plt.xlabel("Enthalpy")
    plt.ylabel("Temperature [°C or K]")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axvline(0.0, color='black', linewidth=2)
    plt.tight_layout()
    plt.show()

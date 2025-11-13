"""Target heat pump integration for given heating or cooler profiles."""

from scipy.optimize import minimize, NonlinearConstraint, differential_evolution

from ..lib import *
from ..utils import *
from .problem_table_analysis import *
from .gcc_manipulation import *
from .temperature_driving_force import *
from ..classes.stream import Stream
from ..classes.stream_collection import StreamCollection
from ..classes.problem_table import ProblemTable
from ..classes.simple_heat_pump import HeatPumpCycle

__all__ = ["get_heat_pump_targets"]


#######################################################################################################
# Public API
#######################################################################################################


def get_heat_pump_targets(
    pt: ProblemTable, 
    zone_config: Configuration,
    is_T_vals_shifted: bool,
    is_process_integrated: bool,
    is_heat_pumping: bool,
    n_cond: int = 2,
    n_evap: int = 2,
    eta_comp: float = 0.7,
    dtmin_hp: float = 0.0,    
):
    ######### TEMP VARS ######### 
    zone_config.HP_LOAD_FRACTION = 1
    zone_config.DO_HP_SIM = False
    #############################
    res = {}
    if (
        (zone_config.DO_PROCESS_HP_TARGETING == False)
        or 
        (np.abs(pt.col[PT.H_NET_COLD.value]).max() < tol and is_heat_pumping == True)
        or
        (np.abs(pt.col[PT.H_NET_HOT.value]).max() < tol and is_heat_pumping == False)
    ):
        return res
    
    T_vals=pt.col[PT.T.value]
    if is_process_integrated:
        H_hot=pt.col[PT.H_NET_HOT.value]
        H_cold=pt.col[PT.H_NET_COLD.value]                
    else:
        H_hot=pt.col[PT.H_HOT_UT.value]
        H_cold=pt.col[PT.H_COLD_UT.value]

    res.update(
        _get_optimal_heat_pump_placement(
            T_vals=T_vals,
            H_hot=H_hot,
            H_cold=H_cold,
            n_cond=n_cond,
            n_evap=n_evap,
            eta_comp=eta_comp,
            dtmin_hp=dtmin_hp,
            is_process_integrated=is_process_integrated,
            load_fraction=zone_config.HP_LOAD_FRACTION,
            is_cycle_simulation=zone_config.DO_HP_SIM,
            T_env=zone_config.T_ENV,
            dT_env_cont=zone_config.DT_ENV_CONT ,
            dt_phase_change=zone_config.DT_PHASE_CHANGE,
            is_heat_pumping=is_heat_pumping,
            refrigerant=zone_config.REFRIGERANTS
        )
    )
    t_hp = create_problem_table_with_t_int(
        streams=res["cond_streams"] + res["evap_streams"], 
        is_shifted=is_T_vals_shifted
    )
    pt.insert_temperature_interval(
        t_hp[PT.T.value].to_list()
    )
    temp = get_utility_heat_cascade(
        T_int_vals=pt.col[PT.T.value],
        hot_utilities=res["cond_streams"],
        cold_utilities=res["evap_streams"],
        is_shifted=is_T_vals_shifted,
    )
    col_H_net = PT.H_NET_HP_PRO.value if is_process_integrated else PT.H_NET_HP_UT.value  
    pt.update(  
        {
            col_H_net: temp[PT.H_NET_UT.value],
            PT.H_HOT_HP.value: temp[PT.H_HOT_UT.value],
            PT.H_COLD_HP.value: temp[PT.H_COLD_UT.value],
        }
    )
    if 0:
        _plot_multi_hp_profiles(
            T_hot=pt.col[PT.T.value],
            H_hot=pt.col[PT.H_NET_HOT.value],
            T_cold=pt.col[PT.T.value],
            H_cold=pt.col[PT.H_NET_COLD.value],
            T_hp_hot=pt.col[PT.T.value],
            H_hp_hot=pt.col[PT.H_HOT_HP.value],
            T_hp_cold=pt.col[PT.T.value],
            H_hp_cold=pt.col[PT.H_COLD_HP.value],      
        )
    return res


#######################################################################################################
# Helper functions - primary
#######################################################################################################


def _get_optimal_heat_pump_placement(
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    n_cond: int = 1,
    n_evap: int = 1,
    eta_comp: float = 0.7,
    dtmin_hp: float = 0.0,
    is_process_integrated: bool = True,
    load_fraction: float = 1,
    is_cycle_simulation: bool = False,
    refrigerant: list = [],
    T_env: float = 15,
    dT_env_cont: float = 5,
    dt_phase_change: float = 0.01,
    is_heat_pumping: bool = True,  
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
    if load_fraction < tol:
        return {}
        
    T_vals, H_hot, H_cold = T_vals.copy(), H_hot.copy(), H_cold.copy()

    T_hot, T_cold, dtcont_hp = _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(T_vals, dtmin_hp, is_process_integrated)   
    Q_hp_target = min(load_fraction, 1.0) * np.abs(H_cold).max()
    T_cold, H_cold = _get_H_col_till_target_Q(Q_hp_target, T_cold, H_cold)
    T_hot, H_hot, T_cold, H_cold, Q_amb_max = _balance_hot_and_cold_heat_loads_with_ambient_air(
        T_hot=T_hot, 
        H_hot=H_hot, 
        T_cold=T_cold, 
        H_cold=H_cold, 
        dtcont=dtcont_hp, 
        T_env=T_env, 
        dT_env_cont=dT_env_cont, 
        dt_phase_change=dt_phase_change, 
        is_heat_pumping=is_heat_pumping,
    )
    T_hot, H_hot = clean_composite_curve(T_hot, H_hot)
    T_cold, H_cold = clean_composite_curve(T_cold, H_cold)
    T_cond_init, T_evap_init = _set_initial_values_for_condenser_and_evaporator(
        n_cond, n_evap, T_hot, H_hot, T_cold, H_cold,
    )

    args = HeatPumpPlacementArgs(
        Q_hp_target=Q_hp_target,
        Q_amb_max=Q_amb_max,
        T_cond_init=T_cond_init,
        T_evap_init=T_evap_init,
        T_hot=T_hot,
        H_hot=H_hot,
        T_cold=T_cold,
        H_cold=H_cold,
        n_cond=int(n_cond),
        n_evap=int(n_evap),
        eta_comp=float(eta_comp),
        dtcont_hp=dtcont_hp,
        dt_phase_change=dt_phase_change,
        dt_range_max=max(T_cold[0], T_hot[0]) - min(T_cold[-1], T_hot[-1]),
        is_process_integrated=bool(is_process_integrated),
        is_heat_pump=bool(is_heat_pumping),
        refrigerant=refrigerant[0],
    )
        
    res = _optimise_multi_temperature_carnot_heat_pump_placement(args)
    if is_cycle_simulation:
        res = _optimise_multi_simulated_heat_pump_placement(args)
        res.cond_streams, res.evap_streams = _build_simulated_hps_streams(res.hp_list)

    return dict(res)


def _optimise_multi_temperature_carnot_heat_pump_placement(
    args: HeatPumpPlacementArgs
) -> CarnotHeatPumpResults:
    """Compute baseline condenser/evaporator temperature levels and duties for a single multi-temperature heat-pump layout.
    """

    x_cond = _map_T_to_x_cond(args.T_cond_init, args.T_cold[0], args.dt_range_max)
    x_evap = _map_T_to_x_evap(args.T_evap_init, args.T_hot[-1], args.dt_range_max)
    x0, bnds = _prepare_data_for_minimizer(x_cond, x_evap)
    opt = differential_evolution(
        func=lambda x: _compute_carnot_heat_pump_system_performance(x, args)["obj"],
        x0=x0,
        bounds=bnds,  
        init='sobol'  
    )
    # opt = minimize(
    #     fun=lambda x: _compute_carnot_heat_pump_system_performance(x, args)["obj"],
    #     x0=x0,
    #     method="SLSQP",
    #     bounds=bnds,
    # )
    res = _compute_carnot_heat_pump_system_performance(opt.x, args)
    res["cond_streams"] = _build_latent_streams(res["T_cond"], 0.01, res["Q_cond"], args.dtcont_hp, is_hot=True)
    res["evap_streams"] = _build_latent_streams(res["T_evap"], 0.01, res["Q_evap"], args.dtcont_hp, is_hot=False)
    if 1:
        _plot_multi_hp_profiles_from_results(
            args, CarnotHeatPumpResults.model_validate(res)
        )    
    return CarnotHeatPumpResults.model_validate(res)


def _optimise_multi_simulated_heat_pump_placement(args: HeatPumpPlacementArgs):
    """Optimise a multi-unit heat-pump cascade against composite curve constraints.
    """
    # --- Prepare data
    args.T_bnds_cond, args.T_bnds_evap = _correct_T_bnds_for_refrigerant(args.T_bnds_cond, args.T_bnds_evap, args.refrigerant, args.unit_system)

    # --- Initial variable values and bounds
    x0, bnds = _prepare_data_for_minimizer(args.T_cond, args.T_evap, args.T_bnds_cond, args.T_bnds_evap, args.Q_cond, args.Q_evap, args.Q_hp_target, args.is_multi_temperature_hp)

    # --- Constraints
    constraints = (
        NonlinearConstraint(
            fun=lambda x: _mta_constraint_multi(x, args),
            lb=0.0,
            ub=np.inf
        ),
        NonlinearConstraint(
            fun=lambda x: _min_Q_cond_constraint(x, args),
            lb=0.0,
            ub=np.inf
        ),
        NonlinearConstraint(
            fun=lambda x: _condenser_vs_evaporator_constraint(x, args),
            lb=0.0,
            ub=np.inf
        ),        
    )
    # constraints = [
    #     {'type': 'ineq', 'fun': lambda x: _mta_constraint_multi(x, args)},
    #     {'type': 'ineq', 'fun': lambda x: _min_Q_cond_constraint(x, args)},
    #     {'type': 'ineq', 'fun': lambda x: _condenser_vs_evaporator_constraint(x, args)},
    # ]

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
        args.hp_list = _create_heat_pump_list(args.T_evap, args.dT_sh, args.T_cond, args.dT_sc, args.Q_cond, args.n_cond, args.n_evap, args.refrigerant, args.unit_system)
    else:
        print("Optimization failed:", res.message)

    return args


#######################################################################################################
# Helper functions - _optimise_multi_temperature_carnot_heat_pump_placement
#######################################################################################################


def _compute_entropic_average_temperature_in_K(
    T: np.ndarray,
    Q: np.ndarray,        
):
    """Compute the entropic average temperature.
    """
    if T.var() < tol:
        return T[0] + 273.15
    S_tot = (Q / (T + 273.15)).sum()
    return Q.sum() / S_tot if S_tot > 0 else (T.mean() + 273.15)
    

def _compute_COP_estimate_from_carnot_limit(
    T_cond: np.ndarray,
    Q_cond: np.ndarray, 
    T_evap: np.ndarray,
    Q_evap: np.ndarray,  
    eff: float = 0.6, 
    min_dt_lift: float = 3.0,          
):  
    """Estimate COP by scaling the Carnot limit using entropic mean temperatures.
    """
    T_hi = _compute_entropic_average_temperature_in_K(T_cond, Q_cond)
    T_lo = _compute_entropic_average_temperature_in_K(T_evap, Q_evap)
    return (
        T_lo / (T_hi - T_lo) * eff + 1 
        if T_hi >= T_lo + min_dt_lift 
        else T_lo / min_dt_lift * eff + 1
    )


def _parse_carnot_hp_state_variables(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs
) -> Tuple[np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n_cond_vars = max(int(args.n_cond) - 1, 0)
    x_cond = np.concatenate((np.array([0.0]), x[:n_cond_vars]))
    x_evap = np.concatenate((x[n_cond_vars:], np.array([0.0])))
    return x_cond, x_evap


def _get_Q_vals_from_T_hp_vals(
    T_hp: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    is_cond: bool = True,
) -> np.ndarray:
    """Interpolate the cascade at a specified temperature to find the corresponding duty for each temperature level.
    """
    H_less_origin = np.interp(T_hp, T_vals[::-1], H_vals[::-1])
    H = np.concatenate((H_less_origin, np.array([0.0]))) if is_cond else np.concatenate((np.array([0.0]), H_less_origin))
    temp = np.roll(H, -1)
    temp[-1] = 0
    Q = H - temp
    Q_hx = Q[:-1]
    return np.where(Q_hx > 0.0, Q_hx, 0.0)    


def _compute_carnot_heat_pump_system_performance(
    x: np.ndarray, 
    args: HeatPumpPlacementArgs
) -> float:
    """Evaluate compressor work for a candidate HP placement defined by vector `x`.
    """
    x_cond, x_evap = _parse_carnot_hp_state_variables(x, args)
    T_cond = _map_x_to_T_cond(x_cond, args.T_cold[0], args.dt_range_max)
    Q_cond = _get_Q_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold, True)
    Qh_tar = args.Q_hp_target

    def _get_optimal_min_evap_T(T_lo):
        """Adjust evaporator ladder to honour minimum temperature limits."""
        T_evap = _map_x_to_T_evap(x_evap, T_lo, args.dt_range_max)
        Q_evap = _get_Q_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot, False)
        Q_evap_max = np.abs(Q_evap.sum())
        cop = _compute_COP_estimate_from_carnot_limit(T_cond, Q_cond, T_evap, Q_evap)

        if Q_evap_max / (cop - 1) > Qh_tar / cop:
            work_hp = Qh_tar / cop
            Q_evap_tot = Qh_tar - work_hp
            work_el = 0.0
            Q_evap = Q_evap * (Q_evap_tot / Q_evap_max)
        else: # Evaporator is limits the heat pump
            work_hp = Q_evap_max / (cop - 1)
            Q_evap_tot = Q_evap_max
            work_el = Qh_tar - Q_evap_tot - work_hp # Direct electric heating

        return {
            "obj": (work_hp + work_el) / args.Q_hp_target, 
            "work": work_hp + work_el, 
            "work_hp": work_hp,  
            "work_el": work_el,
            "T_cond": T_cond,
            "Q_cond": Q_cond,            
            "T_evap": T_evap, 
            "Q_evap": Q_evap, 
            "cop": cop,
        }    

    opt = minimize(
        fun=lambda T: _get_optimal_min_evap_T(T)["obj"],
        method="SLSQP",
        x0=args.T_hot[-1],
        bounds=[(args.T_hot[-1], args.T_hot[0])],
    )
    res = _get_optimal_min_evap_T(opt.x)
    if 0:
        res0 = CarnotHeatPumpResults.model_validate(res)
        res0.cond_streams = _build_latent_streams(res0.T_cond, 0.1, res0.Q_cond, args.dtcont_hp, is_hot=True)
        res0.evap_streams = _build_latent_streams(res0.T_evap, 0.1, res0.Q_evap, args.dtcont_hp, is_hot=False)         
        _plot_multi_hp_profiles_from_results(args, res0, str(f"Work ({x}): {res0.work} = {res0.work_hp} + {res0.work_el}"))
    return res


def _map_T_to_x_cond(
    T: np.ndarray, 
    T_hi: float, 
    deltaT_range: float,
) -> np.ndarray:
    x = []
    for i in range(T.size):
        x.append((T_hi - T[i]) / deltaT_range)
        T_hi = T[i]
    return np.array(x).flatten()


def _map_T_to_x_evap(
    T: np.ndarray, 
    T_lo: float, 
    deltaT_range: float,
) -> np.ndarray:
    T = T[::-1]
    x = []
    for i in range(T.size):
        x.append((T[i] - T_lo) / deltaT_range)
        T_lo = T[i]
    return np.array(x).flatten()[::-1]


def _map_x_to_T_cond(
    x: np.ndarray, 
    T_hi: float, 
    deltaT_range: float,
) -> np.ndarray:
    temp = []
    for i in range(x.size):
        temp.append(T_hi - x[i] * deltaT_range)
        T_hi = temp[-1]
    return np.array(temp).flatten()


def _map_x_to_T_evap(
    x: np.ndarray,
    T_lo: float, 
    deltaT_range: float,
) -> np.ndarray:
    temp = []
    for i in range(x.size):
        temp.append(x[::-1][i] * deltaT_range + T_lo)
        T_lo = temp[-1]
    return np.array(temp).flatten()[::-1]


def _prepare_data_for_minimizer(
    x_cond: np.ndarray = None, 
    x_evap: np.ndarray = None,
    is_first_and_last_levels_vars: bool = False,
    is_multi_temperature_hp: bool = True,
    is_carnot_cycle: bool = True,
) -> Tuple[list, dict, Tuple]:
    """Build initial guesses and bounds for the condenser/evaporator temperature.
    """
    x_ls = []
    bnds = []

    j = (None, None) if is_first_and_last_levels_vars else (1, -1)   
    # dT_sc = T_cond - T_bnds_cond[0] if not is_carnot_cycle else None
    # dT_sh = T_bnds_evap[1] - T_evap if not is_carnot_cycle else None
    # Qe = Q_evap if is_multi_temperature_hp else None

    def _build_lists(candidate, limits, k0, k1):
        if candidate is not None:
            x_ls.append(candidate[k0:k1])
            for _ in range(len(candidate[k0:k1])):
                bnds.append(limits)

    pairs = [
        (x_cond, (0.0, 1.0),  j[0], None),
        (x_evap, (0.0, 1.0),  None, j[1]),
        # (dT_sc,  (0.0, None), None, None),
        # (dT_sh,  (0.0, None), None, None),
        # (Q_cond, (0.0, 1.0),  1,    None),
        # (Qe,     (0.0, 1.0),  1,    None),
    ]
    for candidate, limits, k0, k1 in pairs:
        _build_lists(candidate, limits, k0, k1)
    x0 = np.concatenate(x_ls) 
    return x0, bnds


def _build_hp_section_initial_values(
    n_levels: int,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    is_condenser: bool,
) -> np.ndarray:
    """Return candidate HP temperature levels for a condenser/evaporator section."""
    n_levels = max(1, int(n_levels))
    boundary_idx = 0 if is_condenser else -1
    boundary_temperature = T_vals[boundary_idx]

    areas = np.column_stack(
        (np.abs((boundary_temperature - T_vals) * H_vals), T_vals)
    )
    areas = areas[areas[:, 0].argsort()[::-1]]

    zero_idx = np.flatnonzero(areas[:, 0] == 0)
    if zero_idx.size:
        areas = areas[:zero_idx[0]]

    candidate_count = max(n_levels - 1, 0)
    sorted_candidates = np.sort(areas[:candidate_count, 1])[::-1]
    if sorted_candidates.size > 0:
        idx = np.flatnonzero(np.abs(boundary_temperature - sorted_candidates) > tol)
        if idx.size:
            if is_condenser:
                sorted_candidates = sorted_candidates[idx[0]:]
            else:
                sorted_candidates = sorted_candidates[:idx[-1]]

    if n_levels > sorted_candidates.size:
        default_span = np.linspace(
            T_vals[0], T_vals[-1], n_levels - sorted_candidates.size + 1
        )
        if is_condenser:
            default_span = default_span[:-1]
        else:
            default_span = default_span[1:]
        section_T_vals = np.sort(
            np.concatenate([default_span, sorted_candidates])
        )[::-1]
    else:
        boundary = np.array([boundary_temperature])
        section_T_vals = (
            np.concatenate([boundary, sorted_candidates])
            if is_condenser
            else np.concatenate([sorted_candidates, boundary])
        )
    return section_T_vals


def _set_initial_values_for_condenser_and_evaporator(
    n_cond: int,
    n_evap: int, 
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray,    
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate evenly spaced starting points for condenser and evaporator levels.
    """
    T_cond_init = _build_hp_section_initial_values(
        n_levels=n_cond,
        T_vals=T_cold,
        H_vals=H_cold,
        is_condenser=True,
    )
    T_evap_init = _build_hp_section_initial_values(
        n_levels=n_evap,
        T_vals=T_hot,
        H_vals=H_hot,
        is_condenser=False,
    )
    return T_cond_init, T_evap_init


def _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
    T_vals: np.ndarray, 
    dtmin_hp: float,
    is_process_integrated: bool,
):
    """Apply ΔTmin adjustments to cascade temperatures for heat pump calculations.
    """
    dT_shift = dtmin_hp * 0.5 if is_process_integrated else dtmin_hp * 1.5
    T_hot  = T_vals - dT_shift
    T_cold = T_vals + dT_shift
    return T_hot, T_cold, dT_shift


def _get_H_col_till_target_Q(
    Q_max: float,        
    T_vals: np.ndarray,    
    H_vals: np.ndarray,  
    is_cold: bool = True,      
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim the cold composite to the target heat duty, interpolating the boundary point."""
    if abs(np.abs(H_vals).max() - Q_max) < tol:
        return T_vals, H_vals
    mask = np.where(
        np.abs(H_vals) >= Q_max,
        1.0, 0.0
    )
    i_vals = np.flatnonzero(mask)
    if is_cold:
        i = i_vals[-1]
        T_vals[i] = linear_interpolation(
            Q_max, H_vals[i], H_vals[i + 1], T_vals[i], T_vals[i + 1]
        )      
        H_vals[i] = Q_max  
        return T_vals[i:], H_vals[i:]
    else:
        i = i_vals[0]
        T_vals[i] = linear_interpolation(
            -Q_max, H_vals[i], H_vals[i - 1], T_vals[i], T_vals[i - 1]
        )
        H_vals[i] = -Q_max
        return T_vals[:i+1], H_vals[:i+1]


def _balance_hot_and_cold_heat_loads_with_ambient_air(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray,
    dtcont: float,
    T_env: float = 15,
    dT_env_cont: float = 5,
    dt_phase_change: float = 0.01,
    is_heat_pumping: bool = True,    
):
    """Balance net hot and cold duties using an ambient sink/source if required.
    """
    delta_H = np.abs(H_cold).max() - np.abs(H_hot).max()
    if np.abs(delta_H) < tol:
        return H_hot, H_cold
    
    if H_hot.max() > tol:
        H_hot *= -1

    if H_cold.min() < -tol:
        H_cold *= -1
    
    if is_heat_pumping:
        if delta_H > tol:
            # Heat pump with potentially limited sources
            mask = np.where(
                T_hot <= (T_env - dT_env_cont - dt_phase_change - dtcont),
                delta_H, 0.0
            )
            H_hot -= mask
        # elif delta_H < -tol:
        #     # Heat pump with excess sources -> remove excess
        #     T_hot, H_hot = _get_H_col_till_target_Q(
        #         np.abs(H_cold).max(),
        #         T_hot,
        #         H_hot,
        #         False,
        #     )
            
    # if is_heat_pumping == False:
    #     if delta_H < -tol:
    #         # Refrigeration with potentially limited sinks
    #         mask = np.where(
    #             T_cold >= (T_env + dT_env_cont + dt_phase_change + dtcont),
    #             delta_H, 0.0
    #         )
    #         H_cold += mask
    #     elif delta_H > tol:
    #         # Refrigeration with excess sinks -> remove excess
    #         T_cold, H_cold = _get_H_col_till_target_Q(
    #             np.abs(H_hot).max(),
    #             T_cold,
    #             H_cold,
    #             True,
    #         )

    return T_hot, H_hot, T_cold, H_cold, delta_H


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
        False,
    )
    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value],
            hp_hot_streams,
            hp_cold_streams,
            is_shifted=False,
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


def _prepare_latent_hp_profile(
    T_hp: list,
    Q_hp: list,
    dT_phase_change: float,
    is_hot: bool,
    i: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp near-equal HP levels and merge their heat duties."""
    inc = 1 if is_hot else -1
    if i == None:
        i = 0 if is_hot else len(T_hp)
        
    i_range = range(i, len(T_hp) - 1) if is_hot else reversed(range(1, i))
    for i in i_range:
        if abs(T_hp[i] - T_hp[i+inc]) < dT_phase_change:
            T_hp.pop(i+inc)
            Q_hp[i] += Q_hp[i+inc]
            Q_hp.pop(i+inc)
            T_hp, Q_hp = _prepare_latent_hp_profile(
                T_hp,
                Q_hp,
                dT_phase_change,
                is_hot,
                i,                
            )
            break
    
    return T_hp, Q_hp


def _build_latent_streams(
    T_hp: np.ndarray, 
    dT_phase_change: float,
    Q_hp: np.ndarray,
    dt_cont: float,
    is_hot: bool
) -> StreamCollection:

    if len(T_hp) > 1:
        T_hp, Q_hp = _prepare_latent_hp_profile(T_hp.tolist(), Q_hp.tolist(), dT_phase_change, is_hot)

    sc = StreamCollection()
    for i in range(len(Q_hp)):
        sc.add(
            Stream(
                name=f"HP_H{i + 1}" if is_hot else f"HP_C{i + 1}",
                t_supply=T_hp[i] if is_hot else T_hp[i] - dT_phase_change,
                t_target=T_hp[i] - dT_phase_change if is_hot else T_hp[i],
                heat_flow=Q_hp[i],
                dt_cont=dt_cont, # Shift to intermediate process temperature scale
                is_process_stream=False,
            )            
        )
    return sc


def _build_simulated_hps_streams(
    hp_list: list,
) -> Tuple[StreamCollection, StreamCollection]:
    cond_streams, evap_streams = StreamCollection(), StreamCollection()
    hp: HeatPumpCycle
    for hp in hp_list:
        cond_streams.add_many(hp.build_stream_collection(include_cond=True))
        evap_streams.add_many(hp.build_stream_collection(include_evap=True))
    return cond_streams, evap_streams


# ============================================================
# ─── VISUALIZATION ──────────────────────────────────────────
# ============================================================


def _plot_multi_hp_profiles(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,
    H_cold: np.ndarray,
    T_hp_hot: np.ndarray,
    H_hp_hot: np.ndarray,
    T_hp_cold: np.ndarray,
    H_hp_cold: np.ndarray,
    title: str = None,
):
    """Plot HP cascade and source/sink profiles.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(H_hot, T_hot, label="Sink", linewidth=2, color="red")
    plt.plot(H_cold, T_cold, label="Source", linewidth=2, color="blue")
    plt.plot(H_hp_hot, T_hp_hot, "--", color="darkred", linewidth=1.8, label="Condenser")
    plt.plot(H_hp_cold, T_hp_cold, "--", color="darkblue", linewidth=1.8, label="Evaporator")
    plt.title(title)
    plt.xlabel("Heat Flow / kW")
    plt.ylabel("Temperature / °C")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axvline(0.0, color='black', linewidth=2)
    plt.tight_layout()
    plt.show()


def _plot_multi_hp_profiles_from_results(
    args: HeatPumpPlacementArgs, 
    res: CarnotHeatPumpResults, 
    title: str = None
):
    """Plot HP cascade and source/sink profiles.
    """        
    T_hot, H_hot = clean_composite_curve_ends(args.T_hot, args.H_hot)
    T_cold, H_cold = clean_composite_curve_ends(args.T_cold, args.H_cold)

    cascade = _get_heat_pump_cascade(hp_hot_streams=res.cond_streams, hp_cold_streams=res.evap_streams)
    T_hp_hot = cascade[PT.T.value]
    T_hp_cold = cascade[PT.T.value]
    H_hp_hot = cascade[PT.H_HOT_UT.value]
    H_hp_cold = cascade[PT.H_COLD_UT.value]

    _plot_multi_hp_profiles(
        T_hot,
        H_hot,
        T_cold,
        H_cold,
        T_hp_hot,
        H_hp_hot,
        T_hp_cold,
        H_hp_cold,
        title,
    )

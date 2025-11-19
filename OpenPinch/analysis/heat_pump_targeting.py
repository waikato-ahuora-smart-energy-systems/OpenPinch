"""Target heat pump integration for given heating or cooler profiles."""

import itertools
from scipy.optimize import (
    differential_evolution,
    minimize, 
    NonlinearConstraint,
    minimize_scalar
)

from ..lib import *
from ..utils import *
from .problem_table_analysis import *
from .gcc_manipulation import *
from .temperature_driving_force import *
from ..classes.stream import Stream
from ..classes.stream_collection import StreamCollection
from ..classes.problem_table import ProblemTable
from ..classes.simple_heat_pump import HeatPumpCycle

__all__ = ["get_heat_pump_targets", "calc_heat_pump_cascade", "plot_multi_hp_profiles_from_results"]


#######################################################################################################
# Public API
#######################################################################################################


def get_heat_pump_targets(
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    zone_config: Configuration,
    is_direct_integration: bool,
    is_heat_pumping: bool,  
):
    """Optimise multi-stage heat-pump placement for the supplied problem table.

    Args:
        pt: ProblemTable that already contains the composite-cascade columns.
        zone_config: Scenario configuration holding HP parameters, ambient data,
            and targeting flags.
        is_T_vals_shifted: Indicates whether ``pt`` temperatures already include
            ΔTmin/2 shifting.
        is_direct_integration: True when targeting is performed against the
            process cascade; False for utility-only cascades.
        is_heat_pumping: Selects heating mode (True) versus refrigeration (False).
        n_cond: Number of condenser temperature levels to optimise.
        n_evap: Number of evaporator temperature levels to optimise.
        eta_comp: Assumed compressor isentropic efficiency for optimisation.
        dtmin_hp: Minimum approach temperature enforced on the heat pump.

    Returns:
        HeatPumpTargetOutputs with the optimal placement, or an empty dict when
        targeting is skipped by the configuration screens.
    """    
    ######### TEMP VARS ######### 
    zone_config.HP_LOAD_FRACTION = 1
    zone_config.DO_HP_SIM = False
    #############################

    args = _prepare_heat_pump_target_inputs(
        T_vals=T_vals,
        H_hot=H_hot,
        H_cold=H_cold,
        is_direct_integration=is_direct_integration,
        is_heat_pumping=is_heat_pumping,        
        zone_config=zone_config,
    )
    if zone_config.MULTI_TEMPERATURE_HP:
        carnot_res = _optimise_multi_temperature_carnot_heat_pump_placement(args)
        if 0:
            plot_multi_hp_profiles_from_results(
                T_hot=args.T_hot,
                H_hot=args.H_hot,
                T_cold=args.T_cold,
                H_cold=args.H_cold,
                hp_hot_streams=carnot_res.cond_streams,
                hp_cold_streams=carnot_res.evap_streams,
            )        
    else:
        pass

    if zone_config.DO_HP_SIM or 0:
        args.n_cond = args.n_evap = max(args.n_cond, args.n_evap)
        res = _optimise_multi_simulated_heat_pump_placement(args, carnot_res)
    else:
        res = carnot_res

    res.amb_stream = _build_latent_streams(
        T_ls=np.array([zone_config.T_ENV]),
        dT_phase_change=zone_config.DT_PHASE_CHANGE,
        Q_ls=np.array([res.Q_amb]),
        dt_cont=zone_config.DT_ENV_CONT,
        is_hot=True if res.Q_amb >= -tol else False,
        is_process_stream=True,
        prefix="AIR",
    )

    return HeatPumpTargetOutputs.model_validate(res)


def calc_heat_pump_cascade(
    pt: ProblemTable,
    res: HeatPumpTargetOutputs,
    is_T_vals_shifted: bool,
    is_direct_integration: bool,
) -> ProblemTable:
    """Augment the base problem table with HP condenser/evaporator cascades."""
    t_hp = create_problem_table_with_t_int(
        streams=res.cond_streams + res.evap_streams, 
        is_shifted=is_T_vals_shifted,
    )
    pt.insert_temperature_interval(
        t_hp[PT.T.value].to_list()
    )
    temp = get_utility_heat_cascade(
        T_int_vals=pt.col[PT.T.value],
        hot_utilities=res.cond_streams,
        cold_utilities=res.evap_streams,
        is_shifted=is_T_vals_shifted,
    )
    col_H_net = PT.H_NET_HP_PRO.value if is_direct_integration else PT.H_NET_HP_UT.value  
    pt.update(  
        {
            col_H_net: temp[PT.H_NET_UT.value],
            PT.H_HOT_HP.value: temp[PT.H_HOT_UT.value],
            PT.H_COLD_HP.value: temp[PT.H_COLD_UT.value],
        }
    )

    # Calculate ambient air portion for the heat pump cascade
    if res.Q_amb > tol:
        hot_streams = res.amb_stream
        cold_streams = StreamCollection()
    elif res.Q_amb < tol:
        hot_streams = StreamCollection()
        cold_streams = res.amb_stream

    pt_air, _, _ = get_process_heat_cascade(
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        all_streams=hot_streams+cold_streams,
        include_real_pt=False,
    )
    pt_air.insert_temperature_interval(
        pt[PT.T.value].to_list()
    )
    pt.col[PT.H_NET_W_AIR.value] = pt.col[PT.H_NET_A.value] + pt_air.col[PT.H_NET.value]
    if res.Q_amb > tol:
        pt.col[PT.H_NET_HOT.value] -= pt_air.col[PT.H_NET.value]
    elif res.Q_amb < tol:
        pt.col[PT.H_NET_COLD.value] += pt_air.col[PT.H_NET.value]
    

def plot_multi_hp_profiles_from_results(
    T_hot: np.ndarray = None,
    H_hot: np.ndarray = None,
    T_cold: np.ndarray = None,
    H_cold: np.ndarray = None,
    hp_hot_streams: StreamCollection = None,
    hp_cold_streams: StreamCollection = None,
    title: str = None, 
):
    """Plot HP cascade and source/sink profiles.
    """
    plt.figure(figsize=(7, 5))

    if T_hot is not None and H_hot is not None:
        T_hot, H_hot = clean_composite_curve_ends(T_hot, H_hot)
        plt.plot(H_hot, T_hot, label="Sink", linewidth=2, color="red")

    if T_cold is not None and H_cold is not None:
        T_cold, H_cold = clean_composite_curve_ends(T_cold, H_cold)
        plt.plot(H_cold, T_cold, label="Source", linewidth=2, color="blue")

    if hp_hot_streams is not None and hp_cold_streams is not None:
        cascade = _get_heat_pump_cascade(hp_hot_streams=hp_hot_streams, hp_cold_streams=hp_cold_streams)
        T_hp_hot, H_hp_hot = clean_composite_curve_ends(cascade[PT.T.value], cascade[PT.H_HOT_UT.value])
        T_hp_cold, H_hp_cold = clean_composite_curve_ends(cascade[PT.T.value], cascade[PT.H_COLD_UT.value])
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


#######################################################################################################
# Helper functions: prepare args and get intial placement
#######################################################################################################


def _prepare_heat_pump_target_inputs(
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    is_direct_integration: bool = True,
    load_fraction: float = 1,
    is_heat_pumping: bool = True,
    zone_config: Configuration = Configuration(),
):
    """Format temperature/enthalpy inputs and options for heat pump targeting.
    """
    T_vals, H_hot, H_cold = T_vals.copy(), H_hot.copy(), H_cold.copy()
    T_hot, T_cold, dtcont_hp = _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(T_vals, zone_config.DTMIN_HP, is_direct_integration)   
    Q_hp_target = min(load_fraction, 1.0) * np.abs(H_cold).max()
    T_cold, H_cold = _get_H_col_till_target_Q(Q_hp_target, T_cold, H_cold)
    T_hot, H_hot, T_cold, H_cold, Q_amb_max = _balance_hot_and_cold_heat_loads_with_ambient_air(
        T_hot=T_hot, 
        H_hot=H_hot, 
        T_cold=T_cold, 
        H_cold=H_cold, 
        dtcont=dtcont_hp, 
        T_env=zone_config.T_ENV, 
        dT_env_cont=zone_config.DT_ENV_CONT, 
        dt_phase_change=zone_config.DT_PHASE_CHANGE, 
        is_heat_pumping=is_heat_pumping,
    )
    T_hot, H_hot = clean_composite_curve(T_hot, H_hot)
    T_cold, H_cold = clean_composite_curve(T_cold, H_cold)

    return HeatPumpTargetInputs(
        Q_hp_target=Q_hp_target,
        Q_amb_max=Q_amb_max,
        T_hot=T_hot,
        H_hot=H_hot,
        T_cold=T_cold,
        H_cold=H_cold,
        dt_range_max=max(T_cold[0], T_hot[0]) - min(T_cold[-1], T_hot[-1]),
        is_direct_integration=bool(is_direct_integration),
        is_heat_pumping=bool(is_heat_pumping),
        n_cond=zone_config.N_COND,
        n_evap=zone_config.N_EVAP,
        eta_comp=zone_config.ETA_COMP,
        dtcont_hp=zone_config.DTMIN_HP,
        y_cond_min= zone_config.Y_COND_MIN,
        load_fraction=zone_config.HP_LOAD_FRACTION,
        T_env=zone_config.T_ENV,
        dT_env_cont=zone_config.DT_ENV_CONT ,
        dt_phase_change=zone_config.DT_PHASE_CHANGE,
        refrigerant=zone_config.REFRIGERANTS[0],
        price_ratio=zone_config.PRICE_RATIO_ELE_TO_FUEL,
        max_multi_start=zone_config.MAX_HP_MULTISTART,
    )


def _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
    T_vals: np.ndarray, 
    dtmin_hp: float,
    is_direct_integration: bool,
):
    """Apply ΔTmin adjustments to cascade temperatures for heat pump calculations.
    """
    dT_shift = dtmin_hp * 0.5 if is_direct_integration else dtmin_hp * 1.5
    T_hot  = T_vals - dT_shift
    T_cold = T_vals + dT_shift
    return T_hot, T_cold, dT_shift


def _get_H_col_till_target_Q(
    Q_max: float,        
    T_vals: np.ndarray,    
    H_vals: np.ndarray,  
    is_cold: bool = True,      
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim the cold composite to the target heat duty, interpolating the boundary point.
    """
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
    T_env: float,
    dT_env_cont: float,
    dt_phase_change: float,
    is_heat_pumping: bool,    
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
            Q_amb_max = delta_H
        elif delta_H < -tol:
            # Heat pump with excess sources -> remove excess
            T_hot, H_hot = _get_H_col_till_target_Q(
                np.abs(H_cold).max(),
                T_hot,
                H_hot,
                False,
            )
            Q_amb_max = 0.0
            
    # else: # Refrigeration
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

    return T_hot, H_hot, T_cold, H_cold, Q_amb_max


#######################################################################################################
# Helper functions: Optimise multi-temperature Carnot heat pump placement
#######################################################################################################

@timing_decorator
def _optimise_multi_temperature_carnot_heat_pump_placement(
    args: HeatPumpTargetInputs
) -> HeatPumpTargetOutputs:
    """Compute baseline condenser/evaporator temperature levels and duties for a single multi-temperature heat-pump layout.
    """
    x0_ls, bnds = _get_x0_and_bnds_for_carnot_hp_opt(args)
    def _mapped_fun(x0):
        return minimize(
            fun=lambda x: _compute_carnot_hp_opt_obj(x, args).obj,
            x0=x0,
            bounds=bnds,
            method="SLSQP",            
        )
    minima = list(map(
        _mapped_fun,
        x0_ls
    ))
    opt = minima[0]
    for m in minima:
        if opt["fun"] > m["fun"]:
            opt = m

    res = _compute_carnot_hp_opt_obj(opt.x, args)
    res.cond_streams = _build_latent_streams(res.T_cond, args.dt_phase_change, res.Q_cond, args.dtcont_hp, is_hot=True)
    res.evap_streams = _build_latent_streams(res.T_evap, args.dt_phase_change, res.Q_evap, args.dtcont_hp, is_hot=False)
    return HeatPumpTargetOutputs.model_validate(res)


def _get_x0_and_bnds_for_carnot_hp_opt(
    args: HeatPumpTargetInputs
) -> Tuple[np.ndarray, list]:
    """Return the initial x vector and bounds for the Carnot HP optimizer.
    """
    x0_cond_ls = _initialise_carnot_hp_optimisation(
        n=args.n_cond,
        is_condenser=True,
        T0=args.T_cold[0],
        map_fun=_map_x_to_T_cond,
        T_vals=args.T_cold,
        H_vals=args.H_cold,
        dt_range_max=args.dt_range_max,
        Q_hp_target=args.Q_hp_target,
    )
    x0 = np.array(x0_cond_ls[0] + np.zeros(args.n_evap-1).tolist())
    res = _compute_carnot_hp_opt_obj(x0, args)
    x0_evap_ls = _initialise_carnot_hp_optimisation(
        n=args.n_evap,
        is_condenser=False,
        T0=res.T_evap[-1],
        map_fun=_map_x_to_T_evap,
        T_vals=args.T_hot,
        H_vals=args.H_hot,
        dt_range_max=args.dt_range_max,
        price_ratio=args.price_ratio,
    )
    x0_ls = list(itertools.product(x0_cond_ls, x0_evap_ls))
    x0_ls = [np.concatenate(x0) for x0 in x0_ls]
    bnds = [(0.0, 1.0) for _ in range(args.n_cond+args.n_evap-1)]
    return x0_ls, bnds


def _initialise_carnot_hp_optimisation(
    n: float,
    is_condenser: bool,
    T0: float,
    map_fun,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    dt_range_max: float,
    Q_hp_target: float = None,
    price_ratio: float = 1,
) -> list:
    """Build initial guesses based on a simple area maximisation problem, i.e., sum(Q x delta T).
    """
    n_stages = n if is_condenser else n-1
    args = {
        "is_condenser": is_condenser,
        "map_fun": map_fun,
        "T0": T0,
        "T_vals": T_vals,
        "H_vals": H_vals,
        "dt_range_max": dt_range_max,
        "Q_hp_target": Q_hp_target,
        "price_ratio": price_ratio,
    }
    local_minima_fun, local_minima_x = dual_annealing_multiminima(
        func=lambda x: _compute_entropy_generation_reduction_at_constant_T(x, args),
        bounds=[(0.0, 1.0) for _ in range(n_stages)], 
    )
    for i, f in enumerate(local_minima_fun):
        if f > 0 and i != 0:
            break
    local_minima_x = local_minima_x[:i]
    if len(local_minima_x) == 0: 
        res = differential_evolution(
            func=lambda x: _compute_entropy_generation_reduction_at_constant_T(x, args),
            bounds=[(0.0, 1.0) for _ in range(n_stages)], 
        )
        local_minima_x = [res.x]
        
    return local_minima_x


def _compute_entropy_generation_reduction_at_constant_T(
    x: int,
    args: dict,
) -> np.ndarray:    
    """Compute the reduction in entropy generation as a result of selection of x; used for seeding the Carnot HP optimisation.
    """
    T = args["map_fun"](x, args["T0"], args["dt_range_max"])
    Q = _get_Q_vals_from_T_hp_vals(T, args["T_vals"], args["H_vals"], args["is_condenser"])
    if args["is_condenser"]:
        W = args["Q_hp_target"] - Q.sum()
        S_red = (Q * (T - args["T0"]) / ((T + 273.15) * (args["T0"] + 273.15))).sum() + W / (T.max() + 273.15) / args["price_ratio"]
    else: 
        S_red = (Q * (args["T0"] - T) / ((T + 273.15) * (args["T0"] + 273.15))).sum()
    return S_red # is negative, meaning a reduction in entropy generation


def _compute_carnot_hp_opt_obj(
    x: np.ndarray, 
    args: HeatPumpTargetInputs
) -> HeatPumpTargetOutputs:
    """Evaluate compressor work for a candidate HP placement defined by vector `x`.
    """
    x_cond, x_evap = _parse_carnot_hp_state_variables(x, args.n_cond)
    T_cond = _map_x_to_T_cond(x_cond, args.T_cold[0], args.dt_range_max)
    Q_cond = _get_Q_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold, True)

    opt = minimize_scalar(
        fun=lambda T: _get_optimal_min_evap_T(T, [args, T_cond, Q_cond, x_evap])["obj"],
        bracket=(args.T_hot[-1], args.T_hot[0]),
        method="brent",
        tol=1e-6,
    )
    res = HeatPumpTargetOutputs.model_validate(
        _get_optimal_min_evap_T(opt.x, [args, T_cond, Q_cond, x_evap])
    )
    Q_cond_max = res.Q_cond.sum()    
    Q_evap_max = res.Q_evap.sum()
    if Q_evap_max > Q_cond_max * (1 - 1 / res.cop):
        Q_cond_tot = Q_cond_max
        work_hp = Q_cond_tot / res.cop
        Q_evap_tot = Q_cond_tot - work_hp
        res.Q_evap = res.Q_evap * (Q_evap_tot / Q_evap_max)
    else: # Evaporator limits the heat pump
        Q_evap_tot = Q_evap_max
        work_hp = Q_evap_tot / (res.cop - 1)
        Q_cond_tot = Q_evap_tot + work_hp
        res.Q_cond = res.Q_cond * (Q_cond_tot / Q_cond_max)

    if 0:
        res.cond_streams = _build_latent_streams(res.T_cond, args.dt_phase_change, res.Q_cond, args.dtcont_hp, is_hot=True)
        res.evap_streams = _build_latent_streams(res.T_evap, args.dt_phase_change, res.Q_evap, args.dtcont_hp, is_hot=False)         
        plot_multi_hp_profiles_from_results(args.T_hot, args.H_hot, args.T_cold, args.H_cold, res.cond_streams, res.evap_streams, str(f"Obj: {res.obj} = {res.work_hp} + {res.Q_ext}"))
    return res


def _get_optimal_min_evap_T(
    T_lo: np.ndarray, 
    input_args: list
) -> dict:
    """Adjust evaporator ladder to honour minimum temperature limits."""
    args, T_cond, Q_cond, x_evap = input_args
    T_evap = _map_x_to_T_evap(x_evap, T_lo, args.dt_range_max)
    Q_evap = _get_Q_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot, False)
    cop = _compute_COP_estimate_from_carnot_limit(T_cond, Q_cond, T_evap, Q_evap)

    Q_evap_max = Q_evap.sum()
    Q_cond_max = Q_cond.sum()
    if Q_evap_max > Q_cond_max * (1 - 1 / cop):
        Q_cond_tot = Q_cond_max
        work_hp = Q_cond_tot / cop
        Q_evap_tot = Q_cond_tot - work_hp
    else: # Evaporator limits the heat pump
        Q_evap_tot = Q_evap_max
        work_hp = Q_evap_tot / (cop - 1)
        Q_cond_tot = Q_evap_tot + work_hp
    
    Q_ext = max(args.Q_hp_target - Q_cond_tot, 0.0) # Direct electric heating

    return {
        "obj": (work_hp + Q_ext / args.price_ratio) / args.Q_hp_target, 
        "utility_tot": work_hp + Q_ext, 
        "work_hp": work_hp,  
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "Q_cond": Q_cond,            
        "T_evap": T_evap, 
        "Q_evap": Q_evap, 
        "cop": cop,
        "Q_amb": max(Q_evap.sum() - (np.abs(args.H_hot).max() - args.Q_amb_max), args.Q_amb_max),
    }


def _parse_carnot_hp_state_variables(
    x: np.ndarray, 
    n_cond: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n_cond_vars = int(n_cond)
    x_cond = x[:n_cond_vars]
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


def _compute_entropic_average_temperature_in_K(
    T: np.ndarray,
    Q: np.ndarray,     
    T_units: str = "C",   
):
    """Compute the entropic average temperature.
    """
    unit_offset = 273.15 if T_units == "C" else 0
    if T.var() < tol:
        return T[0] + unit_offset
    S_tot = (Q / (T + unit_offset)).sum()
    return Q.sum() / S_tot if S_tot > 0 else (T.mean() + unit_offset)
    

def _compute_COP_estimate_from_carnot_limit(
    T_cond: np.ndarray,
    Q_cond: np.ndarray, 
    T_evap: np.ndarray,
    Q_evap: np.ndarray,  
    eff: float = 0.5, 
    min_dt_lift: float = 3.0,          
):  
    """Estimate COP by scaling the Carnot limit using entropic mean temperatures.
    """
    T_h = _compute_entropic_average_temperature_in_K(T_cond, Q_cond)
    T_l = _compute_entropic_average_temperature_in_K(T_evap, Q_evap)
    cop = (
        T_l / (T_h - T_l) * eff + 1
        if (T_h - T_l) > min_dt_lift 
        else T_l / min_dt_lift * eff + 1
    )
    return cop


#######################################################################################################
# Helper functions: Optimise simulated heat pump placement
#######################################################################################################

@timing_decorator
def _optimise_multi_simulated_heat_pump_placement(
    args: HeatPumpTargetInputs,
    carnot_res: HeatPumpTargetOutputs,
) -> HeatPumpTargetOutputs:
    """Optimise a multi-unit heat-pump cascade against composite curve constraints.
    """
    args.net_hot_streams, _ = _create_net_hot_and_cold_stream_collections_for_background_profile(args.T_hot, np.abs(args.H_hot))
    _, args.net_cold_streams = _create_net_hot_and_cold_stream_collections_for_background_profile(args.T_cold, args.H_cold)

    x0, bnds = _prepare_data_for_minimizer(carnot_res.T_cond, carnot_res.Q_cond, carnot_res.T_evap, args)
    constraints = (
        NonlinearConstraint(
            fun=lambda x: _constrain_min_temperature_lift(x, args),
            lb=0.0,
            ub=np.inf,
        ),
        NonlinearConstraint(
            fun=lambda x: _constrain_max_cond_duty(x, args),
            lb=0.0,
            ub=1.0,
        ),
    )
    opt = minimize(
        fun=lambda x: _compute_simulated_heat_pump_system_performance(x, args)["obj"],
        x0=x0,
        constraints=constraints,
        method="SLSQP", # SLSQP. COBYQA
        bounds=bnds,
        options={'disp': False, 'maxiter': 1000},
        tol=1e-6,
    )

    if opt.success:
        res = _compute_simulated_heat_pump_system_performance(opt.x, args)
        if 0:
            res = HeatPumpTargetOutputs.model_validate(res)
            plot_multi_hp_profiles_from_results(args.T_hot, args.H_hot, args.T_cold, args.H_cold, res.cond_streams, res.evap_streams)
        
    else:
        print("Optimization failed:", opt.message)

    return HeatPumpTargetOutputs.model_validate(res)


def _prepare_data_for_minimizer(
    T_cond: np.ndarray, 
    Q_cond: np.ndarray, 
    T_evap: np.ndarray,
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray, list]:
    """Build initial guesses and bounds for the condenser/evaporator temperature.
    """
    x_cond_bnds, x_evap_bnds = _x_bnds_for_refrigerant(args)
    x_sc_bnds = [np.float64(args.dt_phase_change / 10 / args.dt_range_max), (args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max]
    x_sh_bnds = [np.float64(args.dt_phase_change / 10 / args.dt_range_max), (args.T_hot[0]  - args.T_hot[-1] ) / args.dt_range_max]

    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds[1] = x_sc_bnds[0]
    
    if x_sh_bnds[0] > x_sh_bnds[1]:
        x_sh_bnds[1] = x_sh_bnds[0]

    y_cond_bnds = (0.0, 1.0)

    x_cond = _map_T_to_x_cond(T_cond, args.T_cold[0], args.dt_range_max)
    x_sc = np.ones(x_cond.size) / args.dt_range_max
    y_cond = Q_cond[1:] / args.Q_hp_target

    x_evap = _map_T_to_x_evap(T_evap, args.T_hot[-1], args.dt_range_max)
    x_sh = np.ones(x_evap.size) / args.dt_range_max

    x_ls = []
    bnds = []

    def _build_lists(candidate, limits):
        x_ls.append(candidate[:])
        bnds.extend([limits for _ in range(len(candidate[:]))])

    pairs = [
        (x_cond, x_cond_bnds),
        (x_sc,   x_sc_bnds  ),
        (y_cond, y_cond_bnds),
        (x_evap, x_evap_bnds),
        (x_sh,   x_sh_bnds  ),
    ]
    for candidate, limits in pairs:
        _build_lists(candidate, limits)
    x0 = np.concatenate(x_ls) 

    return x0, bnds


def _x_bnds_for_refrigerant(
    args: HeatPumpTargetInputs
):
    """Return bounds of the refrigerant temperatures accounting for min and critical temperatures of a refrigerant.
    """
    T_min, T_crit = PropsSI('Tmin', args.refrigerant), PropsSI('Tcrit', args.refrigerant)
    if args.unit_system == 'EUR':  # Convert to °C if needed
        T_min -= 273.15
        T_crit -= 273.15
    T_cond_bnds = np.array((
        max(args.T_cold[-1], T_min + 1),
        min(args.T_cold[0] - args.dt_phase_change, T_crit - 1),
    ))
    x_cond_bnds = (
        _map_T_to_x_cond(T_cond_bnds[1], args.T_cold[0], args.dt_range_max), # Account for cond stream temp reduction minimum
        _map_T_to_x_cond(T_cond_bnds[0], args.T_cold[0], args.dt_range_max),
    )
    
    T_evap_bnds = np.array((
        max(args.T_hot[-1], T_min + 1),
        min(args.T_hot[0] - args.dt_phase_change, T_crit - 1), # Account for evap stream temp rise minimum
    ))
    x_evap_bnds = (
        _map_T_to_x_evap(T_evap_bnds[0], args.T_hot[-1], args.dt_range_max),
        _map_T_to_x_evap(T_evap_bnds[1], args.T_hot[-1], args.dt_range_max),
    )

    return x_cond_bnds, x_evap_bnds


def _constrain_min_temperature_lift(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
):
    """Ensure T_cond - dT_sc > T_evap + dtcont_hp
    """
    T_cond, dT_sc, _, T_evap, _ = _parse_simulated_hp_state_temperatures(x, args)
    T_diff = (T_cond - dT_sc) - T_evap - args.dtcont_hp
    return T_diff.min()


def _constrain_max_cond_duty(    
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
):
    """Ensure sum of the heating duty fractions sum to less than 1.
    """
    n = args.n_cond
    return 1 - x[n*2:n*3-1].sum()


def _parse_simulated_hp_state_temperatures(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x.
    """
    n = args.n_cond 

    T_cond = args.T_cold[0] - x[:n] * args.dt_range_max
    i = n
    dT_sc = x[i:i+n] * args.dt_range_max
    i += n
    Q_cond = np.concatenate([
        [(1 - x[i:i+(n-1)].sum()) * args.Q_hp_target], 
        x[i:i+(n-1)] * args.Q_hp_target
    ])
    i += (n-1)
    T_evap = x[i:i+n] * args.dt_range_max + args.T_hot[-1]  
    i += n
    dT_sh = x[i:] * args.dt_range_max

    return T_cond, dT_sc, Q_cond, T_evap, dT_sh


def _compute_simulated_heat_pump_system_performance(
    x: np.ndarray, 
    args: HeatPumpTargetInputs
) -> float:
    """Objective: minimize total compressor work for multi-HP configuration.
    """
    T_cond, dT_sc, Q_cond, T_evap, dT_sh = _parse_simulated_hp_state_temperatures(x, args)

    # Create HP objects with correct temperature mapping
    hp_list = _create_heat_pump_list(
        T_cond=T_cond, 
        dT_sc=dT_sc, 
        Q_cond=Q_cond, 
        T_evap=T_evap, 
        dT_sh=dT_sh,
        args=args,
    )
    cond_streams, evap_streams = _build_simulated_hps_streams(hp_list)
    hot_streams = cond_streams + args.net_hot_streams
    cold_streams = evap_streams + args.net_cold_streams
    all_streams = hot_streams + cold_streams
    pt, _, _ = get_process_heat_cascade(
        hot_streams, cold_streams, all_streams
    )

    work_hp = sum([hp.work for hp in hp_list])
    work_el = pt.col[PT.H_NET.value][0] # Acts as a penalty
    Q_evap = np.array([hp.Q_evap for hp in hp_list])

    if 0:
        plot_multi_hp_profiles_from_results(args.T_hot, args.H_hot, args.T_cold, args.H_cold, cond_streams, evap_streams)
        plot_multi_hp_profiles_from_results(pt.col[PT.T.value], pt.col[PT.H_NET.value])

    return {
        "obj": (work_hp + work_el) / args.Q_hp_target, 
        "utility_tot": work_hp + work_el, 
        "work_hp": work_hp,  
        "Q_ext": work_el,
        "T_cond": T_cond,
        "dT_sc": dT_sc,
        "Q_cond": Q_cond,            
        "T_evap": T_evap, 
        "dT_sh": dT_sh,
        "Q_evap": Q_evap,
        "cop": args.Q_hp_target / (work_hp + work_el),
        "Q_amb": Q_evap.sum() - (np.abs(args.H_hot).max() - args.Q_amb_max),
        "cond_streams": cond_streams,
        "evap_streams": evap_streams,
    }


def _create_heat_pump_list(
    T_cond: np.ndarray, 
    dT_sc: np.ndarray, 
    Q_cond: np.ndarray, 
    T_evap: np.ndarray, 
    dT_sh: np.ndarray,
    args: HeatPumpTargetInputs,
) -> List[HeatPumpCycle]:
    """Instantiate HeatPumpCycle objects that span the condenser/evaporator ladders.
    """
    hp_list = []
    n_hp = args.n_cond
    for i in range(n_hp):
        hp = HeatPumpCycle(
            refrigerant=args.refrigerant, 
            unit_system=args.unit_system, 
            Q_total=Q_cond[i],
        )
        if Q_cond[i] > 0.0:
            Te = min(T_evap[i], T_cond[i] - dT_sc[i] - 3)
            hp.solve_t_dt(
                Te=Te,
                Tc=T_cond[i],
                dT_sh=dT_sh[i] if isinstance(dT_sh, np.ndarray) else 0.001,
                dT_sc=dT_sc[i] if isinstance(dT_sh, np.ndarray) else 0.001,
                eta_comp=args.eta_comp,
                SI=False
            )
        hp_list.append(hp)
    return hp_list


def _build_simulated_hps_streams(
    hp_list: list,
) -> Tuple[StreamCollection, StreamCollection]:
    """Aggregate condenser and evaporator streams for each simulated HP cycle."""
    cond_streams, evap_streams = StreamCollection(), StreamCollection()
    for hp in hp_list:
        hp: HeatPumpCycle
        cond_streams.add_many(hp.build_stream_collection(include_cond=True, is_process_stream=True))
        evap_streams.add_many(hp.build_stream_collection(include_evap=True, is_process_stream=True))
    return cond_streams, evap_streams


def _create_net_hot_and_cold_stream_collections_for_background_profile(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> Tuple[StreamCollection, StreamCollection]:
    """Constructs net stream segments that require utility input across temperature intervals."""
    net_hot_streams = StreamCollection()
    net_cold_streams = StreamCollection()
    
    T_vals = np.array(T_vals)
    H_vals = np.array(H_vals)

    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    dh_vals = delta_vals(H_vals)

    for i, dh in enumerate(dh_vals):
        if dh > tol:
            net_cold_streams.add(
                Stream(
                    t_supply=T_vals[i+1],
                    t_target=T_vals[i],
                    heat_flow=dh,
                )
            )
        elif -dh > tol:
            net_hot_streams.add(
                Stream(
                    t_supply=T_vals[i],
                    t_target=T_vals[i+1],
                    heat_flow=-dh,
                )
            )

    return net_hot_streams, net_cold_streams


#######################################################################################################
# Helper functions: other
#######################################################################################################


def _map_T_to_x_cond(
    T: np.ndarray, 
    T_hi: float, 
    deltaT_range: float,
) -> np.ndarray:
    """Map monotonically decreasing condenser temperatures to x values."""
    if T.size > 1:
        T_base = np.roll(T, 1)
        T_base[0] = T_hi
    else:
        T_base = T_hi
    return (T_base - T) / deltaT_range


def _map_T_to_x_evap(
    T: np.ndarray, 
    T_lo: float, 
    deltaT_range: float,
) -> np.ndarray:
    """Map monotonically increasing evaporator temperatures to x values."""
    if T.size > 1:
        T_base = np.roll(T, -1)
        T_base[-1] = T_lo
    else:
        T_base = T_lo
    return (T - T_base) / deltaT_range


def _map_x_to_T_cond(
    x: np.ndarray, 
    T_hi: float, 
    deltaT_range: float,
) -> np.ndarray:
    """Recover condenser temperatures from normalized x values."""
    temp = []
    for i in range(x.shape[0]):
        temp.append(T_hi - x[i] * deltaT_range)
        T_hi = temp[-1]
    return np.array(temp).flatten()


def _map_x_to_T_evap(
    x: np.ndarray,
    T_lo: float, 
    deltaT_range: float,
) -> np.ndarray:
    """Recover evaporator temperatures from normalized x values.
    """    
    temp = []
    for i in range(x.size):
        temp.append(x[::-1][i] * deltaT_range + T_lo)
        T_lo = temp[-1]
    return np.array(temp).flatten()[::-1]


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


def _prepare_latent_hp_profile(
    T_hp: list,
    Q_hp: list,
    dT_phase_change: float,
    is_hot: bool,
    i: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp near-equal HP levels and merge their heat duties.
    """
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
    T_ls: np.ndarray, 
    dT_phase_change: float,
    Q_ls: np.ndarray,
    dt_cont: float,
    is_hot: bool,
    is_process_stream: bool = False,
    prefix: str = "HP",
) -> StreamCollection:
    """Convert a series of temperature levels into a StreamCollection, e.g. for condenser/evaporator levels or ambient air.
    """
    if len(T_ls) > 1:
        T_ls, Q_ls = _prepare_latent_hp_profile(T_ls.tolist(), Q_ls.tolist(), dT_phase_change, is_hot)

    sc = StreamCollection()
    for i in range(len(Q_ls)):
        sc.add(
            Stream(
                name=f"{prefix}_H{i + 1}" if is_hot else f"{prefix}_C{i + 1}",
                t_supply=T_ls[i] if is_hot else T_ls[i] - dT_phase_change,
                t_target=T_ls[i] - dT_phase_change if is_hot else T_ls[i],
                heat_flow=Q_ls[i],
                dt_cont=dt_cont, # Shift to intermediate process temperature scale
                is_process_stream=is_process_stream,
            )            
        )
    return sc
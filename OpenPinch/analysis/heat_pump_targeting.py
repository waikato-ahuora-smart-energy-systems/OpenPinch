"""Heat-pump targeting and cascade-construction utilities for composite curves."""

import numpy as np
from scipy.optimize import minimize
from ..lib import *
from ..utils import *
from .problem_table_analysis import *
from .gcc_manipulation import *
from .temperature_driving_force import *
from ..classes.stream import Stream
from ..classes.stream_collection import StreamCollection
from ..classes.problem_table import ProblemTable
from ..classes.brayton_heat_pump import SimpleBraytonHeatPumpCycle
from ..classes.cascade_heat_pump import CascadeHeatPumpCycle
from ..classes.multi_simple_heat_pump import MultiSimpleHeatPumpCycle

__all__ = ["get_heat_pump_targets", "calc_heat_pump_cascade", "plot_multi_hp_profiles_from_results"]


#######################################################################################################
# Public API
#######################################################################################################


def get_heat_pump_targets(  
    Q_hp_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    zone_config: Configuration,
    is_direct_integration: bool,
    is_heat_pumping: bool,
) -> HeatPumpTargetOutputs:
    """Optimise heat pump placement for a target duty and cascade profiles.

    Parameters
    ----------
    Q_hp_target : float
        Target condenser duty to be delivered by the heat pump system [kW].
    T_vals : np.ndarray
        Temperature interval grid used by the background cascade [degC].
    H_hot : np.ndarray
        Net hot profile aligned with ``T_vals``. Values are interpreted as hot
        availability and are normalised internally to negative values.
    H_cold : np.ndarray
        Net cold profile aligned with ``T_vals``. Values are interpreted as cold
        demand and are normalised internally to positive values.
    zone_config : Configuration
        Zone-level configuration containing heat pump type, stage counts,
        efficiencies, refrigerants, and ambient settings.
    is_direct_integration : bool
        ``True`` for process-integrated targeting, ``False`` for utility-level
        targeting.
    is_heat_pumping : bool
        ``True`` for heating mode, ``False`` for refrigeration mode.

    Returns
    -------
    HeatPumpTargetOutputs
        Validated optimisation output, including objective terms, cycle state
        variables, and generated stream collections.

    Raises
    ------
    ValueError
        If ``zone_config.HP_TYPE`` does not map to a supported optimiser.
    """
    zone_config.HP_TYPE = HeatPumpType.MultiSimpleVapourComp.value
    args = _prepare_heat_pump_target_inputs(
        Q_hp_target=Q_hp_target,
        T_vals=T_vals,
        H_hot=np.abs(H_hot) * -1,
        H_cold=np.abs(H_cold),
        is_direct_integration=is_direct_integration,
        is_heat_pumping=is_heat_pumping,        
        zone_config=zone_config,
        debug=True,
    )
    handler = _HP_PLACEMENT_HANDLERS.get(zone_config.HP_TYPE)
    if handler is None:
        raise ValueError("No valid heat pump targeting type selected.")
    res = handler(args)
    res.amb_stream = _get_ambient_air_stream(res.Q_amb, args)
    return HeatPumpTargetOutputs.model_validate(res)


def calc_heat_pump_cascade(
    pt: ProblemTable,
    res: HeatPumpTargetOutputs,
    is_T_vals_shifted: bool,
    is_direct_integration: bool,
) -> ProblemTable:
    """Insert heat pump cascade contributions into a problem table.

    Parameters
    ----------
    pt : ProblemTable
        Base problem table to augment. This object is modified in place.
    res : HeatPumpTargetOutputs
        Heat-pump targeting result containing condenser, evaporator, and ambient
        stream collections.
    is_T_vals_shifted : bool
        Whether ``pt`` temperatures are already on the shifted temperature scale.
    is_direct_integration : bool
        Selects the destination net-cascade column:
        ``PT.H_NET_HP_PRO`` for direct integration and ``PT.H_NET_HP_UT`` for
        utility integration.

    Returns
    -------
    ProblemTable
        The same ``pt`` instance, after insertion of HP intervals and update of
        HP-related columns.

    Notes
    -----
    In-place updates include ``PT.H_HOT_HP``, ``PT.H_COLD_HP``, and
    ``PT.H_NET_W_AIR``. Ambient-air interactions are included when present in
    ``res.amb_stream``.
    """
    t_hp = create_problem_table_with_t_int(
        streams=res.hp_hot_streams + res.hp_cold_streams, 
        is_shifted=is_T_vals_shifted,
    )
    pt.insert_temperature_interval(
        t_hp[PT.T.value].to_list()
    )
    temp = get_utility_heat_cascade(
        T_int_vals=pt.col[PT.T.value],
        hot_utilities=res.hp_hot_streams,
        cold_utilities=res.hp_cold_streams,
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
    hot_streams = StreamCollection()
    cold_streams = StreamCollection()
    if res.Q_amb >= tol:
        hot_streams = res.amb_stream
    elif res.Q_amb <= -tol:
        cold_streams = res.amb_stream

    if len(res.amb_stream) > 0:
        pt_air = get_process_heat_cascade(
            hot_streams=hot_streams,
            cold_streams=cold_streams,
            is_shifted=True,
        )
        T_ls_pt = pt[PT.T.value].to_list()
        T_ls_pt_air = pt_air[PT.T.value].to_list()
        pt_air.insert_temperature_interval(
            T_ls_pt
        )
        pt.insert_temperature_interval(
            T_ls_pt_air
        )        
        pt.col[PT.H_NET_W_AIR.value] = pt.col[PT.H_NET_A.value] + pt_air.col[PT.H_NET.value]

        if res.Q_amb >= tol:
            pt.col[PT.H_NET_HOT.value] -= pt_air.col[PT.H_NET.value]
        elif res.Q_amb <= -tol:
            pt.col[PT.H_NET_COLD.value] += pt_air.col[PT.H_NET.value]
    else:
        pt.col[PT.H_NET_W_AIR.value] = pt.col[PT.H_NET_A.value]

    return pt


def plot_multi_hp_profiles_from_results(
    T_hot: np.ndarray = None,
    H_hot: np.ndarray = None,
    T_cold: np.ndarray = None,
    H_cold: np.ndarray = None,
    hp_hot_streams: StreamCollection = None,
    hp_cold_streams: StreamCollection = None,
    title: str = None, 
) -> None:
    """Plot source/sink composites together with HP condenser/evaporator profiles.

    Parameters
    ----------
    T_hot, H_hot : np.ndarray, optional
        Temperature and heat-flow arrays for the sink-side composite.
    T_cold, H_cold : np.ndarray, optional
        Temperature and heat-flow arrays for the source-side composite.
    hp_hot_streams, hp_cold_streams : StreamCollection, optional
        Stream collections describing HP condenser and evaporator duties.
        Plotting of HP profiles occurs only when both are provided.
    title : str, optional
        Figure title.

    Returns
    -------
    None
        This function creates and renders a matplotlib figure.

    Notes
    -----
    When arrays are provided, each temperature vector must align with its
    corresponding heat-flow vector.
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
    plt.ylabel("Temperature / degC")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axvline(0.0, color='black', linewidth=2)
    plt.tight_layout()
    plt.show()


#######################################################################################################
# Helper functions: prepare args and get intial placement
#######################################################################################################


def _prepare_heat_pump_target_inputs(
    Q_hp_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    *,
    is_direct_integration: bool = True,
    is_heat_pumping: bool = True,
    zone_config: Configuration = Configuration(),
    debug: bool = False,    
):
    """Build a validated optimisation-input bundle for heat pump targeting.

    Parameters
    ----------
    Q_hp_target : float
        Target heat pump duty [kW].
    T_vals : np.ndarray
        Background temperature grid [degC].
    H_hot, H_cold : np.ndarray
        Background hot/cold cascade coordinates aligned to ``T_vals``.
    is_direct_integration : bool, default=True
        Integration context flag used for temperature shifting.
    is_heat_pumping : bool, default=True
        Heating/refrigeration mode selection.
    zone_config : Configuration
        Configuration source for optimisation settings.
    debug : bool, default=False
        Enables optimisation debug plotting and extra diagnostics.

    Returns
    -------
    HeatPumpTargetInputs
        Normalised and validated optimisation arguments.
    """
    T_vals, H_hot, H_cold = T_vals.copy(), H_hot.copy(), H_cold.copy()
    T_hot, T_cold, dtcont_hp = _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(T_vals, zone_config.DTMIN_HP, is_direct_integration)   
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
    net_hot_streams, _ = _create_net_hot_and_cold_stream_collections_for_background_profile(T_hot, np.abs(H_hot))
    _, net_cold_streams = _create_net_hot_and_cold_stream_collections_for_background_profile(T_cold, H_cold)    

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
        n_cond=zone_config.N_COND +1,
        n_evap=zone_config.N_EVAP +1,
        eta_comp=zone_config.ETA_COMP,
        eta_exp=zone_config.ETA_EXP,
        eta_hp_carnot=zone_config.ETA_HP_CARNOT,
        eta_he_carnot=zone_config.ETA_HE_CARNOT,
        dtcont_hp=zone_config.DTMIN_HP,
        dt_hp_ihx = zone_config.DT_HP_IHX,
        dt_cascade_hx = zone_config.DT_CASCADE_HX,
        load_fraction=zone_config.HP_LOAD_FRACTION,
        T_env=zone_config.T_ENV,
        dt_env_cont=zone_config.DT_ENV_CONT,
        dt_phase_change=zone_config.DT_PHASE_CHANGE,
        refrigerant_ls=[r.strip().upper() for r in zone_config.REFRIGERANTS],
        do_refrigerant_sort=zone_config.DO_REFRIGERANT_SORT,
        price_ratio=zone_config.PRICE_RATIO_ELE_TO_FUEL if zone_config.PRICE_RATIO_ELE_TO_FUEL > 0.0 else 1,
        max_multi_start=zone_config.MAX_HP_MULTISTART,
        bb_minimiser=zone_config.BB_MINIMISER,
        allow_integrated_expander=zone_config.ALLOW_INTEGRATED_EXPANDER,
        eta_penalty=0.001,
        rho_penalty=100,        
        net_hot_streams=net_hot_streams,
        net_cold_streams=net_cold_streams,
        debug=debug,
        initialise_simulated_hp=zone_config.INITIALISE_SIMULATED_HP,
    )


def _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
    T_vals: np.ndarray, 
    dtmin_hp: float,
    is_direct_integration: bool,
):
    """Apply HP-specific temperature shifting to a cascade grid.

    Parameters
    ----------
    T_vals : np.ndarray
        Temperature interval grid [degC].
    dtmin_hp : float
        Minimum approach temperature for HP integration [K].
    is_direct_integration : bool
        ``True`` applies process-scale shifting; ``False`` applies utility-scale
        shifting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Shifted hot temperatures, shifted cold temperatures, and applied shift.
    """
    dT_shift = dtmin_hp * 0.5 if is_direct_integration else dtmin_hp * 1.5 # utility streams are real temperatures??? 
    T_hot  = T_vals - dT_shift
    T_cold = T_vals + dT_shift
    return T_hot, T_cold, dT_shift


def _get_H_col_till_target_Q(
    Q_max: float,        
    T_vals: np.ndarray,    
    H_vals: np.ndarray,  
    *,
    is_cold: bool = True,      
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim a cascade profile to a target duty using boundary interpolation.

    Parameters
    ----------
    Q_max : float
        Target absolute duty to retain.
    T_vals : np.ndarray
        Temperature coordinates of the profile.
    H_vals : np.ndarray
        Heat coordinates of the profile.
    is_cold : bool, default=True
        ``True`` trims from the cold-profile side, ``False`` from the hot side.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Trimmed temperature and heat arrays.
    """
    if abs(np.abs(H_vals).max() - Q_max) < tol:
        return T_vals, H_vals
    if Q_max <= tol:
        return (T_vals[-1:], H_vals[-1:]) if is_cold else (T_vals[:1], H_vals[:1])
    mask = np.where(
        np.abs(H_vals) >= Q_max,
        1.0, 0.0
    )
    i_vals = np.flatnonzero(mask)
    if i_vals.size == 0:
        return (T_vals[-1:], H_vals[-1:]) if is_cold else (T_vals[:1], H_vals[:1])
    if is_cold:
        i = i_vals[-1]
        if i >= len(H_vals) - 1:
            return T_vals[i:], H_vals[i:]
        T_vals[i] = linear_interpolation(
            Q_max, H_vals[i], H_vals[i + 1], T_vals[i], T_vals[i + 1]
        )      
        H_vals[i] = Q_max  
        return T_vals[i:], H_vals[i:]
    else:
        i = i_vals[0]
        if i <= 0:
            return T_vals[:1], H_vals[:1]
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
    """Balance hot/cold targets with optional ambient-air exchange.

    Parameters
    ----------
    T_hot, H_hot, T_cold, H_cold : np.ndarray
        Cascade coordinates for hot and cold profiles.
    dtcont : float
        Effective temperature shift used for contact constraints.
    T_env : float
        Ambient temperature [degC].
    dT_env_cont : float
        Minimum ambient contact temperature difference [K].
    dt_phase_change : float
        Representative phase-change temperature span [K].
    is_heat_pumping : bool
        ``True`` for heating mode, ``False`` for refrigeration mode.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        Updated ``(T_hot, H_hot, T_cold, H_cold, Q_amb_max)``.
    """
    Q_amb_max = 0.0
    delta_H = np.abs(H_cold).max() - np.abs(H_hot).max()
    # if np.abs(delta_H) < tol:
    #     return T_hot, H_hot, T_cold, H_cold, Q_amb_max
    
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
                is_cold=False,
            )
            
            
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
    """Compute baseline condenser/evaporator temperature levels and duties for a single multi-temperature heat pump layout.
    """
    bnds = [(0.0, 1.0) for _ in range(args.n_cond + args.n_evap)]
    minima = multiminima(
        func=_compute_multi_temperature_carnot_hp_opt_obj,
        func_kwargs=args,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,
    )
    x = minima[0]

    res = _compute_multi_temperature_carnot_hp_opt_obj(x, args, debug=args.debug)
    if res["opt_success"] == False:
        raise ValueError("Multi-temperature Carnot heat pump targeting failed to return an optimal result.")
    res.update(_get_carnot_hp_streams(res["T_cond"], res["Q_cond"], res["T_evap"], res["Q_evap"], args))
    return HeatPumpTargetOutputs.model_validate(res)


def _parse_multi_temperature_carnot_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels.
    """
    # Get condensing temperatures and available duties at x
    T_cond = _map_x_to_T(x[:int(args.n_cond)], args.T_cold[0], args.T_cold[-1])
    Q_cond = _get_Q_vals_from_T_vals(T_cond, args.T_cold, args.H_cold, is_cond=True)
    # Get evaproating temperatures and available duties at x
    T_evap = _map_x_to_T(x[int(args.n_cond):], args.T_hot[0], args.T_hot[-1])
    Q_evap = _get_Q_vals_from_T_vals(T_evap, args.T_hot, args.H_hot, is_cond=False)    
    return T_cond, Q_cond, T_evap, Q_evap


def _compute_multi_temperature_carnot_hp_opt_obj(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate compressor work for a candidate multi-temperature Carnot HP placement defined by vector `x`.
    """
    T_cond, Q_cond, T_evap, Q_evap = _parse_multi_temperature_carnot_hp_state_variables(x, args)

    # Determine the work of the heat pump based on the limiting side
    cop = _compute_COP_estimate_from_carnot_limit(T_cond, Q_cond, T_evap, Q_evap)
    work_hp_from_evap = Q_evap.sum() / (cop - 1)
    work_hp_from_cond = Q_cond.sum() / cop
    work_hp = min(work_hp_from_evap, work_hp_from_cond)

    # Scale the side that has excess duty available
    scale_cond = min(work_hp_from_evap / work_hp_from_cond, 1) if work_hp_from_cond > 0.0 else 0.0
    Q_cond *= scale_cond
    scale_evap = min(work_hp_from_cond / work_hp_from_evap, 1) if work_hp_from_evap > 0.0 else 0.0
    Q_evap *= scale_evap

    # Determine the key performance metrics of the heat pump
    Q_amb = _calc_Q_amb(Q_evap.sum(), np.abs(args.H_hot[-1]), args.Q_amb_max)
    g = cop * Q_evap.sum() - (cop - 1) * Q_cond.sum()
    p = g_ineq_penalty(g, rho=args.rho_penalty, form="square")
    Q_ext = max(args.Q_hp_target - Q_cond.sum(), 0.0) # e.g., direct electric heating
    obj = _calc_obj(work_hp, Q_ext, args.Q_hp_target, args.price_ratio, p)

    if debug: # If in debug mode, plot the graph immediately
        res = _get_carnot_hp_streams(T_cond, Q_cond, T_evap, Q_evap, args)
        plot_multi_hp_profiles_from_results(
            args.T_hot, args.H_hot, args.T_cold, args.H_cold, res["hp_hot_streams"], res["hp_cold_streams"], 
            title=f"Obj {float(obj):.5f} = {(work_hp / args.Q_hp_target):.5f} + {(Q_ext / args.Q_hp_target):.5f} + {(g / args.Q_hp_target):.5f}"
        )        

    return {
        "obj": obj, 
        "utility_tot": work_hp + Q_ext, 
        "work_hp": work_hp,  
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "Q_cond": Q_cond,            
        "T_evap": T_evap, 
        "Q_evap": Q_evap, 
        "cop": cop,
        "Q_amb": Q_amb,
        "opt_success": True,
    }


#######################################################################################################
# Helper functions: Optimise cascade heat pump placement with CoolProp simulation
#######################################################################################################

@timing_decorator
def _optimise_cascade_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HeatPumpTargetOutputs:
    """Optimise the integration of a cascade heat pump with a specified number of condensers and evaporators.
    """
    num_stages = int(args.n_cond + args.n_evap - 1)

    # Use as initialisation
    if args.initialise_simulated_hp:
        init_res = _optimise_multi_temperature_carnot_heat_pump_placement(args)

    # Validate specific inputs related to this approach
    args.refrigerant_ls = _validate_vapour_hp_refrigerant_ls(num_stages, args)

    # Prepare and run the optimisation
    bnds = _get_bounds_for_cascade_hp_opt(args)
    
    x0_ls = _get_x0_for_cascade_hp_opt(
        T_cond=init_res.T_cond,
        Q_heat=init_res.Q_cond,
        T_evap=init_res.T_evap,
        Q_cool=init_res.Q_evap,
        args=args,
        bnds=bnds,
    ) if args.initialise_simulated_hp else None

    local_minima_x = multiminima(
        func=_compute_cascade_hp_system_performance,
        x0_ls=x0_ls,        
        func_kwargs=args,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,        
    )   

    if local_minima_x.size > 0:
        res = _compute_cascade_hp_system_performance(local_minima_x[0], args, debug=args.debug)
        res["opt_success"] = True
    else:
        raise ValueError("Optimal placement of cascade vapour-compression heat pump failed.")

    return HeatPumpTargetOutputs.model_validate(res)


def _get_x0_for_cascade_hp_opt(
    T_cond: np.ndarray, 
    Q_heat: np.ndarray, 
    T_evap: np.ndarray,
    Q_cool: np.ndarray,
    args: HeatPumpTargetInputs,
    bnds: list,
) -> np.ndarray:
    """Build initial guess vectors for the cascade condenser/evaporator optimization."""
    x_sc_bnds = [
        np.float64(args.dt_phase_change / args.dt_range_max), 
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    ]

    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds[1] = x_sc_bnds[0]
    
    x_cond = (args.T_cold[0] - T_cond) / args.dt_range_max
    x_sc = np.array([(x_sc_bnds[0] + x_sc_bnds[1]) / 2 for _ in range(len(T_cond))])
    y_heat = Q_heat / args.Q_hp_target
    x_evap = (T_evap - args.T_hot[-1]) / args.dt_range_max
    y_cool = Q_cool / args.Q_hp_target

    x_ls = []
    for candidate in [x_cond, x_sc, y_heat, x_evap]:
        x_ls.extend(candidate[:])
    if y_cool.size > 1:
        x_ls.extend(y_cool[:-1])
        
    x0 = np.array(x_ls) 

    if x0.size != len(bnds):
        raise ValueError("Bounds size must match x0 size for cascade heat pump optimisation.")

    for i in range(x0.size):
        if not(bnds[i][0] <= x0[i] <= bnds[i][1]):
            x0[i] = (bnds[i][1] + bnds[i][0]) / 2
    return np.asarray([x0])


def _get_bounds_for_cascade_hp_opt(
    args: HeatPumpTargetInputs,
) -> list:
    """Build bounds for the cascade condenser/evaporator optimization."""
    x_cond_bnds = []
    x_evap_bnds = []
    for i, refrigerant in enumerate(args.refrigerant_ls):
        T_min, T_max = PropsSI("Tmin", refrigerant) - 273.15, PropsSI("Tmax", refrigerant) - 273.15
        if i + 1 <= args.n_cond:
            T_cond_bnds = np.array(
                (
                    min(args.T_cold[0], T_max - 1),
                    max(args.T_cold[-1] + args.dt_phase_change, T_min + 1),
                )
            )
            if T_cond_bnds[1] > T_cond_bnds[0]:
                T_cond_bnds[0] = T_cond_bnds[1]
            x_cond_bnds += [
                (
                    (args.T_cold[0] - T_cond_bnds[0]) / args.dt_range_max,
                    (args.T_cold[0] - T_cond_bnds[1]) / args.dt_range_max,
                )
            ]

        if i + 1 >= args.n_cond:
            T_evap_bnds = np.array(
                (
                    max(args.T_hot[-1], T_min + 1),
                    min(args.T_hot[0] - args.dt_phase_change, T_max - 1),
                )
            )
            if T_evap_bnds[0] > T_evap_bnds[1]:
                T_evap_bnds[1] = T_evap_bnds[0]
            x_evap_bnds += [
                (
                    (T_evap_bnds[0] - args.T_hot[-1]) / args.dt_range_max,
                    (T_evap_bnds[1] - args.T_hot[-1]) / args.dt_range_max,
                )
            ]

    x_sc_bnds = (
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    )
    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds = (x_sc_bnds[0], x_sc_bnds[0])

    y_heat_bnds = (0.0, 1.0)
    y_cool_bnds = (0.0, 1.0)

    bnds = []

    def _build_bounds(count, limits):
        if isinstance(limits, list):
            bnds.extend(limits[:count])
        else:
            bnds.extend([limits] * count)

    _build_bounds(args.n_cond, x_cond_bnds)
    _build_bounds(args.n_cond, x_sc_bnds)
    _build_bounds(args.n_cond, y_heat_bnds)
    _build_bounds(args.n_evap, x_evap_bnds)

    if args.n_evap > 1:
        _build_bounds(args.n_evap - 1, y_cool_bnds)

    return bnds


def _parse_cascade_hp_state_variables(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x.
    """
    n0, n1 = args.n_cond, args.n_evap 
    T_cond = np.array(args.T_cold[0] - x[:n0] * args.dt_range_max, dtype=np.float64)
    dT_subcool = np.array(x[n0:2*n0] * args.dt_range_max, dtype=np.float64)
    Q_heat = np.array(x[2*n0:3*n0] * args.Q_hp_target, dtype=np.float64)
    T_evap = np.array(x[3*n0:3*n0+n1] * args.dt_range_max + args.T_hot[-1], dtype=np.float64)
    temp = np.array(x[3*n0+n1:] * args.Q_hp_target, dtype=np.float64)
    Q_cool = np.concatenate([temp.tolist(), [np.nan]]) if temp.size > 0 else np.array(np.nan)
    return T_cond, dT_subcool, Q_heat, T_evap, Q_cool


def _compute_cascade_hp_system_performance(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Objective: minimise total compressor work for multi-HP configuration.
    """
    T_cond, dT_subcool, Q_heat, T_evap, Q_cool = _parse_cascade_hp_state_variables(x, args)
    
    hp = CascadeHeatPumpCycle()
    hp.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        Q_heat=Q_heat,
        Q_cool=Q_cool,
        dT_subcool=dT_subcool,
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dt_ihx_gas_side=args.dt_hp_ihx,
        dt_cascade_hx=args.dt_cascade_hx,
    )
    if not(hp.solved):
        obj = (hp.work / args.Q_hp_target + 1)
        return {"obj": obj}

    # Build streams based on heat pump profiles
    hp_hot_streams = hp.build_stream_collection(
        include_cond=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )    
    hp_cold_streams = hp.build_stream_collection(
        include_evap=True, 
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )

    # Determine the heat cascade
    pt_cond = get_process_heat_cascade(
        hot_streams=hp_hot_streams, 
        cold_streams=args.net_cold_streams,
        is_shifted=True,
    )    
    pt_evap = get_process_heat_cascade(
        hot_streams=args.net_hot_streams, 
        cold_streams=hp_cold_streams,
        is_shifted=True,        
    )

    # Calculate key perfromance indicators
    work_hp = hp.work
    cop = args.Q_hp_target / work_hp if work_hp > 0 else 1.0
    Q_amb = _calc_Q_amb(hp.Q_cool, np.abs(args.H_hot[-1]), args.Q_amb_max)
    
    Q_ext = pt_cond.col[PT.H_NET.value][0] # e.g., direct eletric heating
    g = np.array([pt_evap.col[PT.H_NET.value][0], pt_cond.col[PT.H_NET.value][-1], hp.penalty])
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    obj = _calc_obj(work_hp, Q_ext, args.Q_hp_target, args.price_ratio, p)

    if debug: # for debugging purposes, a quick plot function
        plot_multi_hp_profiles_from_results(
            args.T_hot, args.H_hot, args.T_cold, args.H_cold, hp_hot_streams, hp_cold_streams, 
            title=f"Obj {float(obj):.5f} = {(work_hp / args.Q_hp_target):.5f} + {(Q_ext / args.Q_hp_target):.5f} + {(g.sum() / args.Q_hp_target):.5f}"
        )

    return {
        "obj": obj, 
        "utility_tot": work_hp + Q_ext, 
        "work_hp": hp.work_arr,
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": hp.Q_heat_arr,          
        "T_evap": T_evap,
        "Q_cool": hp.Q_cool_arr,
        "cop": cop,
        "Q_amb": Q_amb,
        "hp_hot_streams": hp_hot_streams,
        "hp_cold_streams": hp_cold_streams,
        "hp_model": hp,
    }


#######################################################################################################
# Helper functions: Optimise multiple simple Carnot heat pump placement
#######################################################################################################

@timing_decorator
def _optimise_multi_simple_carnot_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HeatPumpTargetOutputs:
    """Compute baseline condenser/evaporator temperature levels and duties for a multiple simple heat pumps layout.
    """
    args.n_cond = args.n_evap = max(args.n_cond, args.n_evap)
    bnds = [(0.0, 1.0) for _ in range(args.n_cond + args.n_evap)]
    local_minima_x = multiminima(
        func=_compute_multi_simple_carnot_hp_opt_obj,
        func_kwargs=args,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,        
    )    
    res = _compute_multi_simple_carnot_hp_opt_obj(local_minima_x[0], args, debug=args.debug)
    res.update(_get_carnot_hp_streams(res["T_cond"], res["Q_cond"], res["T_evap"], res["Q_evap"], args))
    return HeatPumpTargetOutputs.model_validate(res)


def _parse_multi_simple_carnot_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels.
    """
    x_cond = x[:args.n_cond]
    x_evap = x[args.n_cond:]

    T_cond = _map_x_to_T(x_cond, args.T_cold[0], args.T_cold[-1])
    Q_cond = _get_Q_vals_from_T_vals(T_cond, args.T_cold, args.H_cold, is_cond=True)
    
    T_evap = args.T_hot[-1] + np.array(x_evap) * (args.T_hot[0] - args.T_hot[-1])

    return T_cond, Q_cond, T_evap


def _compute_multi_simple_carnot_hp_opt_obj(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate compressor work for a candidate multiple single Carnot HP placement defined by vector `x`.
    """
    T_cond, Q_cond, T_evap = _parse_multi_simple_carnot_hp_state_variables(x, args)

    if Q_cond.sum() < tol:
        obj = _calc_obj(args.Q_hp_target, 0.0, args.Q_hp_target, args.price_ratio, 0.0)
        return {"obj": obj}
    
    # Determine the work of each heat pump (or heat engine)
    delta_T_lift = T_cond - T_evap
    is_hp = delta_T_lift > 0.0
    is_he = (delta_T_lift < 0.0) & args.allow_integrated_expander
    work_hp = np.zeros_like(Q_cond)
    work_he = np.zeros_like(Q_cond)
    if np.any(is_hp):
        cop_hp = (T_evap[is_hp] + 273.15) / delta_T_lift[is_hp] * args.eta_hp_carnot + 1
        work_hp[is_hp] = Q_cond[is_hp] / cop_hp     
    if np.any(is_he):
        eff_he = (delta_T_lift[is_he] / T_cond[is_he]) * args.eta_he_carnot
        work_he[is_he] = Q_cond[is_he] * eff_he

    # Calculate evaporator duty
    Q_evap = Q_cond - work_hp + work_he
    Q_evap_max = _get_Q_vals_from_T_vals(T_evap, args.T_hot, args.H_hot, is_cond=False)

    # Determine the key performance metrics of the heat pump
    Q_amb = _calc_Q_amb(Q_evap.sum(), np.abs(args.H_hot[-1]), args.Q_amb_max)
    g = Q_evap_max - Q_evap
    p = g_ineq_penalty(g, rho=args.rho_penalty, form="square")
    Q_ext = max(args.Q_hp_target - Q_cond.sum(), 0.0) # e.g., direct electric heating
    obj = _calc_obj(work_hp.sum() - work_he.sum(), Q_ext, args.Q_hp_target, args.price_ratio, p)

    if debug: # If in debug mode, plot the graph immediately
        res = _get_carnot_hp_streams(T_cond, Q_cond, T_evap, Q_evap, args)     
        plot_multi_hp_profiles_from_results(
            args.T_hot, args.H_hot, args.T_cold, args.H_cold, res["hp_hot_streams"], res["hp_cold_streams"], 
            title=f"Obj {float(obj):.5f} = {((work_hp.sum() - work_he.sum()) / args.Q_hp_target):.5f} + {(Q_ext / args.Q_hp_target):.5f} + {(g.sum() / args.Q_hp_target):.5f}"
        )

    return {
        "obj": obj, 
        "utility_tot": work_hp.sum() - work_he.sum() + Q_ext, 
        "work_hp": work_hp.sum(),
        "work_he": work_he.sum(),
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "Q_cond": Q_cond,            
        "T_evap": T_evap, 
        "Q_evap": Q_evap, 
        "cop": Q_cond.sum() / work_hp.sum() if work_hp.sum() > tol else 1.0,
        "Q_amb": Q_amb,
        "opt_success": True,
    }


#######################################################################################################
# Helper functions: Optimise multi simple heat pumps placement with CoolProp simulation
#######################################################################################################

@timing_decorator
def _optimise_multi_simple_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HeatPumpTargetOutputs:
    """Optimise the integration of multiple single heat pump units with a specified number of units.
    """
    num_stages = args.n_cond = args.n_evap = int(max(args.n_cond, args.n_evap))
    
    # Use as initialisation
    if args.initialise_simulated_hp:
        init_res = _optimise_multi_simple_carnot_heat_pump_placement(args)

    # Validate specific inputs related to this approach
    args.refrigerant_ls = _validate_vapour_hp_refrigerant_ls(num_stages, args)

    # Prepare and run the optimisation
    bnds = _get_bounds_for_multi_single_hp_opt(args)
    x0 = _get_x0_for_multi_single_hp_opt(
        T_cond=init_res.T_cond, 
        Q_cond=init_res.Q_cond, 
        T_evap=init_res.T_evap, 
        args=args, 
        bnds=bnds,
    ) if args.initialise_simulated_hp else None

    local_minima_x = multiminima(
        func=_compute_multi_simple_hp_system_performance,
        x0_ls=x0,        
        func_kwargs=args,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,        
    )   

    if local_minima_x.size > 0:
        res = _compute_multi_simple_hp_system_performance(local_minima_x[0], args, debug=args.debug)
        res["opt_success"] = True
    else:
        raise ValueError("Optimal placement of multiple vapour-compression units failed.")            

    return HeatPumpTargetOutputs.model_validate(res)


def _get_x0_for_multi_single_hp_opt(
    T_cond: np.ndarray, 
    Q_cond: np.ndarray, 
    T_evap: np.ndarray,
    args: HeatPumpTargetInputs,
    bnds: list,
) -> np.ndarray:
    """Build initial guesses for the multi single-HP condenser/evaporator optimization."""
    x_sc_bnds = [
        np.float64(args.dt_phase_change / args.dt_range_max), 
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    ]
    x_sh_bnds = [
        np.float64(args.dt_phase_change / args.dt_range_max), 
        np.float64((args.T_hot[0]  - args.T_hot[-1] ) / args.dt_range_max),
    ]

    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds[1] = x_sc_bnds[0]
    
    if x_sh_bnds[0] > x_sh_bnds[1]:
        x_sh_bnds[1] = x_sh_bnds[0]

    x_cond = (args.T_cold[0] - T_cond) / args.dt_range_max
    x_sc = np.array([args.dt_phase_change / args.dt_range_max for _ in range(len(T_cond))])
    y_cond = Q_cond / args.Q_hp_target

    x_evap = (T_evap - args.T_hot[-1]) / args.dt_range_max
    x_sh = np.array([args.dt_phase_change / args.dt_range_max for _ in range(len(T_evap))])

    x_ls = []
    for candidate in [x_cond, x_sc, y_cond, x_evap, x_sh]:
        x_ls.extend(candidate[:])
        
    x0 = np.array(x_ls) 

    if x0.size != len(bnds):
        raise ValueError("Bounds size must match x0 size for multiple single HP optimisation.")

    for i in range(x0.size):
        if not(bnds[i][0] <= x0[i] <= bnds[i][1]):
            x0[i] = bnds[i][1] if i * 2 < x0.size else bnds[i][0]

    return np.asarray([x0])


def _get_bounds_for_multi_single_hp_opt(
    args: HeatPumpTargetInputs,
) -> list:
    """Build bounds for the multi single-HP condenser/evaporator optimization."""
    x_cond_bnds = []
    x_evap_bnds = []
    for i, refrigerant in enumerate(args.refrigerant_ls):
        T_min, T_max = PropsSI("Tmin", refrigerant) - 273.15, PropsSI("Tmax", refrigerant) - 273.15
        if i == args.n_cond:
            break

        T_cond_bnds = np.array(
            (
                min(args.T_cold[0], T_max - 1),
                max(args.T_cold[-1] + args.dt_phase_change, T_min + 1),
            )
        )
        if T_cond_bnds[1] > T_cond_bnds[0]:
            T_cond_bnds[0] = T_cond_bnds[1]
        x_cond_bnds += [
            (
                (args.T_cold[0] - T_cond_bnds[0]) / args.dt_range_max,
                (args.T_cold[0] - T_cond_bnds[1]) / args.dt_range_max,
            )
        ]

        T_evap_bnds = np.array(
            (
                max(args.T_hot[-1], T_min + 1),
                min(args.T_hot[0] - args.dt_phase_change, T_max - 1),
            )
        )
        if T_evap_bnds[0] > T_evap_bnds[1]:
            T_evap_bnds[1] = T_evap_bnds[0]
        x_evap_bnds += [
            (
                (T_evap_bnds[0] - args.T_hot[-1]) / args.dt_range_max,
                (T_evap_bnds[1] - args.T_hot[-1]) / args.dt_range_max,
            )
        ]

    x_sc_bnds = (
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    )
    x_sh_bnds = (
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_hot[0] - args.T_hot[-1]) / args.dt_range_max),
    )

    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds = (x_sc_bnds[0], x_sc_bnds[0])
    if x_sh_bnds[0] > x_sh_bnds[1]:
        x_sh_bnds = (x_sh_bnds[0], x_sh_bnds[0])

    y_cond_bnds = (0.0, 1.0)

    bnds = []

    def _build_bounds(count, limits):
        if isinstance(limits, list):
            bnds.extend(limits[:count])
        else:
            bnds.extend([limits] * count)

    _build_bounds(args.n_cond, x_cond_bnds)
    _build_bounds(args.n_cond, x_sc_bnds)
    _build_bounds(args.n_cond, y_cond_bnds)
    _build_bounds(args.n_cond, x_evap_bnds)
    _build_bounds(args.n_cond, x_sh_bnds)

    return bnds


def _constrain_min_temperature_lift(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
):
    """Ensure T_cond - dT_subcool > T_evap + dtcont_hp
    """
    T_cond, dT_subcool, _, T_evap, _ = _parse_multi_simple_hp_state_temperatures(x, args)
    T_diff = T_cond - dT_subcool - T_evap - args.dtcont_hp
    return T_diff.min()


def _parse_multi_simple_hp_state_temperatures(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x.
    """
    n = args.n_cond 
    T_cond = np.array(args.T_cold[0] - x[:n] * args.dt_range_max, dtype=np.float64)
    dT_subcool = np.array(x[n:2*n] * args.dt_range_max, dtype=np.float64)
    Q_cond = np.array(x[2*n:3*n] * args.Q_hp_target, dtype=np.float64)
    T_evap = np.array(x[3*n:4*n] * args.dt_range_max + args.T_hot[-1], dtype=np.float64)
    dT_superheat = np.array(x[4*n:] * args.dt_range_max, dtype=np.float64)
    return T_cond, dT_subcool, Q_cond, T_evap, dT_superheat


def _compute_multi_simple_hp_system_performance(
    x: np.ndarray, 
    args: HeatPumpTargetInputs, 
    debug: bool = False,
) -> dict:
    """Objective: minimize total compressor work for multi-HP configuration.
    """
    T_cond, dT_subcool, Q_heat, T_evap, dT_superheat = _parse_multi_simple_hp_state_temperatures(x, args)

    if _constrain_min_temperature_lift(x, args) < 0:
        return {"obj": np.inf}

    hp = MultiSimpleHeatPumpCycle()
    hp.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dt_ihx_gas_side=args.dt_hp_ihx,
        Q_heat=Q_heat,
    )

    # Build streams based on heat pump profiles
    hp_hot_streams = hp.build_stream_collection(
        include_cond=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )
    hp_cold_streams = hp.build_stream_collection(
        include_evap=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )

    # Determine the heat cascade
    pt_cond = get_process_heat_cascade(
        hot_streams=hp_hot_streams, 
        cold_streams=args.net_cold_streams,
        is_shifted=True,
    )    
    pt_evap = get_process_heat_cascade(
        hot_streams=args.net_hot_streams, 
        cold_streams=hp_cold_streams,
        is_shifted=True,        
    )

    # Calculate key perfromance indicators
    work_hp = hp.work
    cop = args.Q_hp_target / work_hp if work_hp > 0 else 1.0 
    Q_amb = _calc_Q_amb(hp.Q_cool, np.abs(args.H_hot[-1]), args.Q_amb_max)
    
    g = np.array([pt_evap.col[PT.H_NET.value][0], pt_cond.col[PT.H_NET.value][-1], hp.penalty])
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    Q_ext = pt_cond.col[PT.H_NET.value][0] # Alternative to heat pump, an external heat source
    obj = _calc_obj(work_hp, Q_ext, args.Q_hp_target, args.price_ratio, p)

    # For debugging purposes, a quick plot function
    if debug:
        plot_multi_hp_profiles_from_results(
            args.T_hot, args.H_hot, args.T_cold, args.H_cold, hp_hot_streams, hp_cold_streams, 
            title=f"Obj {float(obj):.5f} = {(work_hp / args.Q_hp_target):.5f} + {(Q_ext / args.Q_hp_target):.5f} + {(g.sum() / args.Q_hp_target):.5f}"
        )

    return {
        "obj": obj, 
        "utility_tot": work_hp + Q_ext, 
        "work_hp": hp.work_arr,  
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": hp.Q_heat_arr,            
        "T_evap": T_evap, 
        "dT_superheat": dT_superheat,
        "Q_cool": hp.Q_cool_arr,
        "cop": cop,
        "Q_amb": Q_amb,
        "hp_hot_streams": hp_hot_streams,
        "hp_cold_streams": hp_cold_streams,
        "hp_model": hp,        
    }


#######################################################################################################
# Helper functions: Optimise single Brayton heat pump placement - TODO
#######################################################################################################

@timing_decorator
def _optimise_brayton_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HeatPumpTargetOutputs:
    """Optimise a single Brayton heat pump against a given composite curve.
    """
    args.n_cond = args.n_evap = 1 # Must be one gas cooler and one gas heater
    args.refrigerant_ls = ["air"]

    x0 = _get_x0_for_brayton_hp_opt(args)
    bnds = _get_bounds_for_brayton_hp_opt(args)
    opt = minimize(
        fun=lambda x: _compute_brayton_hp_system_performance(x, args)["obj"],
        x0=x0,
        method="SLSQP",
        bounds=bnds,
        options={'disp': False, 'maxiter': 1000},
        tol=1e-7,
    )

    res = None
    if opt.success:
        res = _compute_brayton_hp_system_performance(opt.x, args)
        res["opt_success"] = opt.success
        if 0:   
            plot_multi_hp_profiles_from_results(args.T_hot, args.H_hot, args.T_cold, args.H_cold, res["hp_hot_streams"], res["hp_cold_streams"])
    else:
        raise ValueError(f"Brayton heat pump targeting failed: {opt.message}")

    return HeatPumpTargetOutputs.model_validate(res)


def _get_x0_for_brayton_hp_opt(
    args: HeatPumpTargetInputs,
) -> list:
    """Build initial guesses for Brayton HP optimisation."""
    x0 = [
        0.0,
        abs(args.T_cold[0] - args.T_hot[0]) / args.dt_range_max,
        abs(args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max,
        1.0
    ]
    return x0


def _get_bounds_for_brayton_hp_opt(
    args: HeatPumpTargetInputs,
) -> list:
    """Build bounds for Brayton HP optimisation."""
    return [
        (-0.2, 1.0), # T_comp_out = T_cold_max + x[0] * dT_range_max
        (0.01, 1.5), # dT_comp = x[1] * dT_range_max
        (0.01, 1.5), # dT_gc = x[2] * dT_range_max
        (0.01, 1.0), # Q_heat = x[3] * Q_hp_target
    ]  


def _parse_brayton_hp_state_variables(
    x: np.ndarray, 
    args: HeatPumpTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x.
    """
    T_comp_out = np.array(args.T_cold[0] + x[0] * args.dt_range_max, dtype=np.float64)
    dT_comp = np.array(x[1] * args.dt_range_max, dtype=np.float64)
    dT_gc = np.array(x[2] * args.dt_range_max, dtype=np.float64)
    Q_heat = np.array(x[3] * args.Q_hp_target, dtype=np.float64)
    return [T_comp_out], [dT_comp], [dT_gc], [Q_heat]


def _compute_brayton_hp_system_performance(
    x: np.ndarray, 
    args: HeatPumpTargetInputs
) -> float:
    """Objective: minimize total compressor work for multi-HP configuration.
    """
    T_comp_out, dT_comp, dT_gc, Q_heat = _parse_brayton_hp_state_variables(x, args)
    
    hp_list = _create_brayton_hp_list(
        T_comp_out=T_comp_out, 
        dT_comp=dT_comp, 
        dT_gc=dT_gc, 
        Q_gc=Q_heat,
        args=args,
    )
    
    hp_hot_streams, hp_cold_streams = _build_simulated_hps_streams(hp_list)
    
    T_exp_out = hp_list[0].cycle_states[3]['T']

    pt_gas_cooler = get_process_heat_cascade(
        hot_streams=hp_hot_streams, 
        cold_streams=args.net_cold_streams,
        is_shifted=True,
    )    
    pt_gas_heater = get_process_heat_cascade(
        hot_streams=args.net_hot_streams, 
        cold_streams=hp_cold_streams,
        is_shifted=True,        
    )

    work_hp = sum([hp.work_net for hp in hp_list])
    c = (pt_gas_cooler.col[PT.H_NET.value][-1] + pt_gas_heater.col[PT.H_NET.value][0]) * 10 # Penalty for supplying too much heat
    Q_ext = pt_gas_cooler.col[PT.H_NET.value][0] # Extra heating required on either side of the gcc 
    Q_cool = np.array([hp.Q_cool for hp in hp_list])
    COP = (args.Q_hp_target - Q_ext) / (work_hp + 1e-9)
    Q_amb = _calc_Q_amb(Q_cool.sum(), np.abs(args.H_hot[-1]), args.Q_amb_max)
    obj = _calc_obj(work_hp, Q_ext, args.Q_hp_target, penalty=c)

    if 0:
        plot_multi_hp_profiles_from_results(pt_gas_cooler.col[PT.T.value], pt_gas_cooler.col[PT.H_NET.value])
        plot_multi_hp_profiles_from_results(pt_gas_heater.col[PT.T.value], pt_gas_heater.col[PT.H_NET.value])
        plot_multi_hp_profiles_from_results(args.T_hot, args.H_hot, args.T_cold, args.H_cold, hp_hot_streams, hp_cold_streams, title=f"T_hi {T_comp_out} -> {float(obj), float(c / args.Q_hp_target)}")

    return {
        "obj": obj, 
        "utility_tot": work_hp + Q_ext, 
        "work_hp": work_hp,  
        "Q_ext": Q_ext,
        "T_comp_out": np.array(T_comp_out),
        "dT_gc": np.array(dT_gc),
        "Q_heat": np.array(Q_heat),            
        "T_evap": np.array(T_exp_out), 
        "dT_comp": np.array(dT_comp),
        "Q_cool": np.array(Q_cool),
        "cop": COP,
        "Q_amb": Q_amb,
        "hp_hot_streams": hp_hot_streams,
        "hp_cold_streams": hp_cold_streams,
        "hp_model": hp_list,        
    }


def _create_brayton_hp_list(
    T_comp_out: np.ndarray, 
    dT_gc: np.ndarray, 
    Q_gc: np.ndarray, 
    dT_comp: np.ndarray, 
    args: HeatPumpTargetInputs,
) -> List[SimpleBraytonHeatPumpCycle]:
    """Instantiate SimpleBrytonHeatPumpCycle objects.
    """
    hp_list = []
    n_hp = args.n_cond
    for i in range(n_hp):
        hp = SimpleBraytonHeatPumpCycle()
        hp.solve(
            T_comp_out=T_comp_out[i],
            T_comp_in=T_comp_out[i] - dT_comp[i],
            dT_gc=dT_gc[i],
            Q_heat=Q_gc[i],
            eta_comp=args.eta_comp,
            eta_exp=args.eta_exp,
            is_recuperated=False,
            refrigerant=args.refrigerant_ls[0],
        )
        hp_list.append(hp)
    return hp_list


#######################################################################################################
# Helper functions: other / non-specific
#######################################################################################################


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


def _get_Q_vals_from_T_vals(
    T_hp: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *,
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
    *,   
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
    *,
    eff: float = 0.5,
    min_dt_lift: float = 0.1,          
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


def _map_x_to_T(
    x: np.ndarray,
    T_0: float, 
    T_1: float,
) -> np.ndarray:
    """Recover evaporator temperatures from normalized x values.
    """    
    temp = []
    for i in range(x.size):
        temp.append(T_0 - x[i] * (T_0 - T_1))
        T_0 = temp[-1]
    return np.sort(np.array(temp).flatten())[::-1]


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


def _validate_vapour_hp_refrigerant_ls(
    num_stages: int,
    args: HeatPumpTargetInputs,    
) -> list:
    """Validate and normalize refrigerant list length/order for staged cycles.

    Parameters
    ----------
    num_stages : int
        Required number of refrigerant entries.
    args : HeatPumpTargetInputs
        Optimisation input bundle containing refrigerant options.

    Returns
    -------
    list
        Refrigerant names with length exactly ``num_stages``. When insufficient
        inputs are provided, the last refrigerant is repeated. When no
        refrigerants are provided, ``"water"`` is used for all stages.
    """
    if len(args.refrigerant_ls) > 0:
        if args.do_refrigerant_sort:
            refrigerants = [
                ref for ref, _ in sorted(
                    ((ref, PropsSI("Tcrit", ref)) for ref in args.refrigerant_ls),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
        else:
            refrigerants = args.refrigerant_ls
        if num_stages <= len(refrigerants):
            return refrigerants[:num_stages]

        padding = [refrigerants[-1]] * (num_stages - len(refrigerants))
        return refrigerants + padding
    return ["water" for _ in range(num_stages)]


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


def _get_carnot_hp_streams(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    args: HeatPumpTargetInputs,
) -> dict:
    """Build condenser and evaporator stream collections from Carnot outputs.

    Parameters
    ----------
    T_cond, Q_cond : np.ndarray
        Condenser temperature levels and duties.
    T_evap, Q_evap : np.ndarray
        Evaporator temperature levels and duties.
    args : HeatPumpTargetInputs
        Optimisation input bundle containing profile settings.

    Returns
    -------
    dict
        Mapping with ``hp_hot_streams`` and ``hp_cold_streams``.
    """
    return {
        "hp_hot_streams": _build_latent_streams(T_cond, args.dt_phase_change, Q_cond, is_hot=True),
        "hp_cold_streams": _build_latent_streams(T_evap, args.dt_phase_change, Q_evap, is_hot=False),
    }
    

def _build_latent_streams(
    T_ls: np.ndarray, 
    dT_phase_change: float,
    Q_ls: np.ndarray,
    *,
    dt_cont: float = 0.0,
    is_hot: bool = True,
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


def _build_simulated_hps_streams(
    hp_list,
    *,
    is_process_stream: bool = False,
    include_cond: bool = False,
    include_evap: bool = False,
    dtcont_hp: float = 0.0,
) -> StreamCollection:
    """Aggregate condenser/gas-cooler and evaporator/gas-heater streams for each simulated HP cycle."""
    hp_streams = StreamCollection()
    for hp in hp_list:
        hp_streams.add_many(
            hp.build_stream_collection(
                include_cond=include_cond, 
                include_evap=include_evap, 
                is_process_stream=is_process_stream,
                dtcont=dtcont_hp,
            )
        )
    return hp_streams


def _get_ambient_air_stream(
    Q_amb: float,
    args: HeatPumpTargetInputs,
) -> StreamCollection:  
    """Build ambient-air source/sink stream representation from ambient duty.

    Parameters
    ----------
    Q_amb : float
        Net ambient duty. Positive indicates ambient as heat source, negative as
        sink.
    args : HeatPumpTargetInputs
        Optimisation input bundle containing ambient settings.

    Returns
    -------
    StreamCollection
        Ambient stream collection or an empty collection when duty is near zero.
    """
    if -tol < Q_amb < tol:
        return StreamCollection()
    else:
        return _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb]),
            dt_cont=args.dt_env_cont,
            is_hot=True if Q_amb >= -tol else False,
            is_process_stream=True,
            prefix="AIR",
        )     


def _calc_obj(
    work_hp: float,
    Q_ext: float,
    Q_hp_target: float,
    price_ratio: float = 1.0,
    penalty: float = 0.0,
) -> float:
    """Return the normalized objective value."""
    return (work_hp + Q_ext / price_ratio + penalty) / Q_hp_target


def _calc_Q_amb(
    Q_evap_total: float,
    H_hot_limit: float,
    Q_amb_max: float,
) -> float:
    """Return ambient heat exchange required beyond process hot availability."""
    return max(Q_evap_total - (H_hot_limit - Q_amb_max), 0.0)


#######################################################################################################
# Helper functions: prepare args and get intial placement
#######################################################################################################


_HP_PLACEMENT_HANDLERS = {
    HeatPumpType.Brayton.value: _optimise_brayton_heat_pump_placement,
    HeatPumpType.MultiTempCarnot.value: _optimise_multi_temperature_carnot_heat_pump_placement,
    HeatPumpType.MultiSimpleVapourComp.value: _optimise_multi_simple_heat_pump_placement,
    HeatPumpType.CascadeVapourComp.value: _optimise_cascade_heat_pump_placement,
    HeatPumpType.MultiSimpleCarnot.value: _optimise_multi_simple_carnot_heat_pump_placement,
}

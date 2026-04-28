"""Heat pump targeting and cascade construction utilities for composite curves."""

from ast import literal_eval
from typing import Callable
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
from ..classes.cascade_vapour_compression_cycle import CascadeVapourCompressionCycle
from ..classes.parallel_vapour_compression_cycles import ParallelVapourCompressionCycles

__all__ = [
    "validate_heat_pump_or_refrigeration_targeting_required",
    "get_heat_pump_and_refrigeration_targets",
    "calc_heat_pump_and_refrigeration_cascade",
    "plot_multi_hp_profiles_from_results",
]


#######################################################################################################
# Public API
#######################################################################################################


def validate_heat_pump_or_refrigeration_targeting_required(
    pt: ProblemTable,
    is_heat_pumping: bool = False,
    is_refrigeration: bool = False,
    zone_name: str = None,
    zone_config: dict = Configuration(),
    r: dict | float | int = None,
) -> float:
    """Return the requested heat-pump or refrigeration load capped by feasibility.

    Parameters
    ----------
    pt : ProblemTable
        Problem table containing the background process cascade used to derive
        the maximum recoverable heating or cooling duty.
    hpr_load : dict | float
        User-specified target load. A scalar applies globally, while a
        dictionary is resolved per zone via ``zone_name``.
    is_heat_pumping : bool
        Whether heat-pump targeting is enabled for the zone. When ``True``, the
        available duty is limited by the largest magnitude cold-side cascade
        deficit.
    is_refrigeration : bool
        Whether refrigeration targeting is enabled for the zone. When ``True``,
        the available duty is limited by the largest magnitude hot-side cascade
        surplus.
    zone_name : str
        Name of the active zone used when resolving zone-specific load values.

    Returns
    -------
    float
        Requested target load clipped to the maximum duty supported by the
        cascade profiles. Returns ``0.0`` when neither targeting mode is
        enabled.
    """
    if is_heat_pumping == True:
        Q_max = np.abs(pt.col[PT.H_NET_COLD.value]).max()
    elif is_refrigeration == True:
        Q_max = np.abs(pt.col[PT.H_NET_HOT.value]).max()
    else:
        return 0.0

    Q = Q_max
    hpr_load = zone_config.HPR_LOAD_VALUE if r is None else r
    if isinstance(hpr_load, float | int):
        Q = Q_max * hpr_load
    elif isinstance(hpr_load, dict):
        Q = min(
            get_value(hpr_load, val2=Q_max, zone_name=zone_name),
            Q_max,
        )
    elif isinstance(hpr_load, str):
        hpr_load = literal_eval(hpr_load.strip())
        if isinstance(hpr_load, float | int | dict):
            Q = validate_heat_pump_or_refrigeration_targeting_required(
                pt=pt,
                is_heat_pumping=is_heat_pumping,
                is_refrigeration=is_refrigeration,
                zone_name=zone_name,
                zone_config=zone_config,
                r=hpr_load,
            )
    return Q


def get_heat_pump_and_refrigeration_targets(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    zone_config: Configuration,
    is_heat_pumping: bool,
) -> HPRTargetOutputs:
    """Optimise heat pump placement for a target duty and cascade profiles.

    Parameters
    ----------
    Q_hpr_target : float
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
    is_heat_pumping : bool
        ``True`` for heating mode, ``False`` for refrigeration mode.

    Returns
    -------
    HPRTargetOutputs
        Validated optimisation output, including objective terms, cycle state
        variables, and generated stream collections.

    Raises
    ------
    ValueError
        If ``zone_config.HPR_TYPE`` does not map to a supported optimiser.
    """
    zone_config.HPR_TYPE = HPRcycle.MultiTempCarnot.value
    zone_config.N_COND = 2
    zone_config.N_EVAP = 2
    args = _construct_HPRTargetInputs(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_vals,
        H_hot=np.abs(H_hot) * -1,
        H_cold=np.abs(H_cold),
        is_heat_pumping=is_heat_pumping,
        zone_config=zone_config,
        debug=True,
    )
    # Select the appropriate targeting handler based on the specified heat pump targeting approach
    handler = _HP_PLACEMENT_HANDLERS.get(zone_config.HPR_TYPE)
    if handler is None:
        raise ValueError("No valid heat pump targeting type selected.")
    res = handler(args)
    return HPRTargetOutputs.model_validate(res)


def calc_heat_pump_and_refrigeration_cascade(
    pt: ProblemTable,
    res: HPRTargetOutputs,
    is_T_vals_shifted: bool,
    is_heat_pumping: bool,
) -> ProblemTable:
    """Insert heat pump cascade contributions into a problem table.

    Parameters
    ----------
    pt : ProblemTable
        Base problem table to augment. This object is modified in place.
    res : HPRTargetOutputs
        Heat-pump targeting result containing condenser, evaporator, and ambient
        stream collections.
    is_T_vals_shifted : bool
        Whether ``pt`` temperatures are already on the shifted temperature scale.

    Returns
    -------
    ProblemTable
        The same ``pt`` instance, after insertion of HP intervals and update of
        HP- or RFRG-related columns.
    """
    pt_hp = create_problem_table_with_t_int(
        streams=res.hpr_hot_streams + res.hpr_cold_streams,
        is_shifted=is_T_vals_shifted,
    )
    pt.insert_temperature_interval(pt_hp[PT.T.value].to_list())
    temp = get_utility_heat_cascade(
        T_int_vals=pt.col[PT.T.value],
        hot_utilities=res.hpr_hot_streams,
        cold_utilities=res.hpr_cold_streams,
        is_shifted=is_T_vals_shifted,
    )
    if is_heat_pumping:
        pt.update(
            {
                PT.H_NET_HP.value: temp[PT.H_NET_UT.value],
                PT.H_HOT_HP.value: temp[PT.H_HOT_UT.value],
                PT.H_COLD_HP.value: temp[PT.H_COLD_UT.value],
            }
        )
    else:  # is refrigeration
        pt.update(
            {
                PT.H_NET_RFRG.value: temp[PT.H_NET_UT.value],
                PT.H_HOT_RFRG.value: temp[PT.H_HOT_UT.value],
                PT.H_COLD_RFRG.value: temp[PT.H_COLD_UT.value],
            }
        )

    # Calculate ambient air portion for the heat pump cascade
    if len(res.amb_streams) > 0:
        pt_air = get_process_heat_cascade(
            hot_streams=res.amb_streams.get_hot_streams(),
            cold_streams=res.amb_streams.get_cold_streams(),
            is_shifted=True,
        )
        T_ls = pt[PT.T.value].to_list() + pt_air[PT.T.value].to_list()
        pt_air.insert_temperature_interval(T_ls)
        pt.insert_temperature_interval(T_ls)
        pt.col[PT.H_NET_W_AIR.value] = (
            pt.col[PT.H_NET_A.value] + pt_air.col[PT.H_NET.value]
        )

        # Adjust hot/cold columns to reflect ambient contributions, if any
        if res.Q_amb_hot > tol:
            pt.col[PT.H_NET_HOT.value] -= pt_air.col[PT.H_NET.value]
        elif res.Q_amb_cold > tol:
            pt.col[PT.H_NET_COLD.value] += pt_air.col[PT.H_NET.value]
        else:
            pass  # amb stream has zero duty, so no need to adjust hot/cold columns
    else:
        pt.col[PT.H_NET_W_AIR.value] = pt.col[PT.H_NET_A.value]

    return pt


def plot_multi_hp_profiles_from_results(
    T_hot: np.ndarray = None,
    H_hot: np.ndarray = None,
    T_cold: np.ndarray = None,
    H_cold: np.ndarray = None,
    hpr_hot_streams: StreamCollection = None,
    hpr_cold_streams: StreamCollection = None,
    title: str = None,
) -> None:
    """Plot source/sink composites together with HP condenser/evaporator profiles.

    Parameters
    ----------
    T_hot, H_hot : np.ndarray, optional
        Temperature and heat-flow arrays for the sink-side composite.
    T_cold, H_cold : np.ndarray, optional
        Temperature and heat-flow arrays for the source-side composite.
    hot_streams, cold_streams : StreamCollection, optional
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

    if hpr_hot_streams is not None and hpr_cold_streams is not None:
        T_hpr_arr, H_hpr_hot, H_hpr_cold = _get_hpr_cascade(
            hpr_hot_streams, hpr_cold_streams
        )
        T_hpr_hot, H_hpr_hot = clean_composite_curve_ends(T_hpr_arr, H_hpr_hot)
        T_hpr_cold, H_hpr_cold = clean_composite_curve_ends(T_hpr_arr, H_hpr_cold)
        plt.plot(
            H_hpr_hot,
            T_hpr_hot,
            "--",
            color="darkred",
            linewidth=1.8,
            label="Condenser",
        )
        plt.plot(
            H_hpr_cold,
            T_hpr_cold,
            "--",
            color="darkblue",
            linewidth=1.8,
            label="Evaporator",
        )

    plt.title(title)
    plt.xlabel("Heat Flow / kW")
    plt.ylabel("Temperature / degC")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.axvline(0.0, color="black", linewidth=2)
    plt.tight_layout()
    plt.show()


#######################################################################################################
# Helper functions: prepare args and get intial placement
#######################################################################################################


def _construct_HPRTargetInputs(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    *,
    is_heat_pumping: bool = True,
    zone_config: Configuration = Configuration(),
    debug: bool = False,
) -> HPRTargetInputs:
    """Build a validated optimisation-input bundle for heat pump targeting."""
    T_vals, H_hot, H_cold = T_vals.copy(), H_hot.copy(), H_cold.copy()
    T_hot, T_cold = _apply_temperature_shift_for_hpr_stream_dtmin_cont(
        T_vals, zone_config.DT_CONT_HP
    )

    for T_arr, H_arr, is_cold in [(T_hot, H_hot, False), (T_cold, H_cold, True)]:
        if (is_cold and is_heat_pumping) or (not (is_cold) and not (is_heat_pumping)):
            T_arr, H_arr = _get_reduced_bckgrd_cascade_till_Q_target(
                Q_hpr_target, T_arr, H_arr, is_cold=is_cold
            )
        T_arr, H_arr, z_amb_arr = _get_simplified_bckgrd_cascade_and_z_amb(
            T_vals=T_arr,
            H_vals=H_arr,
            zone_config=zone_config,
            is_cold=is_cold,
        )
        s = _create_stream_collection_of_background_profile(T_arr, H_arr)
        if is_cold:
            T_cold, H_cold, z_amb_cold, s_cold = T_arr, H_arr, z_amb_arr, s
        else:
            T_hot, H_hot, z_amb_hot, s_hot = T_arr, H_arr, z_amb_arr, s
    inputs = {
        "Q_hpr_target": Q_hpr_target,
        "Q_heat_max": H_cold[0],
        "Q_cool_max": -H_hot[-1],
        "T_hot": T_hot,
        "H_hot": H_hot,
        "z_amb_hot": z_amb_hot,
        "T_cold": T_cold,
        "H_cold": H_cold,
        "z_amb_cold": z_amb_cold,
        "dt_range_max": max(T_cold[0], T_hot[0]) - min(T_cold[-1], T_hot[-1]),
        "is_heat_pumping": bool(is_heat_pumping),
        "eta_penalty": 0.001,
        "rho_penalty": 10,
        "bckgrd_hot_streams": s_hot,
        "bckgrd_cold_streams": s_cold,
        "debug": debug,
        "hpr_type": zone_config.HPR_TYPE,
        "n_cond": zone_config.N_COND,
        "n_evap": zone_config.N_EVAP,
        "eta_comp": zone_config.ETA_COMP,
        "eta_exp": zone_config.ETA_EXP,
        "eta_ii_hpr_carnot": zone_config.ETA_II_HPR_CARNOT,
        "eta_ii_he_carnot": zone_config.ETA_II_HE_CARNOT
        if zone_config.ALLOW_INTEGRATED_EXPANDER
        else 0.0,
        "dtcont_hp": zone_config.DT_CONT_HP,
        "dt_hp_ihx": zone_config.DT_HPR_IHX,
        "dt_cascade_hx": zone_config.DT_HPR_CASCADE_HX,
        "T_env": zone_config.T_ENV,
        "dt_env_cont": zone_config.DT_ENV_CONT,
        "dt_phase_change": zone_config.DT_PHASE_CHANGE,
        "refrigerant_ls": [r.strip().upper() for r in zone_config.REFRIGERANTS],
        "do_refrigerant_sort": zone_config.DO_REFRIGERANT_SORT,
        "heat_to_power_ratio": zone_config.PRICE_RATIO_HEAT_TO_ELE,
        "cold_to_power_ratio": zone_config.PRICE_RATIO_COLD_TO_ELE,
        "max_multi_start": zone_config.MAX_HP_MULTISTART,
        "bb_minimiser": zone_config.BB_MINIMISER,
        "allow_integrated_expander": zone_config.ALLOW_INTEGRATED_EXPANDER,
        "initialise_simulated_cycle": zone_config.INITIALISE_SIMULATED_CYCLE,
    }
    return HPRTargetInputs(**inputs)


def _apply_temperature_shift_for_hpr_stream_dtmin_cont(
    T_vals: np.ndarray,
    dtmin_hp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply HP-specific temperature shifting to a cascade grid."""
    return T_vals - dtmin_hp, T_vals + dtmin_hp


def _get_reduced_bckgrd_cascade_till_Q_target(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *,
    is_cold: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim a cascade profile to a target duty using boundary interpolation."""
    if is_cold:
        if H_vals[0] < Q_hpr_target:
            return T_vals, H_vals

        i = H_vals.size - np.searchsorted(H_vals[::-1], Q_hpr_target, side="left") - 1
        if i == T_vals.size - 1:
            raise ValueError("Target for heat pumping cannot be zero.")
        T_vals[i] = linear_interpolation(
            Q_hpr_target, H_vals[i], H_vals[i + 1], T_vals[i], T_vals[i + 1]
        )
        H_vals[i] = Q_hpr_target
        return T_vals[i:], H_vals[i:]

    else:
        if -H_vals[-1] < Q_hpr_target:
            return T_vals, H_vals

        i = np.searchsorted(-H_vals, Q_hpr_target, side="left")
        if i == 0:
            raise ValueError("Target for refrigeration cannot be zero.")
        T_vals[i] = linear_interpolation(
            -Q_hpr_target, H_vals[i], H_vals[i - 1], T_vals[i], T_vals[i - 1]
        )
        H_vals[i] = -Q_hpr_target
        return T_vals[: i + 1], H_vals[: i + 1]


def _get_z_ambient(
    T_vals: np.ndarray,
    T_amb_star: float,
    is_cold: bool,
) -> Tuple[np.ndarray, float]:
    """Return the ambient-load indicator for each temperature interval."""
    if is_cold:
        return np.where(T_vals > T_amb_star, 1.0, 0.0)
    else:
        return np.where(T_vals < T_amb_star, -1.0, 0.0)


def _get_simplified_bckgrd_cascade_and_z_amb(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    zone_config: Configuration,
    *,
    is_cold: bool,
) -> Tuple[np.ndarray, float]:
    """Clean a background cascade load profile and return the ambient-load indicator."""
    sign = 1 if is_cold else -1
    T_amb_star = (
        zone_config.T_ENV + (zone_config.DT_ENV_CONT + zone_config.DT_CONT_HP) * sign
    )
    T_vals, H_vals = _add_T_amb_interval(
        T_vals, H_vals, T_amb_star, zone_config.DT_PHASE_CHANGE, is_cold
    )
    z_amb = _get_z_ambient(
        T_vals=T_vals,
        T_amb_star=T_amb_star,
        is_cold=is_cold,
    )
    H_vals += z_amb

    T_vals, H_vals = clean_composite_curve(T_vals, H_vals)

    z_amb = _get_z_ambient(
        T_vals=T_vals,
        T_amb_star=zone_config.T_ENV
        + (zone_config.DT_ENV_CONT + zone_config.DT_CONT_HP) * sign,
        is_cold=is_cold,
    )
    H_vals -= z_amb

    T_vals, H_vals, z_amb = _extend_profile_with_temperature_margin(
        T_vals, H_vals, z_amb, dt_margin=10.0
    )
    return T_vals, H_vals, z_amb


def _add_T_amb_interval(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    T_amb: float,
    dt_phase_change: float,
    is_cold: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add an ambient temperature interval to a background cascade profile."""
    H_label = PT.H_NET_COLD.value if is_cold else PT.H_NET_HOT.value
    pt = ProblemTable(
        {
            PT.T.value: T_vals,
            H_label: H_vals,
        }
    )
    T_amb_ls = (
        [T_amb, T_amb + dt_phase_change]
        if is_cold
        else [T_amb, T_amb - dt_phase_change]
    )
    pt.insert_temperature_interval(T_amb_ls)
    return pt.col[PT.T.value], pt.col[H_label]


def _extend_profile_with_temperature_margin(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    z_amb: np.ndarray,
    *,
    dt_margin: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add flat end intervals above and below a composite profile."""
    if T_vals.size == 0:
        return T_vals, H_vals

    T_ext = np.empty(T_vals.size + 2, dtype=T_vals.dtype)
    H_ext = np.empty(H_vals.size + 2, dtype=H_vals.dtype)
    z_ext = np.empty(z_amb.size + 2, dtype=z_amb.dtype)

    T_ext[0] = T_vals[0] + dt_margin
    T_ext[1:-1] = T_vals
    T_ext[-1] = T_vals[-1] - dt_margin

    H_ext[0] = H_vals[0]
    H_ext[1:-1] = H_vals
    H_ext[-1] = H_vals[-1]

    z_ext[0] = z_amb[0]
    z_ext[1:-1] = z_amb
    z_ext[-1] = z_amb[-1]

    return T_ext, H_ext, z_ext


#######################################################################################################
# Helper functions: Optimise multi-temperature Carnot heat pump placement
#######################################################################################################


@timing_decorator
def _optimise_multi_temperature_carnot_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    """Compute baseline condenser/evaporator temperature levels and duties for a single multi-temperature heat pump layout."""
    # Solve the placement problem
    res = _solve_hpr_placement(
        f_obj=_compute_multi_temperature_carnot_cycle_obj,
        x0_ls=_get_x0_for_multi_temperature_carnot_hp_opt(args),
        bnds=_get_bounds_for_multi_temperature_carnot_hp_opt(args),
        args=args,
    )

    # Post-process the results
    res.update(
        _get_carnot_hpr_cycle_streams(
            res["T_cond"], res["Q_cond"], res["T_evap"], res["Q_evap"], args
        )
    )
    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_multi_temperature_carnot_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build initial guess for the cascade condenser/evaporator optimization."""
    n_cond, n_evap = int(args.n_cond), int(args.n_evap)
    return (
        [0.0]            # Ambient source/sink bounds
        + [0.0] * n_cond # Condenser temperature bounds
        + [0.0] * n_evap # Evaporator temperature bounds
    )


def _get_bounds_for_multi_temperature_carnot_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build bounds for the cascade condenser/evaporator optimization."""
    n_cond, n_evap = int(args.n_cond), int(args.n_evap)
    return (
        [(-1.0, 10.0)]          # Ambient source/sink bounds
        + [(0.0, 1.0)] * n_cond # Condenser temperature bounds
        + [(0.0, 1.0)] * n_evap # Evaporator temperature bounds
    )


def _parse_multi_temperature_carnot_cycle_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    """Compile the full list of condenser and evaporator temperature levels and ambient source/sink."""
    x_amb = x[0]
    a = 1 + int(args.n_cond)
    x_cond = x[1: a]
    b = a + int(args.n_evap)
    x_evap = x[a: b]

    Q_amb_hot, Q_amb_cold = _map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    T_cond = _map_x_arr_to_T_arr(
        x_cond, args.T_cold[0], args.T_cold[-1]
    )
    T_evap = _map_x_arr_to_T_arr(
        x_evap, args.T_hot[-1], args.T_hot[0]
    )
    return {
        "T_cond": T_cond,
        "T_evap": T_evap,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold
    }


def _get_multi_temperature_carnot_stage_duties_and_work(
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    H_hot_with_amb: np.ndarray,
    H_cold_with_amb: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    """Estimate COP by scaling the Carnot limit using entropic mean temperatures."""
    Q_cond = _get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )
    Q_evap = _get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap, args.T_hot, H_hot_with_amb, is_cond=False
    )

    Qc_he = np.zeros_like(Q_cond)
    Qe_he = np.zeros_like(Q_evap)
    Qc_hx = np.zeros_like(Q_cond)
    Qe_hx = np.zeros_like(Q_evap)
    Qc_hpr = np.zeros_like(Q_cond)
    Qe_hpr = np.zeros_like(Q_evap)
    w_he = 0.0
    w_hpr = 0.0
    cop = 1.0

    def _get_idx_and_Q_available(
        is_on: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx_c, idx_e = np.nonzero(is_on)
        idx_c = np.unique(idx_c)
        idx_e = np.unique(idx_e)
        used_cond = Qc_he + Qc_hx + Qc_hpr
        used_evap = Qe_he + Qe_hx + Qe_hpr        
        Qc_pool = np.maximum(Q_cond[idx_c] - used_cond[idx_c], 0.0)
        Qe_pool = np.maximum(Q_evap[idx_e] - used_evap[idx_e], 0.0)
        return idx_c, idx_e, Qc_pool, Qe_pool

    T_diff = np.subtract.outer(T_cond, T_evap)

    # Positive lift: pooled heat-pump mode with the remaining duties.
    is_hp = T_diff >= tol
    if np.any(is_hp):
        i_c, i_e, Qc_pool, Qe_pool = _get_idx_and_Q_available(is_hp)
        if Qc_pool.sum() > tol and Qe_pool.sum() > tol:
            T_h = _compute_entropic_mean_temperature(T_cond[i_c], Qc_pool)
            T_l = _compute_entropic_mean_temperature(T_evap[i_e], Qe_pool)
            cop = _calc_carnot_heat_pump_cop(T_h, T_l, args.eta_ii_hpr_carnot)
            if cop > 1.0 + tol:
                Qe_used = min(
                    Qe_pool.sum(),
                    Qc_pool.sum() * (cop - 1.0) / cop,
                )
                w_hpr = Qe_used / (cop - 1.0)
                Qc_used = Qe_used + w_hpr
                Qc_hpr[i_c] = Qc_pool * (Qc_used / Qc_pool.sum())
                Qe_hpr[i_e] = Qe_pool * (Qe_used / Qe_pool.sum())
            else:
                cop = 1.0

    # Negative lift: pooled heat-engine mode.
    is_he = (T_diff <= -tol) & (args.eta_ii_he_carnot >= tol)
    if np.any(is_he):
        i_c, i_e, Qc_pool, Qe_pool = _get_idx_and_Q_available(is_he)
        if Qc_pool.sum() > tol and Qe_pool.sum() > tol:
            T_h_he = _compute_entropic_mean_temperature(T_evap[i_e], Qe_pool)
            T_l_he = _compute_entropic_mean_temperature(T_cond[i_c], Qc_pool)
            eta_he = _calc_carnot_heat_engine_eta(T_h_he, T_l_he, args.eta_ii_he_carnot)
            if eta_he > tol:
                Qe_used = min(Qe_pool.sum(), Qc_pool.sum() / max(1.0 - eta_he, tol))
                w_he = Qe_used * eta_he
                Qc_used = Qe_used - w_he
                Qc_he[i_c] = Qc_pool * (Qc_used / Qc_pool.sum())
                Qe_he[i_e] = Qe_pool * (Qe_used / Qe_pool.sum())

    # Near-zero lift: direct heat exchange with any source/sink left after HE.
    is_hx = ((T_diff > -tol) | (args.eta_ii_he_carnot < tol)) & (T_diff < tol)
    if np.any(is_hx):
        i_c, i_e, Qc_pool, Qe_pool = _get_idx_and_Q_available(is_hx)
        if Qc_pool.sum() > tol and Qe_pool.sum() > tol:
            Q_transfer = min(Qc_pool.sum(), Qe_pool.sum())
            Qc_hx[i_c] = Qc_pool * (Q_transfer / Qc_pool.sum())
            Qe_hx[i_e] = Qe_pool * (Q_transfer / Qe_pool.sum())

    Qc = Qc_he + Qc_hx + Qc_hpr
    Qe = Qe_he + Qe_hx + Qe_hpr
    heat_ex = Qc_hx.sum()

    if not np.isclose(Qc.sum() + w_he, Qe.sum() + w_hpr, atol=tol):
        raise ValueError(
            "Energy balance not satisfied in multi-temperature Carnot cycle calculation."
        )

    return {
        "w_hpr": w_hpr,
        "w_he": w_he,
        "heat_ex": heat_ex,
        "cop": cop,
        "Qc": Qc,
        "Qe": Qe
    }


def _compute_multi_temperature_carnot_cycle_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate compressor work for a candidate multi-temperature Carnot HP placement defined by vector `x`."""
    state_vars = _parse_multi_temperature_carnot_cycle_state_variables(x, args)

    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars["Q_amb_cold"]
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars["Q_amb_hot"]

    cycle_results = _get_multi_temperature_carnot_stage_duties_and_work(
        T_cond=state_vars["T_cond"],
        T_evap=state_vars["T_evap"],
        H_hot_with_amb=H_hot_with_amb,
        H_cold_with_amb=H_cold_with_amb,
        args=args,
    )

    work = cycle_results["w_hpr"] - cycle_results["w_he"]

    # Determine external heating and cooling demand
    Q_ext_heat = max(np.abs(H_cold_with_amb[0]) - cycle_results["Qc"].sum(), 0.0)
    Q_ext_cold = max(np.abs(H_hot_with_amb[-1]) - cycle_results["Qe"].sum(), 0.0)

    # Determine the objective
    p = (
        g_ineq_penalty(
            g=Q_ext_cold,
            rho=args.rho_penalty,
            form="square",
        )
        if not (args.is_heat_pumping)
        else 0.0
    )
    obj = _calc_obj(
        work=cycle_results["w_hpr"] - cycle_results["w_he"],
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:  # If in debug mode, plot the graph immediately
        res = _get_carnot_hpr_cycle_streams(state_vars["T_cond"], cycle_results["Qc"], state_vars["T_evap"], cycle_results["Qe"], args)
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            res["hpr_hot_streams"],
            res["hpr_cold_streams"],
            title=f"Obj {float(obj):.5f} = {(work / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )
        pass

    return {
        "obj": obj,
        "utility_tot": work + Q_ext_heat + Q_ext_cold,
        "w_net": work,
        "w_hpr": cycle_results["w_hpr"],
        "w_he": cycle_results["w_he"],
        "heat_ex": cycle_results["heat_ex"],
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "Q_cond": cycle_results["Qc"],
        "T_evap": state_vars["T_evap"],
        "Q_evap": cycle_results["Qe"],
        "cop_h": cycle_results["cop"],
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "success": True,
    }


#######################################################################################################
# Helper functions: Optimise cascade heat pump placement with CoolProp simulation
#######################################################################################################


@timing_decorator
def _optimise_cascade_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    """Optimise the integration of a cascade heat pump with a specified number of condensers and evaporators."""
    num_stages = int(args.n_cond + args.n_evap - 1)

    # Use as initialisation
    if args.initialise_simulated_cycle:
        init_res = _optimise_multi_temperature_carnot_heat_pump_placement(args)

    # Validate specific inputs related to this approach
    args.refrigerant_ls = _validate_vapour_hp_refrigerant_ls(num_stages, args)

    # Solve the placement problem
    res = _solve_hpr_placement(
        f_obj=_compute_cascade_hp_system_obj,
        x0_ls=_get_x0_for_cascade_hp_opt(init_res, args) if args.initialise_simulated_cycle else None,
        bnds=_get_bounds_for_cascade_hp_opt(args),
        args=args,
    )
    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_cascade_hp_opt(
    init_res: HPRTargetOutputs,
    args: HPRTargetInputs,
) -> np.ndarray:
    """Build initial guess vectors for the cascade condenser/evaporator optimization."""
    n_cool = max(int(args.n_evap) - 1, 0)
    Q_cool_ex = args.Q_cool_max + init_res.Q_amb_hot
    Q_heat_ex = args.Q_heat_max + init_res.Q_amb_cold

    x_amb = _map_Q_amb_to_x(init_res.Q_amb_hot, init_res.Q_amb_cold, max(Q_heat_ex, Q_cool_ex))
    x_cond = _map_T_arr_to_x_arr(init_res.T_cond, args.T_cold[0], args.T_cold[-1]).tolist()
    x_evap = _map_T_arr_to_x_arr(init_res.T_evap, args.T_hot[-1], args.T_hot[0]).tolist()
    x_subcool = [0.0] * int(args.n_cond)
    x_heat = _map_Q_arr_to_x_arr(init_res.Q_cond, Q_heat_ex).tolist()
    x_cool = _map_Q_arr_to_x_arr(init_res.Q_evap[: n_cool], Q_cool_ex).tolist()
    return np.asarray(
        [x_amb] + x_cond + x_evap + x_subcool + x_heat + x_cool,
        dtype=np.float64,
    )


def _get_bounds_for_cascade_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build bounds for the cascade condenser/evaporator optimization."""
    n_cond = n_heat = int(args.n_cond)
    n_evap = int(args.n_evap)
    n_cool = n_evap - 1
    return (
        [(-1.0, 10.0)]          # Ambient source/sink bounds
        + [(0.0, 1.0)] * n_cond # Condenser temperature bounds
        + [(0.0, 1.0)] * n_evap # Evaporator temperature bounds
        + [(0.0, 1.0)] * n_cond # Subcooling bounds
        + [(0.0, 1.0)] * n_heat # Heating duty bounds
        + [(0.0, 1.0)] * n_cool # Cooling duty bounds
    )


def _parse_cascade_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    """Extract HP variables from optimization vector x."""
    x_amb = x[0]
    a = 1 + int(args.n_cond)
    x_cond = x[1: a]
    b = a + int(args.n_evap)
    x_evap = x[a: b]
    c = b + int(args.n_cond)
    x_subcool = x[b: c]
    d = c + int(args.n_cond)
    x_heat = x[c: d]
    e = d + int(args.n_evap) - 1
    x_cool = x[d: e]

    Q_amb_hot, Q_amb_cold = _map_x_to_Q_amb(x_amb, max(args.Q_heat_max, args.Q_cool_max))
    T_cond = _map_x_arr_to_T_arr(x_cond, args.T_cold[0], args.T_cold[-1])
    T_evap = _map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = _map_x_arr_to_DT_arr(x_subcool, T_cond, args.T_cold[0])
    Q_heat = _map_x_arr_to_Q_arr(x_heat, args.Q_heat_max + Q_amb_hot)
    Q_cool = _append_unspecified_final_cascade_cooling_duty(
        x_cool * (args.Q_cool_max + Q_amb_cold)
    )
    return {
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": Q_heat,
        "T_evap": T_evap,
        "Q_cool": Q_cool,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold
    }


def _compute_cascade_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Objective: minimise total compressor work for multi-HP configuration."""
    state_vars = _parse_cascade_hp_state_variables(x, args)

    # Simulate the cascade heat pump cycle with CoolProp
    hp = CascadeVapourCompressionCycle()
    hp.solve(
        T_evap=state_vars["T_evap"],
        T_cond=state_vars["T_cond"],
        Q_heat=state_vars["Q_heat"],
        Q_cool=state_vars["Q_cool"],
        dT_subcool=state_vars["dT_subcool"],
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dt_ihx_gas_side=args.dt_hp_ihx,
        dt_cascade_hx=args.dt_cascade_hx,
    )
    if not (hp.solved):
        return {"obj": np.inf, "success": False}

    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars["Q_amb_hot"]  
    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars["Q_amb_cold"]

    # Build streams based on heat pump profiles
    hpr_hot_streams = hp.build_stream_collection(
        include_cond=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )
    hpr_cold_streams = hp.build_stream_collection(
        include_evap=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )

    # Determine the heat cascade
    pt_cond = get_process_heat_cascade(
        hot_streams=hpr_hot_streams,
        cold_streams=args.bckgrd_cold_streams
        + _get_ambient_air_stream(Q_amb_cold=state_vars["Q_amb_cold"], args=args),
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams
        + _get_ambient_air_stream(Q_amb_hot=state_vars["Q_amb_hot"], args=args),
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    # Calculate key perfromance indicators
    w_hpr = hp.work
    cop = hp.Q_heat_arr.sum() / w_hpr if w_hpr > 0 else 1.0

    Q_ext_heat = pt_cond.col[PT.H_NET.value][0]
    Q_ext_cold = pt_evap.col[PT.H_NET.value][-1]
    g = [pt_cond.col[PT.H_NET.value][-1], pt_evap.col[PT.H_NET.value][0], hp.penalty]
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    # if not args.is_heat_pumping:
    #     p += g_ineq_penalty(g=Q_ext_cold, rho=args.rho_penalty, form="square")

    obj = _calc_obj(
        work=w_hpr,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:  # for debugging purposes, a quick plot function
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            hpr_hot_streams,
            hpr_cold_streams,
            title=f"Obj {float(obj):.5f} = {(w_hpr / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )
        pass

    return {
        "obj": obj,
        "utility_tot": w_hpr + Q_ext_heat + Q_ext_cold,
        "w_net": w_hpr,
        "w_hpr": hp.work_arr,
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "dT_subcool": state_vars["dT_subcool"],
        "Q_heat": hp.Q_heat_arr,
        "T_evap": state_vars["T_evap"],
        "Q_cool": hp.Q_cool_arr,
        "cop_h": cop,
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "hpr_hot_streams": hpr_hot_streams,
        "hpr_cold_streams": hpr_cold_streams,
        "model": hp,
    }


#######################################################################################################
# Helper functions: Optimise multiple simple Carnot heat pump placement
#######################################################################################################


@timing_decorator
def _optimise_multi_simple_carnot_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    """Compute baseline condenser/evaporator temperature levels and duties for a multiple simple heat pumps layout."""
    args.n_cond = args.n_evap = max(args.n_cond, args.n_evap)
    res = _solve_hpr_placement(
        f_obj=_compute_multi_simple_carnot_hp_opt_obj,
        x0_ls=[0.0 for _ in range(args.n_cond + args.n_evap + 1)],
        bnds=[(0.0, 1.0) for _ in range(args.n_cond + args.n_evap)] + [(-1.0, 4.0)],
        args=args,
    )
    res.update(
        _get_carnot_hpr_cycle_streams(
            res["T_cond"], res["Q_cond"], res["T_evap"], res["Q_evap"], args
        )
    )
    return HPRTargetOutputs.model_validate(res)


def _parse_multi_simple_carnot_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile the full list of condenser and evaporator temperature levels and ambient source/sink."""
    T_cond = _map_x_arr_to_T_arr(x[: args.n_cond], args.T_cold[0], args.T_cold[-1])
    T_evap = args.T_hot[-1] - np.array(x[args.n_cond : -1]) * (
        args.T_hot[-1] - args.T_hot[0]
    )
    scale = max(args.Q_heat_max, args.Q_cool_max)
    Q_amb_hot = min(scale * x[-1], 0.0) * -1
    Q_amb_cold = max(scale * x[-1], 0.0)
    return {
        "T_cond": T_cond,
        "T_evap": T_evap,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
    }


def _get_multi_simple_carnot_stage_duties_and_work(
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    H_hot_with_amb: np.ndarray,
    H_cold_with_amb: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, list]:
    """Scale independent stage duties while sharing the hot-side source profile."""
    Q_cond = _get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )

    Qc_he = np.zeros_like(Q_cond)
    Qe_he = np.zeros_like(Q_cond)
    Qc_hx = np.zeros_like(Q_cond)
    Qe_hx = np.zeros_like(Q_cond)
    Qc_hpr = np.zeros_like(Q_cond)
    Qe_hpr = np.zeros_like(Q_cond)
    w_hpr = np.zeros_like(Q_cond)
    w_he = np.zeros_like(Q_cond)
    q_diff = [0.0]

    T_diff = T_cond - T_evap
    T_cond_abs = T_cond + 273.15
    T_evap_abs = T_evap + 273.15

    is_he = (T_diff <= -tol) & (args.eta_ii_he_carnot >= tol)
    if np.any(is_he):
        eff_he = _calc_carnot_heat_engine_eta(
            T_evap_abs[is_he], T_cond_abs[is_he], args.eta_ii_he_carnot
        )
        Qc_he[is_he] = Q_cond[is_he]
        w_he[is_he] = Qc_he[is_he] * eff_he / (1 - eff_he)
        Qe_he[is_he] = Qc_he[is_he] + w_he[is_he]

    is_hx = ((T_diff > -tol) | (args.eta_ii_he_carnot < tol)) & (T_diff < tol)
    if np.any(is_hx):
        Qc_hx[is_hx] = Q_cond[is_hx]
        Qe_hx[is_hx] = Qc_hx[is_hx]

    is_hp = T_diff >= tol
    if np.any(is_hp):
        cop_hp = _calc_carnot_heat_pump_cop(
            T_cond_abs[is_hp], T_evap_abs[is_hp], args.eta_ii_hpr_carnot
        )
        Qc_hpr[is_hp] = Q_cond[is_hp]
        w_hpr[is_hp] = Qc_hpr[is_hp] / cop_hp
        Qe_hpr[is_hp] = Qc_hpr[is_hp] - w_hpr[is_hp]

    sort_idx = np.argsort(-T_evap, kind="stable")
    Q_allocated = 0.0
    for i in sort_idx:
        Q_stage = Qe_he[i] + Qe_hx[i] + Qe_hpr[i]
        if Q_stage < tol:
            Qc_he[i] = 0.0
            Qe_he[i] = 0.0
            Qc_hx[i] = 0.0
            Qe_hx[i] = 0.0
            Qc_hpr[i] = 0.0
            Qe_hpr[i] = 0.0
            w_he[i] = 0.0
            w_hpr[i] = 0.0
            continue

        Q_available = (
            _get_Q_vals_at_T_hpr_from_bckgrd_profile(
                np.array([T_evap[i]]), args.T_hot, H_hot_with_amb, is_cond=False
            )[0]
            - Q_allocated
        )
        if Q_available < tol:
            scale = 0.0
        elif Q_stage > Q_available:
            scale = Q_available / Q_stage
        else:
            scale = 1.0

        Qc_he[i] *= scale
        Qe_he[i] *= scale
        Qc_hx[i] *= scale
        Qe_hx[i] *= scale
        Qc_hpr[i] *= scale
        Qe_hpr[i] *= scale
        w_he[i] *= scale
        w_hpr[i] *= scale
        Q_allocated += Qe_he[i] + Qe_hx[i] + Qe_hpr[i]

    Qc = Qc_he + Qc_hx + Qc_hpr
    Qe = Qe_he + Qe_hx + Qe_hpr
    heat_ex = Qc_hx.sum()

    if not (np.isclose(Qc.sum() + w_he.sum(), Qe.sum() + w_hpr.sum(), atol=tol)):
        raise ValueError(
            "Energy balance not satisfied in multiple simple Carnot cycle calculation."
        )

    return {
        "w_hpr": w_hpr,
        "w_he": w_he,
        "heat_ex": heat_ex,
        "Qc": Qc,
        "Qe": Qe,
        "q_diff": q_diff
    }


def _compute_multi_simple_carnot_hp_opt_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate compressor work for a candidate multiple single Carnot HP placement defined by vector `x`."""
    state_vars = (
        _parse_multi_simple_carnot_hp_state_variables(x, args)
    )
    # Adjust the background profiles to account for ambient heat exchange
    H_cold_with_amb = args.H_cold + args.z_amb_cold * state_vars["Q_amb_cold"]
    H_hot_with_amb = args.H_hot + args.z_amb_hot * state_vars["Q_amb_hot"]

    # Calculate the target heat for each condenser/evaporator temperature pair from the background profiles
    cycle_results = _get_multi_simple_carnot_stage_duties_and_work(
        T_cond=state_vars["T_cond"],
        T_evap=state_vars["T_evap"],
        H_hot_with_amb=H_hot_with_amb,
        H_cold_with_amb=H_cold_with_amb,
        args=args,
    )

    # Calculate evaporator duty
    work = cycle_results["w_hpr"].sum() - cycle_results["w_he"].sum()
    cop = cycle_results["Qc"].sum() / cycle_results["w_hpr"].sum() if cycle_results["w_hpr"].sum() > tol else 0
    eta_he = cycle_results["w_he"].sum() / (cycle_results["Qc"].sum() + 1e-6) if cycle_results["Qc"].sum() != 0 else 0

    # Determine external heating and cooling demand
    Q_ext_heat = max(np.abs(H_cold_with_amb[0]) - cycle_results["Qc"].sum(), 0.0)
    Q_ext_cold = max(np.abs(H_hot_with_amb[-1]) - cycle_results["Qe"].sum(), 0.0)

    # Determine the key performance metrics of the heat pump
    p = g_ineq_penalty(g=cycle_results["q_diff"], rho=args.rho_penalty, form="square")

    # Calculate the objective function value
    obj = _calc_obj(
        work=work,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:  # If in debug mode, plot the graph immediately
        res = _get_carnot_hpr_cycle_streams(state_vars["T_cond"], cycle_results["Qc"], state_vars["T_evap"], cycle_results["Qe"], args)
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            res["hpr_hot_streams"],
            res["hpr_cold_streams"],
            title=f"Obj {float(obj):.5f} = {(work.sum() / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )
        pass

    return {
        "obj": obj,
        "utility_tot": work.sum() + Q_ext_heat,
        "w_net": work,
        "w_hpr": cycle_results["w_hpr"],
        "w_he": cycle_results["w_he"],
        "heat_ex": cycle_results["heat_ex"],
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "Q_cond": cycle_results["Qc"],
        "T_evap": state_vars["T_evap"],
        "Q_evap": cycle_results["Qe"],
        "cop_h": cop,
        "eta_he": eta_he,
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "success": True,
    }


#######################################################################################################
# Helper functions: Optimise multi simple heat pumps placement with CoolProp simulation
#######################################################################################################


@timing_decorator
def _optimise_multi_simple_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    """Optimise the integration of multiple single heat pump units with a specified number of units."""
    num_stages = args.n_cond = args.n_evap = int(max(args.n_cond, args.n_evap))

    # Use as initialisation
    if args.initialise_simulated_cycle:
        init_res = _optimise_multi_simple_carnot_heat_pump_placement(args)

    # Validate specific inputs related to this approach
    args.refrigerant_ls = _validate_vapour_hp_refrigerant_ls(num_stages, args)

    # Prepare and run the optimisation
    bnds = _get_bounds_for_multi_single_hp_opt(args)
    x0 = (
        _get_x0_for_multi_single_hp_opt(
            T_cond=init_res.T_cond,
            Q_cond=init_res.Q_cond,
            T_evap=init_res.T_evap,
            args=args,
            bnds=bnds,
        )
        if args.initialise_simulated_cycle
        else None
    )
    res = _solve_hpr_placement(
        f_obj=_compute_multi_simple_hp_system_obj,
        x0_ls=x0,
        bnds=bnds,
        args=args,
    )
    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_multi_single_hp_opt(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    args: HPRTargetInputs,
    bnds: list,
) -> np.ndarray:
    """Build initial guesses for the multi single-HP condenser/evaporator optimization."""
    x_sc_bnds = [
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    ]
    x_sh_bnds = [
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_hot[0] - args.T_hot[-1]) / args.dt_range_max),
    ]

    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds[1] = x_sc_bnds[0]

    if x_sh_bnds[0] > x_sh_bnds[1]:
        x_sh_bnds[1] = x_sh_bnds[0]

    x_cond = (args.T_cold[0] - T_cond) / args.dt_range_max
    x_sc = np.array(
        [args.dt_phase_change / args.dt_range_max for _ in range(len(T_cond))]
    )
    y_cond = Q_cond / args.Q_hpr_target

    x_evap = (T_evap - args.T_hot[-1]) / args.dt_range_max
    x_sh = np.array(
        [args.dt_phase_change / args.dt_range_max for _ in range(len(T_evap))]
    )

    x_ls = []
    for candidate in [x_cond, x_sc, y_cond, x_evap, x_sh]:
        x_ls.extend(candidate[:])

    x0 = np.array(x_ls)

    if x0.size != len(bnds):
        raise ValueError(
            "Bounds size must match x0 size for multiple single HP optimisation."
        )

    for i in range(x0.size):
        if not (bnds[i][0] <= x0[i] <= bnds[i][1]):
            x0[i] = bnds[i][1] if i * 2 < x0.size else bnds[i][0]

    return np.asarray([x0])


def _get_bounds_for_multi_single_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build bounds for the multi single-HP condenser/evaporator optimization."""
    x_cond_bnds = []
    x_evap_bnds = []
    for i, refrigerant in enumerate(args.refrigerant_ls):
        T_min, T_max = (
            PropsSI("Tmin", refrigerant) - 273.15,
            PropsSI("Tmax", refrigerant) - 273.15,
        )
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


def _parse_multi_simple_hp_state_temperatures(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x."""
    n = args.n_cond
    T_cond = np.array(args.T_cold[0] - x[:n] * args.dt_range_max, dtype=np.float64)
    dT_subcool = np.array(x[n : 2 * n] * args.dt_range_max, dtype=np.float64)
    Q_cond = np.array(x[2 * n : 3 * n] * args.Q_heat_max, dtype=np.float64)
    T_evap = np.array(
        x[3 * n : 4 * n] * args.dt_range_max + args.T_hot[-1], dtype=np.float64
    )
    dT_superheat = np.array(x[4 * n : -1] * args.dt_range_max, dtype=np.float64)
    scale = max(args.Q_heat_max, args.Q_cool_max)
    Q_amb_hot = min(scale * x[-1], 0.0) * -1
    Q_amb_cold = max(scale * x[-1], 0.0)
    return {
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": Q_cond,
        "T_evap": T_evap,
        "dT_superheat": dT_superheat,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold
    }


def _compute_multi_simple_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    debug: bool = False,
) -> dict:
    """Objective: minimize total compressor work for multi-HP configuration."""
    state_vars = (
        _parse_multi_simple_hp_state_temperatures(x, args)
    )

    T_diff = state_vars["T_cond"] - state_vars["dT_subcool"] - state_vars["T_evap"] - args.dtcont_hp
    if T_diff.min() < 0:
        return {"obj": np.inf, "success": False}

    H_hot_with_amb = args.H_hot + getattr(args, "z_amb_hot", 0.0) * state_vars["Q_amb_hot"]
    H_cold_with_amb = args.H_cold + getattr(args, "z_amb_cold", 0.0) * state_vars["Q_amb_cold"]

    hp = ParallelVapourCompressionCycles()
    hp.solve(
        T_evap=state_vars["T_evap"],
        T_cond=state_vars["T_cond"],
        dT_superheat=state_vars["dT_superheat"],
        dT_subcool=state_vars["dT_subcool"],
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dt_ihx_gas_side=args.dt_hp_ihx,
        Q_heat=state_vars["Q_heat"],
    )

    # Build streams based on heat pump profiles
    hpr_hot_streams = hp.build_stream_collection(
        include_cond=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )
    hpr_cold_streams = hp.build_stream_collection(
        include_evap=True,
        is_process_stream=False,
        dtcont=args.dtcont_hp,
    )

    # Determine the heat cascade
    pt_cond = get_process_heat_cascade(
        hot_streams=hpr_hot_streams,
        cold_streams=args.bckgrd_cold_streams
        + _get_ambient_air_stream(Q_amb_cold=state_vars["Q_amb_cold"], args=args),
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams
        + _get_ambient_air_stream(Q_amb_hot=state_vars["Q_amb_hot"], args=args),
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    # Calculate key perfromance indicators
    w_hpr = hp.work
    cop = hp.Q_heat_arr.sum() / w_hpr if w_hpr > 0 else 1.0

    Q_ext_heat = max(float(pt_cond.col[PT.H_NET.value][0]), 0.0)
    Q_ext_cold = max(float(pt_evap.col[PT.H_NET.value][0]), 0.0)
    g = np.maximum(
        np.array([pt_cond.col[PT.H_NET.value][-1], hp.penalty], dtype=float),
        0.0,
    )
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    if not args.is_heat_pumping:
        p += g_ineq_penalty(g=Q_ext_cold, rho=args.rho_penalty, form="square")

    obj = _calc_obj(
        work=w_hpr,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    # For debugging purposes, a quick plot function
    if debug:
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            hpr_hot_streams,
            hpr_cold_streams,
            title=f"Obj {float(obj):.5f} = {(w_hpr / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )
        pass

    return {
        "obj": obj,
        "utility_tot": w_hpr + Q_ext_heat + Q_ext_cold,
        "w_net": w_hpr,
        "w_hpr": hp.work_arr,
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": state_vars["T_cond"],
        "dT_subcool": state_vars["dT_subcool"],
        "Q_heat": hp.Q_heat_arr,
        "T_evap": state_vars["T_evap"],
        "dT_superheat": state_vars["dT_superheat"],
        "Q_cool": hp.Q_cool_arr,
        "cop_h": cop,
        "Q_amb_hot": state_vars["Q_amb_hot"],
        "Q_amb_cold": state_vars["Q_amb_cold"],
        "hpr_hot_streams": hpr_hot_streams,
        "hpr_cold_streams": hpr_cold_streams,
        "model": hp,
    }


#######################################################################################################
# Helper functions: Optimise single Brayton heat pump placement - TODO
#######################################################################################################


@timing_decorator
def _optimise_brayton_heat_pump_placement(
    args: HPRTargetInputs,
) -> HPRTargetOutputs:
    """Optimise a single Brayton heat pump against a given composite curve."""
    args.n_cond = args.n_evap = 1  # Must be one gas cooler and one gas heater
    args.refrigerant_ls = ["air"]

    x0 = _get_x0_for_brayton_hp_opt(args)
    bnds = _get_bounds_for_brayton_hp_opt(args)
    opt = minimize(
        fun=lambda x: _compute_brayton_hp_system_obj(x, args)["obj"],
        x0=x0,
        method="SLSQP",
        bounds=bnds,
        options={"disp": False, "maxiter": 1000},
        tol=1e-7,
    )

    res = None
    if opt.success:
        res = _compute_brayton_hp_system_obj(opt.x, args)
        res["success"] = opt.success
        if 0:
            plot_multi_hp_profiles_from_results(
                args.T_hot,
                args.H_hot,
                args.T_cold,
                args.H_cold,
                res["hpr_hot_streams"],
                res["hpr_cold_streams"],
            )
    else:
        raise ValueError(f"Brayton heat pump targeting failed: {opt.message}")

    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_brayton_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build initial guesses for Brayton HP optimisation."""
    x0 = [
        0.0,
        abs(args.T_cold[0] - args.T_hot[0]) / args.dt_range_max,
        abs(args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max,
        1.0,
    ]
    return x0


def _get_bounds_for_brayton_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build bounds for Brayton HP optimisation."""
    return [
        (-0.2, 1.0),  # T_comp_out = T_cold_max + x[0] * dT_range_max
        (0.01, 1.5),  # dT_comp = x[1] * dT_range_max
        (0.01, 1.5),  # dT_gc = x[2] * dT_range_max
        (0.01, 1.0),  # Q_heat = x[3] * Q_hpr_target
    ]


def _parse_brayton_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x."""
    T_comp_out = np.array(args.T_cold[0] + x[0] * args.dt_range_max, dtype=np.float64)
    dT_comp = np.array(x[1] * args.dt_range_max, dtype=np.float64)
    dT_gc = np.array(x[2] * args.dt_range_max, dtype=np.float64)
    Q_heat = np.array(x[3] * args.Q_hpr_target, dtype=np.float64)
    return [T_comp_out], [dT_comp], [dT_gc], [Q_heat]


def _create_brayton_hp_list(
    T_comp_out: np.ndarray,
    dT_gc: np.ndarray,
    Q_gc: np.ndarray,
    dT_comp: np.ndarray,
    args: HPRTargetInputs,
) -> List[SimpleBraytonHeatPumpCycle]:
    """Instantiate SimpleBrytonHeatPumpCycle objects."""
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


def _compute_brayton_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> float:
    """Objective: minimize total compressor work for multi-HP configuration."""
    T_comp_out, dT_comp, dT_gc, Q_heat = _parse_brayton_hp_state_variables(x, args)

    hp_list = _create_brayton_hp_list(
        T_comp_out=T_comp_out,
        dT_comp=dT_comp,
        dT_gc=dT_gc,
        Q_gc=Q_heat,
        args=args,
    )

    hpr_hot_streams = _build_simulated_hpr_streams(hp_list, include_cond=True)
    hpr_cold_streams = _build_simulated_hpr_streams(hp_list, include_evap=True)

    T_exp_out = hp_list[0].cycle_states[3]["T"]

    pt_gas_cooler = get_process_heat_cascade(
        hot_streams=hpr_hot_streams,
        cold_streams=args.bckgrd_cold_streams,
        is_shifted=True,
    )
    pt_gas_heater = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams,
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    w_hpr = sum([hp.work_net for hp in hp_list])
    c = (
        pt_gas_cooler.col[PT.H_NET.value][-1] + pt_gas_heater.col[PT.H_NET.value][0]
    ) * 10  # Penalty for supplying too much heat
    Q_ext = pt_gas_cooler.col[PT.H_NET.value][
        0
    ]  # Extra heating required on either side of the gcc
    Q_cool = np.array([hp.Q_cool for hp in hp_list])
    COP = (args.Q_hpr_target - Q_ext) / (w_hpr + 1e-9)
    Q_amb = _calc_Q_amb(Q_cool.sum(), np.abs(args.H_hot[-1]), args.Q_amb_max)
    obj = _calc_obj(
        work=w_hpr,
        Q_ext_heat=Q_ext,
        Q_ext_cold=0.0,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=c,
    )

    if 0:
        plot_multi_hp_profiles_from_results(
            pt_gas_cooler.col[PT.T.value], pt_gas_cooler.col[PT.H_NET.value]
        )
        plot_multi_hp_profiles_from_results(
            pt_gas_heater.col[PT.T.value], pt_gas_heater.col[PT.H_NET.value]
        )
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            args.H_hot,
            args.T_cold,
            args.H_cold,
            hpr_hot_streams,
            hpr_cold_streams,
            title=f"T_hi {T_comp_out} -> {float(obj), float(c / args.Q_hpr_target)}",
        )

    return {
        "obj": obj,
        "utility_tot": w_hpr + Q_ext,
        "w_net": w_hpr,
        "Q_ext": Q_ext,
        "T_comp_out": np.array(T_comp_out),
        "dT_gc": np.array(dT_gc),
        "Q_heat": np.array(Q_heat),
        "T_evap": np.array(T_exp_out),
        "dT_comp": np.array(dT_comp),
        "Q_cool": np.array(Q_cool),
        "cop_h": COP,
        "Q_amb_hot": Q_amb if args.is_heat_pumping else 0.0,
        "Q_amb_cold": 0.0 if args.is_heat_pumping else Q_amb,
        "hpr_hot_streams": hpr_hot_streams,
        "hpr_cold_streams": hpr_cold_streams,
        "model": hp_list,
    }


#######################################################################################################
# Helper functions: other / non-specific
#######################################################################################################


def _solve_hpr_placement(
    f_obj: Callable,
    x0_ls: list | float,
    bnds: list,
    args: HPRTargetInputs,
) -> dict:
    """Solve a heat pump/refrigeration placement problem and return the best result."""
    if isinstance(x0_ls, (int, float)):
        x0_ls = [x0_ls]
    x0_ls = np.asarray(x0_ls, dtype=np.float64)
    if len(x0_ls.shape) == 1:
        x0_ls = [x0_ls]

    local_minima_x = multiminima(
        func=f_obj,
        func_kwargs=args,
        x0_ls=x0_ls,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,
        opt_kwargs={"n_runs": 10},
    )
    if local_minima_x.size == 0:
        raise ValueError(
            f"Heat pump and refrigeration targeting ({args.system_type}) failed to return any local minima."
        )

    res = f_obj(local_minima_x[0], args, debug=args.debug)
    if not res.get("success", True):
        raise ValueError(
            f"Heat pump and refrigeration targeting ({args.system_type}) failed to return an optimal result."
        )
    res["success"] = True
    res["amb_streams"] = _get_ambient_air_stream(
        res["Q_amb_hot"], res["Q_amb_cold"], args
    )
    return res


def _create_stream_collection_of_background_profile(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
) -> StreamCollection:
    """Constructs net stream segments that require utility input across temperature intervals."""
    s = StreamCollection()

    T_vals = np.asarray(T_vals)
    H_vals = np.abs(np.asarray(H_vals))

    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    dh_vals = delta_vals(H_vals)
    dh_indices = np.argwhere(np.abs(dh_vals) > tol).flatten()
    for i in dh_indices:
        if dh_vals[i] > tol:
            s.add(
                Stream(
                    t_supply=T_vals[i + 1],
                    t_target=T_vals[i],
                    heat_flow=dh_vals[i],
                )
            )
        elif -dh_vals[i] > tol:
            s.add(
                Stream(
                    t_supply=T_vals[i],
                    t_target=T_vals[i + 1],
                    heat_flow=-dh_vals[i],
                )
            )
    return s


def _get_Q_vals_at_T_hpr_from_bckgrd_profile(
    T_hpr: np.ndarray,
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    *,
    is_cond: bool = True,
) -> np.ndarray:
    """Interpolate the cascade at a specified temperature to find the corresponding duty for each temperature level."""
    H_less_origin = np.interp(T_hpr, T_vals[::-1], H_vals[::-1])
    H = (
        np.concatenate((H_less_origin, np.array([0.0])))
        if is_cond
        else np.concatenate((np.array([0.0]), H_less_origin))
    )
    temp = np.roll(H, -1)
    temp[-1] = 0
    Q = H - temp
    Q_hx = Q[:-1]
    return np.where(Q_hx > 0.0, Q_hx, 0.0)


def _compute_entropic_mean_temperature(
    T_arr: np.ndarray | list,
    Q_arr: np.ndarray | list,
    *,
    input_T_units: str = "C",
) -> float:
    """Compute the entropic average temperature."""
    T_arr = np.asarray(T_arr, dtype=float)
    Q_arr = np.asarray(Q_arr, dtype=float)
    unit_offset = 273.15 if input_T_units == "C" else 0
    if T_arr.var() < tol:
        return T_arr[0] + unit_offset
    S_tot = (Q_arr / (T_arr + unit_offset)).sum()
    return Q_arr.sum() / S_tot if S_tot > 0 else (T_arr.mean() + unit_offset)


def _calc_carnot_heat_pump_cop(
    T_h: float | np.ndarray,
    T_l: float | np.ndarray,
    eta_ii: float,
) -> float | np.ndarray:
    """Return Carnot COP scaled by second-law efficiency for absolute temperatures."""
    T_h_arr = np.asarray(T_h, dtype=float)
    T_l_arr = np.asarray(T_l, dtype=float)
    delta_T = T_h_arr - T_l_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        cop = np.where(
            delta_T > 0.0,
            T_l_arr / delta_T * eta_ii + 1.0,
            np.inf,
        )
    return cop.item() if cop.ndim == 0 else cop


def _calc_carnot_heat_engine_eta(
    T_h: float | np.ndarray,
    T_l: float | np.ndarray,
    eta_ii: float,
) -> float | np.ndarray:
    """Return Carnot heat engine efficiency scaled by second-law efficiency for absolute temperatures."""
    T_h_arr = np.asarray(T_h, dtype=float)
    T_l_arr = np.asarray(T_l, dtype=float)
    delta_T = T_h_arr - T_l_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(
            delta_T > 0.0,
            delta_T / T_h_arr * eta_ii,
            np.inf,
        )
    return eta.item() if eta.ndim == 0 else eta


def _map_x_arr_to_T_arr(
    x: np.ndarray,
    T_0: float,
    T_1: float,
) -> np.ndarray:
    """Recover condenser/evaporator temperatures from normalized x values."""
    temp = []
    for i in range(x.size):
        temp.append(T_0 - x[i] * (T_0 - T_1))
        T_0 = temp[-1]
    return np.sort(np.array(temp).flatten())[::-1]


def _map_T_arr_to_x_arr(
    T_arr: np.ndarray,
    T_0: float,
    T_1: float,
) -> np.ndarray:
    """Recover normalized x values from condenser/evaporator temperatures."""
    temp = []
    for i in range(T_arr.size):
        temp.append((T_0 - T_arr[i]) / (T_0 - T_1) if T_0 != T_1 else 0.0)
        T_0 = T_arr[i]
    return np.array(temp)


def _map_x_arr_to_DT_arr(
    x: np.ndarray,
    T_arr: np.ndarray,
    T_last: float,
) -> np.ndarray:
    """Recover delta temperatures from normalized x values."""
    return x * np.abs(T_arr - T_last)


def _map_DT_arr_to_x_arr(
    DT_arr: np.ndarray,
    T_arr: np.ndarray,
    T_last: float,
) -> np.ndarray:
    """Recover normalized x values from delta temperatures."""
    return np.where(
        T_arr != T_last,
        DT_arr / np.abs(T_arr - T_last),
        0.0
    )


def _map_x_arr_to_Q_arr(
    x: np.ndarray,
    Q_max: float,
) -> np.ndarray:
    """Recover heat duties from normalized x values."""
    return x * Q_max


def _map_Q_arr_to_x_arr(
    Q_arr: np.ndarray,
    Q_max: float,
) -> np.ndarray:
    """Recover normalized x values from heat duties."""
    return np.where(Q_max != 0, Q_arr / Q_max, 0.0)


def _map_x_to_Q_amb(
    x: float, 
    scale: float,
) -> Tuple[float, float]:
    Q_amb_hot = max(-scale * x, 0.0)
    Q_amb_cold = max(scale * x, 0.0)
    return Q_amb_hot, Q_amb_cold


def _map_Q_amb_to_x(
    Q_amb_hot: float,
    Q_amb_cold: float,
    scale: float,
) -> float:
    return (Q_amb_cold - Q_amb_hot) / scale


def _append_unspecified_final_cascade_cooling_duty(Q_cool: np.ndarray) -> np.ndarray:
    """Append the implicit final cooling duty expected by the cascade solver."""
    if Q_cool.size == 0:
        return np.array(np.nan)
    return np.concatenate([Q_cool, np.array([np.nan])])


def _get_hpr_cascade(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a problem table-based cascade from HP condenser/evaporator streams."""
    pt = create_problem_table_with_t_int(
        streams=hot_streams + cold_streams,
        is_shifted=False,
    )
    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value],
            hot_streams,
            cold_streams,
            is_shifted=False,
        )
    )
    return pt.col[PT.T.value], pt.col[PT.H_HOT_UT.value], pt.col[PT.H_COLD_UT.value]


def _validate_vapour_hp_refrigerant_ls(
    num_stages: int,
    args: HPRTargetInputs,
) -> list:
    """Validate and normalize refrigerant list length/order for staged cycles.

    Parameters
    ----------
    num_stages : int
        Required number of refrigerant entries.
    args : HPRTargetInputs
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
                ref
                for ref, _ in sorted(
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


def _get_carnot_hpr_cycle_cascade_profile(
    T_hpr: list,
    Q_hpr: list,
    dT_phase_change: float,
    is_hot: bool,
    i: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp near-equal HP levels and merge their heat duties."""
    inc = 1 if is_hot else -1
    if i == None:
        i = 0 if is_hot else len(T_hpr)

    i_range = range(i, len(T_hpr) - 1) if is_hot else reversed(range(1, i))
    for i in i_range:
        if abs(T_hpr[i] - T_hpr[i + inc]) < dT_phase_change:
            T_hpr.pop(i + inc)
            Q_hpr[i] += Q_hpr[i + inc]
            Q_hpr.pop(i + inc)
            T_hpr, Q_hpr = _get_carnot_hpr_cycle_cascade_profile(
                T_hpr,
                Q_hpr,
                dT_phase_change,
                is_hot,
                i,
            )
            break

    return T_hpr, Q_hpr


def _get_carnot_hpr_cycle_streams(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    args: HPRTargetInputs,
) -> dict:
    """Build condenser and evaporator stream collections from Carnot outputs."""
    return {
        "hpr_hot_streams": _build_latent_streams(
            T_cond, args.dt_phase_change, Q_cond, is_hot=True
        ),
        "hpr_cold_streams": _build_latent_streams(
            T_evap, args.dt_phase_change, Q_evap, is_hot=False
        ),
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
    """Convert a series of temperature levels into a StreamCollection, e.g. for condenser/evaporator levels or ambient air."""
    if len(T_ls) > 1:
        T_ls, Q_ls = _get_carnot_hpr_cycle_cascade_profile(
            T_ls.tolist(), Q_ls.tolist(), dT_phase_change, is_hot
        )

    sc = StreamCollection()
    for i in range(len(Q_ls)):
        sc.add(
            Stream(
                name=f"{prefix}_H{i + 1}" if is_hot else f"{prefix}_C{i + 1}",
                t_supply=T_ls[i] if is_hot else T_ls[i] - dT_phase_change,
                t_target=T_ls[i] - dT_phase_change if is_hot else T_ls[i],
                heat_flow=Q_ls[i],
                dt_cont=dt_cont,  # Shift to intermediate process temperature scale
                is_process_stream=is_process_stream,
            )
        )
    return sc


def _build_simulated_hpr_streams(
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
    Q_amb_hot: float = 0.0,
    Q_amb_cold: float = 0.0,
    args: HPRTargetInputs = None,
) -> StreamCollection:
    """Build ambient-air source/sink stream representation from ambient duty."""
    sc = StreamCollection()
    if Q_amb_hot > tol:
        sc += _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_hot]),
            dt_cont=args.dt_env_cont,
            is_hot=True,
            is_process_stream=True,
            prefix="AIR",
        )
    if Q_amb_cold > tol:
        sc += _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_cold]),
            dt_cont=args.dt_env_cont,
            is_hot=False,
            is_process_stream=True,
            prefix="AIR",
        )
    return sc


def _calc_obj(
    work: float,
    Q_ext_heat: float,
    Q_ext_cold: float,
    Q_hpr_target: float,
    heat_to_power_ratio: float = 1.0,
    cold_to_power_ratio: float = 0.0,
    penalty: float = 0.0,
) -> float:
    """Return the normalized objective value."""
    return (
        work
        + (Q_ext_heat * heat_to_power_ratio)
        + (Q_ext_cold * cold_to_power_ratio)
        + penalty
    ) / Q_hpr_target


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
    HPRcycle.Brayton.value: _optimise_brayton_heat_pump_placement,
    HPRcycle.MultiTempCarnot.value: _optimise_multi_temperature_carnot_heat_pump_placement,
    HPRcycle.MultiSimpleVapourComp.value: _optimise_multi_simple_heat_pump_placement,
    HPRcycle.CascadeVapourComp.value: _optimise_cascade_heat_pump_placement,
    HPRcycle.MultiSimpleCarnot.value: _optimise_multi_simple_carnot_heat_pump_placement,
}

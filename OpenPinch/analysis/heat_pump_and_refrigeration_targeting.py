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
    zone_config.HPR_TYPE = HPRcycle.MultiSimpleVapourComp.value
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
        T_ls_pt = pt[PT.T.value].to_list()
        T_ls_pt_air = pt_air[PT.T.value].to_list()
        pt_air.insert_temperature_interval(T_ls_pt)
        pt.insert_temperature_interval(T_ls_pt_air)
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
        s = _create_stream_collection_of_background_profile(T_arr, np.abs(H_arr))
        if is_cold:
            T_cold, H_cold, z_amb_cold, s_cold = T_arr, H_arr, z_amb_arr, s
        else:
            T_hot, H_hot, z_amb_hot, s_hot = T_arr, H_arr, z_amb_arr, s
    inputs = {
        "Q_hpr_target": Q_hpr_target,
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
        "eta_ii_he_carnot": zone_config.ETA_II_HE_CARNOT,
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
    # Prepare the initial guess and bounds
    x0_ls = [0.0 for _ in range(args.n_cond + args.n_evap + 1)]
    bnds = [(0.0, 1.0) for _ in range(args.n_cond + args.n_evap)] + [(-1.0, 10.0)]

    # Solve the placement problem
    res = _solve_hpr_placement(
        f_obj=_compute_multi_temperature_carnot_cycle_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )

    # Post-process the results
    res.update(
        _get_carnot_hpr_cycle_streams(
            res["T_cond"], res["Q_cond"], res["T_evap"], res["Q_evap"], args
        )
    )
    return HPRTargetOutputs.model_validate(res)


def _parse_multi_temperature_carnot_cycle_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compile the full list of condenser and evaporator temperature levels and ambient source/sink."""
    T_cond = _map_x_arr_to_T_arr(x[: int(args.n_cond)], args.T_cold[0], args.T_cold[-1])
    T_evap = _map_x_arr_to_T_arr(
        x[int(args.n_cond) : -1], args.T_hot[-1], args.T_hot[0]
    )
    Q_amb_hot = min(args.Q_hpr_target * x[-1], 0.0) * -1
    Q_amb_cold = max(args.Q_hpr_target * x[-1], 0.0)
    return T_cond, T_evap, Q_amb_hot, Q_amb_cold


def _estimate_multi_t_carnot_cycle_perf(
    T_cond: np.ndarray,
    Q_cond: np.ndarray,
    T_evap: np.ndarray,
    Q_evap: np.ndarray,
    *,
    eta_ii_hp: float = 0.5,
    eta_ii_he: float = 0.5,
) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    """Estimate COP by scaling the Carnot limit using entropic mean temperatures."""
    work_gen = 0.0
    is_negative_lift = (
        (np.subtract.outer(T_cond, T_evap) < 0.0)
        & (Q_cond[:, np.newaxis] > 0.0)
        & (Q_evap > 0.0)
    )
    if np.any(is_negative_lift):
        idx_c, idx_e = np.nonzero(is_negative_lift)
        T_h = _compute_entropic_mean_temperature(T_evap[idx_e], Q_evap[idx_e])
        T_l = _compute_entropic_mean_temperature(T_cond[idx_c], Q_cond[idx_c])
        eta_he = eta_ii_he * (1 - T_l / T_h) if T_h > tol else 0.0
        work_gen_from_evap = Q_evap[idx_e].sum() * eta_he
        work_gen_from_cond = Q_cond[idx_c].sum() * eta_he / (1 - eta_he)
        if work_gen_from_evap <= tol or work_gen_from_cond <= tol:
            scale_c = min(Q_evap[idx_e].sum() / Q_cond[idx_c].sum(), 1)
            scale_e = min(Q_cond[idx_c].sum() / Q_evap[idx_e].sum(), 1)
        else:
            # Scale the side that has excess duty available
            scale_c = min(work_gen_from_evap / work_gen_from_cond, 1)
            scale_e = min(work_gen_from_cond / work_gen_from_evap, 1)
            work_gen = min(work_gen_from_evap, work_gen_from_cond)
        Q_cond[idx_c] *= 1 - scale_c
        Q_evap[idx_e] *= 1 - scale_e

    T_h = _compute_entropic_mean_temperature(T_cond, Q_cond)
    T_l = _compute_entropic_mean_temperature(T_evap, Q_evap)
    cop = T_l / (T_h - T_l) * eta_ii_hp + 1 if (T_h - T_l) > 0.0 else np.inf

    # Determine the work of the heat pump based on the limiting side
    work_from_evap = Q_evap.sum() / (cop - 1)
    work_from_cond = Q_cond.sum() / cop
    work_use = min(work_from_evap, work_from_cond)

    # Scale the side that has excess duty available
    Q_cond *= min(work_from_evap / work_from_cond, 1) if work_from_cond > 0.0 else 0.0
    Q_evap *= min(work_from_cond / work_from_evap, 1) if work_from_evap > 0.0 else 0.0
    return work_use, work_gen, Q_cond, Q_evap, cop


def _compute_multi_temperature_carnot_cycle_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate compressor work for a candidate multi-temperature Carnot HP placement defined by vector `x`."""
    T_cond, T_evap, Q_amb_hot, Q_amb_cold = (
        _parse_multi_temperature_carnot_cycle_state_variables(x, args)
    )

    H_cold_with_amb = args.H_cold + args.z_amb_cold * Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * Q_amb_hot

    Q_cond = _get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )
    Q_evap = _get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap, args.T_hot, H_hot_with_amb, is_cond=False
    )

    work_use, work_gen, Q_cond, Q_evap, cop = _estimate_multi_t_carnot_cycle_perf(
        T_cond,
        Q_cond,
        T_evap,
        Q_evap,
        eta_ii_hp=args.eta_ii_hpr_carnot,
        eta_ii_he=args.eta_ii_he_carnot,
    )
    work = work_use - work_gen

    # Determine external heating and cooling demand
    Q_ext_heat = max(np.abs(H_cold_with_amb[0]) - Q_cond.sum(), 0.0)
    Q_ext_cold = max(np.abs(H_hot_with_amb[-1]) - Q_evap.sum(), 0.0)

    # Determine the key performance metrics of the heat pump
    p = (
        g_ineq_penalty(
            g=args.Q_hpr_target - Q_evap.sum(),
            rho=args.rho_penalty * 10,
            form="square",
        )
        if not (args.is_heat_pumping)
        else 0.0
    )

    obj = _calc_obj(
        work=work_use - work_gen,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:  # If in debug mode, plot the graph immediately
        res = _get_carnot_hpr_cycle_streams(T_cond, Q_cond, T_evap, Q_evap, args)
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
        "net_work": work,
        "work_use": work_use,
        "work_gen": work_gen,
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": T_cond,
        "Q_cond": Q_cond,
        "T_evap": T_evap,
        "Q_evap": Q_evap,
        "cop_h": cop,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
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

    # Prepare the initial guess and bounds
    bnds = _get_bounds_for_cascade_hp_opt(args)

    x0 = (
        _get_x0_for_cascade_hp_opt(
            init_res=init_res,
            args=args,
            bnds=bnds,
        )
        if args.initialise_simulated_cycle
        else None
    )
    # Solve the placement problem
    res = _solve_hpr_placement(
        f_obj=_compute_cascade_hp_system_obj,
        x0_ls=x0,
        bnds=bnds,
        args=args,
    )
    return HPRTargetOutputs.model_validate(res)


def _get_x0_for_cascade_hp_opt(
    init_res: HPRTargetOutputs,
    args: HPRTargetInputs,
    bnds: list,
) -> np.ndarray:
    """Build initial guess vectors for the cascade condenser/evaporator optimization."""
    T_cond = init_res.T_cond
    Q_heat = init_res.Q_cond
    T_evap = init_res.T_evap
    Q_cool = init_res.Q_evap

    x_sc_bnds = [
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    ]

    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds[1] = x_sc_bnds[0]

    x_cond = (args.T_cold[0] - T_cond) / args.dt_range_max
    x_sc = np.array([(x_sc_bnds[0] + x_sc_bnds[1]) / 2 for _ in range(len(T_cond))])
    y_heat = Q_heat / args.Q_hpr_target
    x_evap = (T_evap - args.T_hot[-1]) / args.dt_range_max
    y_cool = Q_cool / args.Q_hpr_target

    x_ls = []
    for candidate in [x_cond, x_sc, y_heat, x_evap]:
        x_ls.extend(candidate[:])
    if y_cool.size > 1:
        x_ls.extend(y_cool[:-1])

    x0 = np.array(x_ls)

    if x0.size != len(bnds):
        raise ValueError(
            "Bounds size must match x0 size for cascade heat pump optimisation."
        )

    for i in range(x0.size):
        if not (bnds[i][0] <= x0[i] <= bnds[i][1]):
            x0[i] = (bnds[i][1] + bnds[i][0]) / 2
    return np.asarray([x0])


def _get_bounds_for_cascade_hp_opt(
    args: HPRTargetInputs,
) -> list:
    """Build bounds for the cascade condenser/evaporator optimization."""
    ref_temp_limits = {}

    def _get_ref_temp_limits(refrigerant: str) -> tuple[float, float]:
        limits = ref_temp_limits.get(refrigerant)
        if limits is None:
            limits = (
                PropsSI("Tmin", refrigerant) - 273.15,
                PropsSI("Tmax", refrigerant) - 273.15,
            )
            ref_temp_limits[refrigerant] = limits
        return limits

    def _get_cond_bounds(refrigerant: str) -> tuple[float, float]:
        T_min, T_max = _get_ref_temp_limits(refrigerant)
        T_lo = max(args.T_cold[-1] + args.dt_phase_change, T_min + 1)
        T_hi = max(min(args.T_cold[0], T_max - 1), T_lo)
        return (
            (args.T_cold[0] - T_hi) / args.dt_range_max,
            (args.T_cold[0] - T_lo) / args.dt_range_max,
        )

    def _get_evap_bounds(refrigerant: str) -> tuple[float, float]:
        T_min, T_max = _get_ref_temp_limits(refrigerant)
        T_lo = max(args.T_hot[-1], T_min + 1)
        T_hi = max(min(args.T_hot[0] - args.dt_phase_change, T_max - 1), T_lo)
        return (
            (T_lo - args.T_hot[-1]) / args.dt_range_max,
            (T_hi - args.T_hot[-1]) / args.dt_range_max,
        )

    x_cond_bnds = [
        _get_cond_bounds(refrigerant)
        for refrigerant in args.refrigerant_ls[: args.n_cond]
    ]
    x_evap_bnds = [
        _get_evap_bounds(refrigerant)
        for refrigerant in args.refrigerant_ls[
            args.n_cond - 1 : args.n_cond - 1 + args.n_evap
        ]
    ]

    x_sc_bnds = (
        np.float64(args.dt_phase_change / args.dt_range_max),
        np.float64((args.T_cold[0] - args.T_cold[-1]) / args.dt_range_max),
    )
    if x_sc_bnds[0] > x_sc_bnds[1]:
        x_sc_bnds = (x_sc_bnds[0], x_sc_bnds[0])

    y_heat_bnds = (0.0, 1.0)
    y_cool_bnds = (0.0, 1.0)

    bnds = (
        x_cond_bnds
        + [x_sc_bnds] * args.n_cond
        + [y_heat_bnds] * args.n_cond
        + x_evap_bnds
    )
    if args.n_evap > 1:
        bnds += [y_cool_bnds] * (args.n_evap - 1)

    return bnds


def _parse_cascade_hp_state_variables(
    x: np.ndarray,
    args: HPRTargetInputs,
) -> Tuple[np.ndarray]:
    """Extract HP variables from optimization vector x."""
    n0, n1 = args.n_cond, args.n_evap
    T_cond = np.array(args.T_cold[0] - x[:n0] * args.dt_range_max, dtype=np.float64)
    dT_subcool = np.array(x[n0 : 2 * n0] * args.dt_range_max, dtype=np.float64)
    Q_heat = np.array(x[2 * n0 : 3 * n0] * args.Q_hpr_target, dtype=np.float64)
    T_evap = np.array(
        x[3 * n0 : 3 * n0 + n1] * args.dt_range_max + args.T_hot[-1], dtype=np.float64
    )
    temp = np.array(x[3 * n0 + n1 : -1] * args.Q_hpr_target, dtype=np.float64)
    Q_cool = (
        np.concatenate([temp.tolist(), [np.nan]]) if temp.size > 0 else np.array(np.nan)
    )
    Q_amb_hot = min(args.Q_hpr_target * x[-1], 0.0) * -1
    Q_amb_cold = max(args.Q_hpr_target * x[-1], 0.0)    
    return T_cond, dT_subcool, Q_heat, T_evap, Q_cool, Q_amb_hot, Q_amb_cold


def _compute_cascade_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Objective: minimise total compressor work for multi-HP configuration."""
    T_cond, dT_subcool, Q_heat, T_evap, Q_cool = _parse_cascade_hp_state_variables(
        x, args
    )

    hp = CascadeVapourCompressionCycle()
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
    if not (hp.solved):
        obj = hp.work / args.Q_hpr_target + 1
        return {"obj": obj, "success": False}

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
        cold_streams=args.bckgrd_cold_streams,
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams,
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    # Calculate key perfromance indicators
    work_hp = hp.work
    cop = args.Q_hpr_target / work_hp if work_hp > 0 else 1.0
    Q_amb = _calc_Q_amb(hp.Q_cool, np.abs(args.H_hot[-1]), args.Q_amb_max)

    Q_ext = pt_cond.col[PT.H_NET.value][0]  # e.g., direct eletric heating
    g = np.array(
        [pt_evap.col[PT.H_NET.value][0], pt_cond.col[PT.H_NET.value][-1], hp.penalty]
    )
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    obj = _calc_obj(
        work=work_hp,
        Q_ext_heat=Q_ext,
        Q_ext_cold=0.0,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:  # for debugging purposes, a quick plot function
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            args.H_hot,
            args.T_cold,
            args.H_cold,
            hpr_hot_streams,
            hpr_cold_streams,
            title=f"Obj {float(obj):.5f} = {(work_hp / args.Q_hpr_target):.5f} + {(Q_ext / args.Q_hpr_target):.5f} + {(g.sum() / args.Q_hpr_target):.5f}",
        )

    return {
        "obj": obj,
        "utility_tot": work_hp + Q_ext,
        "net_work": hp.work_arr,
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": hp.Q_heat_arr,
        "T_evap": T_evap,
        "Q_cool": hp.Q_cool_arr,
        "cop_h": cop,
        "Q_amb_hot": Q_amb if args.is_heat_pumping else 0.0,
        "Q_amb_cold": 0.0 if args.is_heat_pumping else Q_amb,
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
    Q_amb_hot = min(args.Q_hpr_target * x[-1], 0.0) * -1
    Q_amb_cold = max(args.Q_hpr_target * x[-1], 0.0)
    return T_cond, T_evap, Q_amb_hot, Q_amb_cold


def _compute_multi_simple_carnot_hp_opt_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    *,
    debug: bool = False,
) -> dict:
    """Evaluate compressor work for a candidate multiple single Carnot HP placement defined by vector `x`."""
    T_cond, T_evap, Q_amb_hot, Q_amb_cold = (
        _parse_multi_simple_carnot_hp_state_variables(x, args)
    )
    # Adjust the background profiles to account for ambient heat exchange
    H_cold_with_amb = args.H_cold + args.z_amb_cold * Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * Q_amb_hot

    # Calculate the target heat for each condenser temperature level from the background profiles
    Q_cond = _get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_cond, args.T_cold, H_cold_with_amb, is_cond=True
    )

    # Determine the scaled work of each heat pump (or heat engine)
    delta_T_lift = T_cond - T_evap
    is_hp = delta_T_lift > 0.0
    is_he = (delta_T_lift < 0.0) & args.allow_integrated_expander

    Q_evap = np.zeros_like(Q_cond)
    work_hp = np.zeros_like(Q_cond)
    work_he = np.zeros_like(Q_cond)
    q_diff = []

    if np.any(is_hp):
        cop_hp = (T_evap[is_hp] + 273.15) / delta_T_lift[is_hp] * args.eta_ii_hpr_carnot + 1
        work_hp[is_hp] = Q_cond[is_hp] / cop_hp
        Q_evap[is_hp] = Q_cond[is_hp] - work_hp[is_hp]

        # Sort and unsort evaporator temperatures and corresponding Q values to ensure correct pairing for work calculation
        sort_idx = np.argsort(T_evap)[::-1]
        Q_allocated = 0.0
        for idx in sort_idx:
            Q_available = (
                _get_Q_vals_at_T_hpr_from_bckgrd_profile(
                    np.array([T_evap[idx]]), args.T_hot, H_hot_with_amb, is_cond=False
                )
                - Q_allocated
            )
            if Q_available.sum() > Q_evap[idx]:
                Q_allocated += Q_evap[idx]
            elif Q_evap[idx] > 0.0:
                scale = Q_available.sum() / Q_evap[idx]
                q_diff.append(Q_evap[idx] - Q_available.sum())
                Q_evap[idx] *= scale
                Q_cond[idx] *= scale
                work_hp[idx] *= scale
                Q_allocated += Q_available.sum()
            else:
                Q_evap[idx] = 0.0
                Q_cond[idx] = 0.0
                work_hp[idx] = 0.0

    if np.any(is_he):
        eff_he = (delta_T_lift[is_he] / T_cond[is_he]) * args.eta_ii_he_carnot
        work_he[is_he] = Q_cond[is_he] * eff_he

    # Calculate evaporator duty
    work = work_hp - work_he
    cop = Q_cond.sum() / work_hp.sum() if work_hp.sum() > tol else 0
    eta_he = work_he.sum() / (Q_cond.sum() + 1e-6) if Q_cond.sum() != 0 else 0

    # Determine external heating and cooling demand
    Q_ext_heat = max(np.abs(H_cold_with_amb[0]) - Q_cond.sum(), 0.0)
    Q_ext_cold = max(np.abs(H_hot_with_amb[-1]) - Q_evap.sum(), 0.0)

    # Determine the key performance metrics of the heat pump
    p = g_ineq_penalty(g=q_diff, rho=args.rho_penalty, form="square")

    # Calculate the objective function value
    obj = _calc_obj(
        work=work_hp.sum() - work_he.sum(),
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    if debug:  # If in debug mode, plot the graph immediately
        res = _get_carnot_hpr_cycle_streams(T_cond, Q_cond, T_evap, Q_evap, args)
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            H_hot_with_amb,
            args.T_cold,
            H_cold_with_amb,
            res["hpr_hot_streams"],
            res["hpr_cold_streams"],
            title=f"Obj {float(obj):.5f} = {(work.sum() / args.Q_hpr_target):.5f} + {(Q_ext_heat / args.Q_hpr_target):.5f} + {(Q_ext_cold / args.Q_hpr_target):.5f} + {(p / args.Q_hpr_target):.5f}",
        )

    return {
        "obj": obj,
        "utility_tot": work.sum() + Q_ext_heat,
        "net_work": work,
        "Q_ext": Q_ext_heat + Q_ext_cold,
        "T_cond": T_cond,
        "Q_cond": Q_cond,
        "T_evap": T_evap,
        "Q_evap": Q_evap,
        "cop_h": cop,
        "eta_he": eta_he,
        "Q_amb_hot": Q_amb_hot,
        "Q_amb_cold": Q_amb_cold,
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
    Q_cond = np.array(x[2 * n : 3 * n] * args.Q_hpr_target, dtype=np.float64)
    T_evap = np.array(
        x[3 * n : 4 * n] * args.dt_range_max + args.T_hot[-1], dtype=np.float64
    )
    dT_superheat = np.array(x[4 * n :] * args.dt_range_max, dtype=np.float64)
    return T_cond, dT_subcool, Q_cond, T_evap, dT_superheat


def _compute_multi_simple_hp_system_obj(
    x: np.ndarray,
    args: HPRTargetInputs,
    debug: bool = False,
) -> dict:
    """Objective: minimize total compressor work for multi-HP configuration."""
    T_cond, dT_subcool, Q_heat, T_evap, dT_superheat = (
        _parse_multi_simple_hp_state_temperatures(x, args)
    )

    T_diff = T_cond - dT_subcool - T_evap - args.dtcont_hp
    if T_diff.min() < 0:
        return {"obj": np.inf, "success": False}

    hp = ParallelVapourCompressionCycles()
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
        cold_streams=args.bckgrd_cold_streams,
        is_shifted=True,
    )
    pt_evap = get_process_heat_cascade(
        hot_streams=args.bckgrd_hot_streams,
        cold_streams=hpr_cold_streams,
        is_shifted=True,
    )

    # Calculate key perfromance indicators
    work_hp = hp.work
    cop = args.Q_hpr_target / work_hp if work_hp > 0 else 1.0
    Q_amb = _calc_Q_amb(hp.Q_cool, np.abs(args.H_hot[-1]), args.Q_amb_max)

    g = np.array(
        [pt_evap.col[PT.H_NET.value][0], pt_cond.col[PT.H_NET.value][-1], hp.penalty]
    )
    p = g_ineq_penalty(g, eta=args.eta_penalty, rho=args.rho_penalty, form="square")
    Q_ext = pt_cond.col[PT.H_NET.value][
        0
    ]  # Alternative to heat pump, an external heat source
    obj = _calc_obj(
        work=work_hp,
        Q_ext_heat=Q_ext,
        Q_ext_cold=0.0,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=p,
    )

    # For debugging purposes, a quick plot function
    if debug:
        plot_multi_hp_profiles_from_results(
            args.T_hot,
            args.H_hot,
            args.T_cold,
            args.H_cold,
            hpr_hot_streams,
            hpr_cold_streams,
            title=f"Obj {float(obj):.5f} = {(work_hp / args.Q_hpr_target):.5f} + {(Q_ext / args.Q_hpr_target):.5f} + {(g.sum() / args.Q_hpr_target):.5f}",
        )

    return {
        "obj": obj,
        "utility_tot": work_hp + Q_ext,
        "net_work": hp.work_arr,
        "Q_ext": Q_ext,
        "T_cond": T_cond,
        "dT_subcool": dT_subcool,
        "Q_heat": hp.Q_heat_arr,
        "T_evap": T_evap,
        "dT_superheat": dT_superheat,
        "Q_cool": hp.Q_cool_arr,
        "cop_h": cop,
        "Q_amb_hot": Q_amb if args.is_heat_pumping else 0.0,
        "Q_amb_cold": 0.0 if args.is_heat_pumping else Q_amb,
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
    x: np.ndarray, args: HPRTargetInputs
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

    work_hp = sum([hp.work_net for hp in hp_list])
    c = (
        pt_gas_cooler.col[PT.H_NET.value][-1] + pt_gas_heater.col[PT.H_NET.value][0]
    ) * 10  # Penalty for supplying too much heat
    Q_ext = pt_gas_cooler.col[PT.H_NET.value][
        0
    ]  # Extra heating required on either side of the gcc
    Q_cool = np.array([hp.Q_cool for hp in hp_list])
    COP = (args.Q_hpr_target - Q_ext) / (work_hp + 1e-9)
    Q_amb = _calc_Q_amb(Q_cool.sum(), np.abs(args.H_hot[-1]), args.Q_amb_max)
    obj = _calc_obj(
        work=work_hp,
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
        "utility_tot": work_hp + Q_ext,
        "net_work": work_hp,
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
    x0_ls = np.asarray(x0_ls, dtype=np.float64)
    if len(x0_ls.shape) == 1:
        x0_ls = [x0_ls]

    local_minima_x = multiminima(
        func=f_obj,
        func_kwargs=args,
        x0_ls=x0_ls,
        bounds=bnds,
        optimiser_handle=args.bb_minimiser,
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
    is_source: bool = True,
) -> StreamCollection:
    """Constructs net stream segments that require utility input across temperature intervals."""
    s = StreamCollection()

    T_vals = np.asarray(T_vals)
    H_vals = np.asarray(H_vals)

    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    dh_vals = delta_vals(H_vals)

    for i, dh in enumerate(dh_vals):
        if dh > tol and not (is_source):
            s.add(
                Stream(
                    t_supply=T_vals[i + 1],
                    t_target=T_vals[i],
                    heat_flow=dh,
                )
            )
        elif -dh > tol and is_source:
            s.add(
                Stream(
                    t_supply=T_vals[i],
                    t_target=T_vals[i + 1],
                    heat_flow=-dh,
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


def _map_x_arr_to_T_arr(
    x: np.ndarray,
    T_0: float,
    T_1: float,
) -> np.ndarray:
    """Recover evaporator temperatures from normalized x values."""
    temp = []
    for i in range(x.size):
        temp.append(T_0 - x[i] * (T_0 - T_1))
        T_0 = temp[-1]
    return np.sort(np.array(temp).flatten())[::-1]


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
    Q_amb_hot: float,
    Q_amb_cold: float,
    args: HPRTargetInputs,
) -> StreamCollection:
    """Build ambient-air source/sink stream representation from ambient duty."""
    if Q_amb_hot > tol:
        return _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_hot]),
            dt_cont=args.dt_env_cont,
            is_hot=True,
            is_process_stream=True,
            prefix="AIR",
        )
    elif Q_amb_cold > tol:
        return _build_latent_streams(
            T_ls=np.array([args.T_env]),
            dT_phase_change=args.dt_phase_change,
            Q_ls=np.array([Q_amb_cold]),
            dt_cont=args.dt_env_cont,
            is_hot=False,
            is_process_stream=True,
            prefix="AIR",
        )
    else:
        return StreamCollection()


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

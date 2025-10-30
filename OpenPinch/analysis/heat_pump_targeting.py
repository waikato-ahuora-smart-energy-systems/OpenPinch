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
    n_cond: int,
    n_evap: int,
    eff_isen: float = 0.7,
    dtmin_hp: float = 5.0,
    is_T_vals_shifted: bool = True,
    zone_config: Configuration = None,
    
):
    init_res = _initialise_heat_pump_temperatures(
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
    )

    return None


#######################################################################################################
# Helper functions
#######################################################################################################

def _initialise_heat_pump_temperatures(
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
):
    H_hot, H_cold = _balance_hot_and_cold_heat_loads_with_ambient_air(T_hot, H_hot, T_cold, H_cold, zone_config)
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
            fun=lambda x: _get_heat_pump_placement_performance(x, args),
            x0=x0,
            method="COBYQA",
            bounds=bnds,
            # constraints=None,
        )
        var_x = res.x
        
    total_work = _get_heat_pump_placement_performance(var_x, args)
    T_cond, T_evap = _get_all_cond_and_evap_temperature_levels(var_x, args)
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold)
    H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot)

    return {
        "T_cond": T_cond, 
        "H_cond": H_cond, 
        "T_evap": T_evap, 
        "H_evap": H_evap, 
        "total_work": total_work,
    }


def _get_Q_from_H(H: np.ndarray):
    H0 = H * -1 if H.min() < -tol else H
    temp = np.roll(H0, -1)
    temp[-1] = 0
    return H0 - temp    


def _get_entropic_average_temperature(
    T: np.ndarray,
    H: np.ndarray,        
):
    if T.var() < tol:
        return T[0]

    Q = _get_Q_from_H(H)
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
    T_lo = _get_entropic_average_temperature(T_evap, H_evap)
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
    H_cond = _get_H_vals_from_T_hp_vals(T_cond, args.T_cold, args.H_cold)
    H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot)

    def _get_optimal_min_evap_T(T_lo):
        for i in range(len(T_evap)):
            if (T_lo > T_evap[i]) or (abs(T_lo - T_evap[i]) > tol and i == 0):
                T_evap[i] = T_lo
            else:
                T_evap[i] = T_evap0[i]  
        H_evap = _get_H_vals_from_T_hp_vals(T_evap, args.T_hot, args.H_hot)
        cop = _get_carnot_COP(T_cond, H_cond, T_evap, H_evap)
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
            # full_output=True,
        )
        pass
    except: 
        if abs(args.bnds_evap[0] - args.bnds_evap[1]) < 0.1:
            res = args.bnds_evap[0]
        else:
            res = args.bnds_evap[0]
    
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


def _convert_idx_to_temperatures(idx_dict: dict, T_hot: np.ndarray, T_cold: np.ndarray) -> dict:
    return {
        "HU": (T_cold[idx_dict["HU"][0]], T_cold[idx_dict["HU"][1]]),
        "CU": (T_hot[idx_dict["CU"][0]], T_hot[idx_dict["CU"][1]]),
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


def _apply_temperature_shift_for_heat_pump_stream_dtmin_cont(
    T_hot: np.ndarray,    
    T_cold: np.ndarray, 
    dtmin_hp: float,
    is_T_vals_shifted: bool,
):
    T_hot  -= dtmin_hp / 2 if is_T_vals_shifted else dtmin_hp
    T_cold += dtmin_hp / 2 if is_T_vals_shifted else dtmin_hp

    return T_hot, T_cold

def _balance_hot_and_cold_heat_loads_with_ambient_air(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,    
    H_cold: np.ndarray, 
    zone_config: Configuration,       
):
    if zone_config == None:
        return H_hot, H_cold
    
    if H_hot.max() > tol:
        H_hot *= -1

    delta_H = np.abs(H_cold).max() - np.abs(H_hot).max()
    if delta_H > 0.0:
        mask = np.where(
            T_hot <= (zone_config.T_ENV - zone_config.DT_ENV_CONT - zone_config.DT_PHASE_CHANGE),
            delta_H, 0.0
        )
        H_hot -= mask

    elif delta_H < 0.0:
        # Case of refrigeration --> not currently a focus.
        pass

    else:
        pass

    return H_hot, H_cold   






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


# ============================================================
# ─── SUPPORT FUNCTIONS ──────────────────────────────────────
# ============================================================

def get_fluid_T_limits(fluid, unit_system='EUR'):
    """Return min and critical temperatures of a fluid."""
    T_min, T_crit = PropsSI('Tmin', fluid), PropsSI('Tcrit', fluid)
    if unit_system == 'EUR':  # Convert to °C if needed
        T_min -= 273.15
        T_crit -= 273.15
    return T_min, T_crit



def parse_hp_variables(x, n_cond, n_evap, Q_total):
    """
    Extract HP variables from optimization vector x.

    - Te, dT_sh: length n_evap
    - Tc, dT_sc: length n_cond
    - Q_cond: length N_hp = max(n_cond, n_evap), last one fills remaining Q_total
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


def create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system):
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
# ============================================================

def objective_multi_hp_work(x, n_cond, n_evap, T_vals, H_hot, H_cold, dtmin_hp,
                            fluid='Ammonia', unit_system='EUR'):
    """Objective: minimize total compressor work for multi-HP configuration."""
    Q_total = max(H_hot)
    Te, dT_sh, Tc, dT_sc, Q_cond = parse_hp_variables(x, n_cond, n_evap, Q_total)

    # Create HP objects with correct temperature mapping
    hp_list = create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system)

    total_work = 0.0
    for i, hp in enumerate(hp_list):
        try:
            total_work += Q_cond[i] / hp.COP_heating()  # use Q_cond directly
        except ValueError:
            return 1e6  # Penalize infeasible points

    print(Q_cond, Te, Tc, total_work)
    return total_work

def mta_constraint_multi(x, n_cond, n_evap, T_vals, H_hot, H_cold, dtmin_hp,
                         fluid='Ammonia', unit_system='EUR'):
    """Ensure minimum temperature approach ≥ dtmin_hp."""
    Q_total = max(H_hot)
    Te, dT_sh, Tc, dT_sc, Q_cond = parse_hp_variables(x, n_cond, n_evap, Q_total)
    hp_list = create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system)

    try:
        min_mta = profiles_crossing_check_multi(T_vals, H_hot, H_cold, hp_list)
    except ValueError:
        min_mta = -1e6  # Infeasible solution

    return min_mta - dtmin_hp


def condenser_vs_evaporator_constraint(x, n_cond, n_evap):
    """Ensure Tc[i] > Te[i % n_evap] + margin."""
    Te = x[:n_evap]
    Tc = x[2 * n_evap:2 * n_evap + n_cond]
    margin = 3.0
    return Tc - np.array([Te[i % n_evap] for i in range(n_cond)]) - margin

def min_Q_cond_constraint(x, n_cond, n_evap, Q_total):
    _, _, _, _, Q_cond = parse_hp_variables(x, n_cond, n_evap, Q_total)
    return Q_cond - 0.05*Q_total  # elementwise, must be >= 0

def profiles_crossing_check_multi(T_vals, H_hot, H_cold, hp_list):
    """
    Check minimum temperature approach (MTA) between source/sink profiles
    and a multi-heat-pump cascade.
    """
    # --- Define source and sink profiles with correct PT keys
    snk_profile = {PT.T.value: T_vals, PT.H_HOT_UT.value: H_hot}
    src_profile = {PT.T.value: T_vals, PT.H_COLD_UT.value: H_cold}

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
            src_profile_cut = cut_profile(
                src_profile, min(cascade[PT.H_COLD_UT.value]), "right", "source"
            )
        else:
            ambient_heat_flow = (
                min(src_profile[PT.H_COLD_UT.value]) - min(cascade[PT.H_COLD_UT.value])
            )
            src_profile_cut = add_ambient_source(src_profile, ambient_heat_flow)
    except Exception:
        return -1e6  # Unbalanced composite curves or data mismatch

    # --- Compute minimum temperature approach
    try:
        deltaTs = _get_min_temperature_approach(cascade, snk_profile, src_profile_cut)
        return min(deltaTs)
    except ValueError:
        return -1e6  # Infeasible temperature overlap


# ============================================================
# ─── OPTIMIZATION ROUTINE ───────────────────────────────────
# ============================================================

def optimize_multi_hp(T_vals, H_hot, H_cold, n_cond, n_evap,
                      dtmin_hp=5.0, fluid='Ammonia', unit_system='EUR'):

    Q_total = max(H_hot)
    T_min, T_crit = get_fluid_T_limits(fluid, unit_system)

    # --- Initial Guess
    x0 = np.array(
        [min(T_vals)] * n_evap +  # Te: all evaporators start at min temp
        [1.0] * n_evap +          # dT_sh: small initial superheat
        [max(T_vals)] * n_cond +  # Tc: all condensers start at max temp
        [1.0] * n_cond +         # dT_sc: initial subcooling guess
        [Q_total / n_cond] * (max(n_cond,n_evap) - 1)  # Q_cond: first N-1 set equally
    )

    # --- Bounds
    bounds = (
        [(T_min + 5, T_crit - 10)] * n_evap +  # Te
        [(0.1, 5)] * n_evap +                # dT_sh
        [(T_min + 10, T_crit - 5)] * n_cond + # Tc
        [(0.1, 100)] * n_cond +               # dT_sc
        [(0.05, Q_total)] * (max(n_cond,n_evap) - 1)         # Q_cond
    )

    # --- Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: mta_constraint_multi(x, n_cond, n_evap, T_vals, H_hot, H_cold, dtmin_hp, fluid, unit_system)},
        {'type': 'ineq', 'fun': lambda x: min_Q_cond_constraint(x, n_cond, n_evap, Q_total)},
        {'type': 'ineq', 'fun': lambda x: condenser_vs_evaporator_constraint(x, n_cond, n_evap)},
    ]

    # --- Optimization
    result = minimize(
        objective_multi_hp_work, x0,
        args=(n_cond, n_evap, T_vals, H_hot, H_cold, dtmin_hp, fluid, unit_system),
        bounds=bounds,
        constraints=constraints,
        method='COBYQA',  
        options={'disp': False, 'maxiter': 1000}
    )

    if result.success:
        Te, dT_sh, Tc, dT_sc, Q_cond = parse_hp_variables(result.x, n_cond, n_evap, Q_total)
        print(f"\n--- Optimization Results ---")
        print(f"Te       = {Te}")
        print(f"dT_sh    = {dT_sh}")
        print(f"Tc       = {Tc}")
        print(f"dT_sc    = {dT_sc}")
        print(f"Q_cond   = {Q_cond}")
        print(f"Total W  = {result.fun:.3f}")
    else:
        print("Optimization failed:", result.message)

    return result


# ============================================================
# ─── VISUALIZATION ──────────────────────────────────────────
# ============================================================

def plot_multi_hp_profiles(T_vals, H_hot, H_cold, x, n_cond, n_evap, 
                           fluid='Ammonia', unit_system='EUR'):
    """Plot HP cascade and source/sink profiles."""
    Te, dT_sh, Tc, dT_sc, Q_cond = parse_hp_variables(x, n_cond, n_evap, Q_total=max(H_hot))
    hp_list = create_heat_pump_list(Te, dT_sh, Tc, dT_sc, Q_cond, n_cond, n_evap, fluid, unit_system)

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

